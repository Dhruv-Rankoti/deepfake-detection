# deepfake_gui_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Deepfake Detection System — Streamlit GUI
#
# Built to match deepfake-detection-metrics.ipynb exactly:
#   • ViT-Base/16  — EPOCHS=10, TRAIN_LIMIT=None (full 100k)
#   • U-Net        — EPOCHS=5,  trained with nn.DataParallel on 2×T4
#   • Label map    — index 0 = FAKE,  index 1 = REAL
#   • clf transform  — Resize → ToTensor → Normalize([0.5],[0.5])
#   • seg transform  — Resize → ToTensor  (NO normalization, matches notebook)
#   • DataParallel strip — unet.pth saved from DataParallel model, so
#                          'module.' prefix is stripped on load automatically
#
# Requirements:
#   pip install streamlit torch torchvision timm opencv-python lime
#               matplotlib pillow scikit-image numpy
#
# Run:
#   streamlit run deepfake_gui_app.py
#
# Place vit.pth and unet.pth in the same folder before running.
# ─────────────────────────────────────────────────────────────────────────────

import os
import io
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from timm import create_model
import streamlit as st

# ── Optional LIME ─────────────────────────────────────────────────────────────
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG  — must match notebook exactly
# ═════════════════════════════════════════════════════════════════════════════
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Confirmed from notebook cell 11: LABEL_MAP = {1: 'REAL', 0: 'FAKE'}
LABEL_MAP = {0: "FAKE", 1: "REAL"}

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))
VIT_PATH    = os.path.join(WEIGHTS_DIR, "vit.pth")
UNET_PATH   = os.path.join(WEIGHTS_DIR, "unet.pth")


# ═════════════════════════════════════════════════════════════════════════════
# MODELS  — identical to notebook cell 18
# ═════════════════════════════════════════════════════════════════════════════
class ViTClassifier(nn.Module):
    """
    Pretrained ViT-Base/16 fine-tuned for binary classification.
    Output logits: [batch, 2]  — index 0=FAKE, index 1=REAL
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    3-level U-Net — identical architecture to notebook cell 18.
    Output: sigmoid mask [batch, 1, H, W]
    """
    def __init__(self):
        super().__init__()
        self.enc1       = DoubleConv(3,   64)
        self.enc2       = DoubleConv(64,  128)
        self.enc3       = DoubleConv(128, 256)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3        = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3       = DoubleConv(512, 256)
        self.up2        = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2       = DoubleConv(256, 128)
        self.up1        = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec1       = DoubleConv(128,  64)
        self.final      = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))


# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: The notebook trained both models inside nn.DataParallel (2×T4).
# torch.save() on a DataParallel model stores keys with a "module." prefix,
# e.g. "module.enc1.block.0.weight" instead of "enc1.block.0.weight".
# _strip_dataparallel() removes that prefix so the weights load cleanly into
# the plain ViTClassifier / UNet classes defined above.
# ═════════════════════════════════════════════════════════════════════════════
def _strip_dataparallel(state_dict: dict) -> dict:
    """Remove 'module.' prefix from every key (saved by nn.DataParallel)."""
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned[new_key] = v
    return cleaned


@st.cache_resource(show_spinner=False)
def load_models():
    clf = ViTClassifier().to(DEVICE)
    seg = UNet().to(DEVICE)
    clf_loaded = False
    seg_loaded = False

    if os.path.exists(VIT_PATH):
        try:
            sd = torch.load(VIT_PATH, map_location=DEVICE)
            clf.load_state_dict(_strip_dataparallel(sd))
            clf_loaded = True
        except Exception as e:
            st.warning(f"Could not load vit.pth: {e}")

    if os.path.exists(UNET_PATH):
        try:
            sd = torch.load(UNET_PATH, map_location=DEVICE)
            seg.load_state_dict(_strip_dataparallel(sd))
            seg_loaded = True
        except Exception as e:
            st.warning(f"Could not load unet.pth: {e}")

    clf.eval()
    seg.eval()
    return clf, seg, clf_loaded, seg_loaded


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMS  — must match notebook inference transforms exactly
# ─────────────────────────────────────────────────────────────────────────────
# Notebook cell 36:
#   clf_infer_tf = Resize(224) → ToTensor → Normalize([0.5]*3, [0.5]*3)
#   seg_infer_tf = Resize(224) → ToTensor          ← NO normalization
# ═════════════════════════════════════════════════════════════════════════════
clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

seg_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # No Normalize — intentional, matches seg_infer_tf in notebook cell 36
])


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def classify(clf: nn.Module, img: Image.Image):
    """
    Returns:
        label (str)  : 'REAL' or 'FAKE'
        conf  (float): confidence in the predicted label [0, 1]
        probs (array): [p_fake, p_real]
    """
    t = clf_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = clf(t)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred  = int(probs.argmax())
    label = LABEL_MAP[pred]
    conf  = float(probs[pred])
    return label, conf, probs


def segment(seg: nn.Module, img: Image.Image) -> np.ndarray:
    """Returns float32 mask [H, W] in [0, 1]."""
    t = seg_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = seg(t)
    # squeeze handles both [1,1,H,W] and [1,H,W] gracefully
    mask = out.cpu().squeeze().numpy()
    return mask


def build_overlay(img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a hot-colormap heat overlay onto the image using the mask as alpha."""
    heatmap = cm.hot(mask)[:, :, :3]            # [H,W,3] float in [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)
    alpha   = (mask * 0.6).clip(0, 0.6)[:, :, np.newaxis]
    overlay = (
        img_np.astype(np.float32) * (1 - alpha)
        + heatmap.astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def mask_to_heatmap_image(mask: np.ndarray) -> Image.Image:
    """Convert float32 [H,W] mask → PIL RGB image via 'hot' colormap."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(mask, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def run_lime(clf: nn.Module, img_np: np.ndarray, num_samples: int = 300):
    """
    LIME explanation matching notebook cell 38 exactly:
        images (uint8) → /255 → (x-0.5)/0.5 → DEVICE → softmax
    Returns (lime_temp, lime_mask_arr, label_name).
    """
    def predict_fn(images):
        # LIME passes uint8 numpy [N, H, W, 3]
        t = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        t = (t - 0.5) / 0.5      # identical normalization to clf training
        t = t.to(DEVICE)
        clf.eval()
        with torch.no_grad():
            out   = clf(t)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs   # [N, 2]  index 0=FAKE, 1=REAL

    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np, predict_fn, top_labels=2, num_samples=num_samples
    )
    top_label  = explanation.top_labels[0]
    label_name = LABEL_MAP[top_label]
    temp, lime_mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=8, hide_rest=False
    )
    return temp, lime_mask, label_name


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.stApp {
    background-color: #0a0c10;
    color: #c9d1d9;
}
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
h1 {
    font-family: 'Share Tech Mono', monospace !important;
    color: #58a6ff !important;
    letter-spacing: 2px;
}
h2, h3 { color: #e6edf3 !important; }

[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
}

.badge-real {
    display: inline-block;
    background: #0d3b1e;
    color: #3fb950;
    border: 1px solid #3fb950;
    border-radius: 6px;
    padding: 6px 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    letter-spacing: 3px;
}
.badge-fake {
    display: inline-block;
    background: #3b0d0d;
    color: #f85149;
    border: 1px solid #f85149;
    border-radius: 6px;
    padding: 6px 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem;
    letter-spacing: 3px;
}

.conf-track {
    background: #21262d;
    border-radius: 4px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
}
.conf-fill-real { background: #3fb950; border-radius: 4px; height: 10px; }
.conf-fill-fake { background: #f85149; border-radius: 4px; height: 10px; }

hr { border-color: #21262d !important; }

.stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 10px 28px;
    letter-spacing: 1px;
    transition: background 0.2s;
}
.stButton > button:hover { background: #388bfd; }

.stAlert { border-radius: 6px !important; }

.caption {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-align: center;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading models…"):
    clf_model, seg_model, clf_loaded, seg_loaded = load_models()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    mode = st.radio(
        "Input Mode",
        ["📁 Upload Image", "🎥 Webcam (Experimental)"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Model Status")

    if clf_loaded:
        st.success("✅ ViT weights loaded  (`vit.pth`)")
    else:
        st.warning("⚠️ `vit.pth` not found — using random weights")

    if seg_loaded:
        st.success("✅ U-Net weights loaded  (`unet.pth`)")
    else:
        st.warning("⚠️ `unet.pth` not found — using random weights")

    st.markdown(f"🖥️ Device: **{DEVICE.upper()}**")

    st.markdown("---")
    st.markdown("### Analysis Options")

    show_mask = st.checkbox("Show tampered region mask", value=True)
    show_lime = st.checkbox(
        "Show LIME explanation  (slow ~1–2 min)",
        value=False,
        disabled=not LIME_AVAILABLE,
        help="Install the `lime` package to enable." if not LIME_AVAILABLE else "",
    )
    lime_samples = 300   # default matches notebook (num_samples=300)
    if show_lime and LIME_AVAILABLE:
        lime_samples = st.slider("LIME samples", 100, 500, 300, step=50)

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>"
        "Label convention:<br>"
        "model idx&nbsp;0 = FAKE<br>"
        "model idx&nbsp;1 = REAL"
        "</small>",
        unsafe_allow_html=True,
    )

    # ── Trained model metrics for reference ─────────────────────
    st.markdown("---")
    st.markdown("### 📊 Trained Model Metrics")
    st.markdown(
        "<small style='color:#8b949e'>"
        "<b style='color:#e6edf3'>ViT Classifier</b> (test, n=20 000)<br>"
        "Accuracy &nbsp;: 0.9911<br>"
        "F1 Score &nbsp;: 0.9911<br>"
        "ROC-AUC &nbsp;: 0.9995<br>"
        "<br>"
        "<b style='color:#e6edf3'>U-Net Segmentation</b> (val)<br>"
        "Pixel Acc : 92.80%<br>"
        "Mean IoU &nbsp;: 0.6013<br>"
        "Dice Coef : 0.7452<br>"
        "</small>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 DEEPFAKE DETECTION SYSTEM")
st.markdown(
    "<p style='color:#8b949e; font-family:Share Tech Mono,monospace; "
    "font-size:0.85rem; letter-spacing:1px;'>"
    "ViT-Base/16 classifier  ·  U-Net segmentation  ·  LIME explainability"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# RENDER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def render_analysis(image: Image.Image):
    """Run full pipeline and display results for one PIL image."""

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_np      = np.array(img_resized)   # uint8 [H, W, 3]

    # ── Classification ───────────────────────────────────────────
    with st.spinner("Classifying…"):
        t0 = time.time()
        label, conf, probs = classify(clf_model, image)
        clf_time = time.time() - t0

    # ── Segmentation ─────────────────────────────────────────────
    mask     = None
    seg_time = 0.0
    if show_mask:
        with st.spinner("Generating segmentation mask…"):
            t0       = time.time()
            mask     = segment(seg_model, image)
            seg_time = time.time() - t0

    # ── Layout: image | verdict ──────────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### Input Image")
        st.image(image, use_container_width=True)

    with col_res:
        st.markdown("#### Prediction")

        badge_class = "badge-real" if label == "REAL" else "badge-fake"
        st.markdown(
            f'<div style="margin: 12px 0 8px 0;">'
            f'<span class="{badge_class}">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        fill_class = "conf-fill-real" if label == "REAL" else "conf-fill-fake"
        st.markdown(
            f'<div class="conf-track">'
            f'  <div class="{fill_class}" style="width:{conf*100:.1f}%"></div>'
            f'</div>'
            f'<small style="color:#8b949e">'
            f'Confidence: <b style="color:#e6edf3">{conf*100:.1f}%</b>'
            f'</small>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        m1, m2, m3 = st.columns(3)
        m1.metric("FAKE prob",  f"{probs[0]*100:.1f}%")
        m2.metric("REAL prob",  f"{probs[1]*100:.1f}%")
        m3.metric("Infer time", f"{clf_time*1000:.0f} ms")

    st.markdown("---")

    # ── Mask + Overlay ───────────────────────────────────────────
    if show_mask and mask is not None:
        st.markdown("#### 🗺️ Tampered Region Analysis")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(img_resized, use_container_width=True)
            st.markdown('<p class="caption">Original</p>', unsafe_allow_html=True)

        with c2:
            heatmap_img = mask_to_heatmap_image(mask)
            st.image(heatmap_img, use_container_width=True)
            st.markdown(
                f'<p class="caption">Mask  (max={mask.max():.2f} '
                f'· mean={mask.mean():.2f})</p>',
                unsafe_allow_html=True,
            )

        with c3:
            overlay = build_overlay(img_np, mask)
            st.image(overlay, use_container_width=True)
            st.markdown(
                '<p class="caption">Overlay (red = high suspicion)</p>',
                unsafe_allow_html=True,
            )

        st.markdown(f"*U-Net inference: {seg_time*1000:.0f} ms*")
        st.markdown("---")

    # ── LIME ─────────────────────────────────────────────────────
    if show_lime and LIME_AVAILABLE:
        st.markdown("#### 💡 LIME Explainability")
        st.info("Running LIME — this may take 1–2 minutes.")

        with st.spinner(f"Running LIME with {lime_samples} samples…"):
            try:
                t0 = time.time()
                lime_temp, lime_mask_arr, lime_label = run_lime(
                    clf_model, img_np, num_samples=lime_samples
                )
                lime_time = time.time() - t0

                lc1, lc2, lc3 = st.columns(3)

                with lc1:
                    st.image(img_np, use_container_width=True)
                    st.markdown(
                        '<p class="caption">Original</p>',
                        unsafe_allow_html=True,
                    )

                with lc2:
                    boundary_img = mark_boundaries(lime_temp / 255.0, lime_mask_arr)
                    boundary_img = (boundary_img * 255).astype(np.uint8)
                    st.image(boundary_img, use_container_width=True)
                    st.markdown(
                        f'<p class="caption">LIME regions → {lime_label}</p>',
                        unsafe_allow_html=True,
                    )

                with lc3:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(lime_mask_arr, cmap="RdYlGn")
                    ax.axis("off")
                    fig.tight_layout(pad=0)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                    buf.seek(0)
                    st.image(Image.open(buf), use_container_width=True)
                    st.markdown(
                        '<p class="caption">Key regions (green = supports prediction)</p>',
                        unsafe_allow_html=True,
                    )

                st.markdown(f"*LIME inference: {lime_time:.1f} s*")

            except Exception as e:
                st.error(f"LIME failed: {e}")

        st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: UPLOAD IMAGE
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📁 Upload Image":
    st.markdown("### Upload an Image")
    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG, JPEG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        st.markdown(
            f"<small style='color:#8b949e'>File: {uploaded.name} — "
            f"{image.width}×{image.height} px</small>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        if st.button("🔍 Analyze", use_container_width=False):
            render_analysis(image)
    else:
        st.markdown(
            "<div style='border:1px dashed #30363d; border-radius:8px; "
            "padding:40px; text-align:center; color:#8b949e;'>"
            "📂 Drop an image above to begin analysis"
            "</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODE: WEBCAM
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "🎥 Webcam (Experimental)":
    st.markdown("### Webcam — Real-Time Detection")
    st.warning(
        "Webcam mode runs best locally with a GPU. "
        "ViT classifies every frame; U-Net mask updates every 5 frames."
    )

    run_cam = st.checkbox("▶ Start Webcam", value=False)

    frame_placeholder  = st.empty()
    status_placeholder = st.empty()

    cap       = cv2.VideoCapture(0)
    frame_idx = 0
    last_mask = None

    while run_cam:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam. Check permissions.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil   = Image.fromarray(frame_rgb)

        # Classify every frame
        label, conf, probs = classify(clf_model, img_pil)

        # Segment every 5 frames (heavier)
        if frame_idx % 5 == 0 and show_mask:
            last_mask = segment(seg_model, img_pil)
            last_mask = cv2.resize(
                last_mask, (frame_rgb.shape[1], frame_rgb.shape[0])
            )

        # Overlay mask
        display = frame_rgb.copy()
        if last_mask is not None and show_mask:
            display = build_overlay(display, last_mask)

        # Draw label bar directly on frame
        bar_color = (63, 185, 80) if label == "REAL" else (248, 81, 73)
        h, w      = display.shape[:2]
        cv2.rectangle(display, (0, 0), (w, 48), (13, 17, 23), -1)
        cv2.putText(
            display,
            f"{label}  {conf*100:.1f}%",
            (14, 33),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            bar_color,
            2,
        )

        frame_placeholder.image(display, channels="RGB", use_container_width=True)
        status_placeholder.markdown(
            f"Frame {frame_idx} &nbsp;|&nbsp; "
            f"FAKE: {probs[0]*100:.1f}%  &nbsp;  REAL: {probs[1]*100:.1f}%"
        )

        frame_idx += 1
        time.sleep(0.03)   # yield so Streamlit can update checkbox state

    cap.release()

    if not run_cam:
        st.markdown(
            "<div style='border:1px dashed #30363d; border-radius:8px; "
            "padding:30px; text-align:center; color:#8b949e;'>"
            "🎥 Tick the checkbox above to start the webcam"
            "</div>",
            unsafe_allow_html=True,
        )