# Deepfake Detection System

A Streamlit-based web app that detects deepfake images using a **ViT-Base/16** classifier and a **U-Net** segmentation model, with optional **LIME** explainability.

## Models

| Model | File | Purpose |
|-------|------|---------|
| ViT-Base/16 | `vit.pth` | Binary classification (REAL / FAKE) |
| U-Net | `unet.pth` | Tampered region segmentation |

> Place both `.pth` files in the project root before running.

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the app**
```bash
streamlit run deepfake_gui_app.py
```

## Features

- **Image Upload** — analyze any JPG/PNG image
- **Webcam Mode** — real-time frame-by-frame detection (experimental)
- **Segmentation Mask** — U-Net highlights tampered regions with a heat overlay
- **LIME Explainability** — visualizes which image regions influenced the prediction

## Label Convention

| Index | Label |
|-------|-------|
| 0 | FAKE |
| 1 | REAL |

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (falls back to CPU automatically)
- CUDA 13.0 (for GPU acceleration)
