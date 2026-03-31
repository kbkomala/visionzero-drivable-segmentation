# VisionZero — Real-Time Drivable Area Segmentation

> Level 4 Autonomous Vehicle perception pipeline  
> **0.7139 mIoU · 155.8 FPS · 4.2M params · Trained from scratch**

---

## Project Overview

Binary semantic segmentation of drivable vs non-drivable space from front-facing camera frames. Built entirely from scratch on the nuScenes dataset — no pretrained weights used.

**Problem:** Level 4 vehicles must identify free space regardless of lane markings, in complex urban settings including construction zones, puddles, and low-light conditions.

**Solution:** Lightweight encoder-decoder network with MobileNetV3-Small backbone + ASPP bottleneck + U-Net decoder, optimised for real-time inference.

---

## Results

| Metric | Value | Target |
|--------|-------|--------|
| Val mIoU | **0.7139** | > 0.50 |
| Inference FPS | **155.8** | > 30 |
| Latency | **6.4 ms/frame** | < 33 ms |
| Parameters | **~4.2M** | Lightweight |
| ONNX size | **291 KB** | Edge deployable |

---

## Model Architecture

```
Input (512x288x3)
       │
  ┌────┴────────────────────────────┐
  │     MobileNetV3-Small Encoder   │
  │  E1(16ch) E2(24ch) E3(48ch) E4(96ch)
  └────┬──────┬──────┬──────────────┘
       │      │      │         │
    skip   skip   skip    ASPP Bottleneck
    conn   conn   conn    (128ch, dilation 1,6,12)
       │      │      │         │
  ┌────┴──────┴──────┴─────────┘
  │     U-Net Decoder              │
  │  D1(96ch) D2(64ch) D3(32ch) D4(16ch)
  └────┬───────────────────────────┘
       │
  1x1 Conv Head → Sigmoid
       │
  Output Mask (512x288x1)
  0 = non-drivable  |  1 = drivable
```

**Key design choices:**
- Depthwise separable convolutions in decoder (~8x cheaper than standard conv)
- Bilinear upsample (no checkerboard artifacts vs transpose conv)
- Squeeze-excitation channel attention in encoder stages 3 and 4
- ASPP captures road boundaries at multiple scales

---

## Dataset

**nuScenes v1.0-mini** — 10 scenes, 404 CAM_FRONT images  
Drivable area masks generated using `NuScenesMap` API by projecting the `drivable_area` vector layer onto the image plane using camera intrinsics + ego pose.

Download: https://www.nuscenes.org/nuscenes#download  
Map expansion: Required (download separately from same page)

Expected folder structure:
```
nuscenes_data/
├── maps/
│   └── expansion/
│       ├── boston-seaport.json
│       └── singapore-onenorth.json
├── samples/
├── sweeps/
└── v1.0-mini/
```

---

## Setup & Installation

```bash
git clone https://github.com/[your-username]/visionzero-drivable-segmentation
cd visionzero-drivable-segmentation
pip install -r requirements.txt
```

**requirements.txt:**
```
nuscenes-devkit
opencv-python-headless
albumentations
pyquaternion
torch>=2.0
onnxruntime-gpu
onnx
```

---

## How to Run

### Phase 1 — Generate masks from nuScenes
```python
# Set your data path in phase1_pipeline.py
NUSCENES_ROOT = '/path/to/nuscenes_data'
OUTPUT_DIR    = '/path/to/output/masks'

# Run all cells top to bottom
# Output: images/ masks/ pairs.json saved to OUTPUT_DIR
```

### Phase 2 — Train the model
```python
# Set paths in phase2_model.py CFG dict
CFG = {
    'output_dir' : '/path/to/output/masks',
    'save_dir'   : '/path/to/checkpoints',
}
# Run all cells — trains for 60 epochs with early stopping
# Output: best_model.pth saved to save_dir
```

### Phase 3 — Inference + evaluation
```python
# Set paths in phase3_inference.py
CKPT_PATH = '/path/to/checkpoints/best_model.pth'
# Run all cells
# Output: inference_demo.png, edge_case_analysis.png, drivable_space_net.onnx
```

### Quick inference on a single image
```python
from phase3_inference import DrivableSpaceInference
import cv2

pipeline = DrivableSpaceInference('best_model.pth')
frame    = cv2.imread('your_image.jpg')
mask     = pipeline.predict(frame)       # binary mask H x W
overlay  = pipeline.overlay(frame, mask) # BGR with green road overlay
cv2.imwrite('result.jpg', overlay)
```

---

## Training Strategy

| Parameter | Value |
|-----------|-------|
| Loss | Dice (0.5) + weighted BCE (0.5) |
| pos_weight | 14.0 (compensates 6.6% drivable pixels) |
| Optimizer | AdamW lr=1e-3 wd=1e-4 |
| Schedule | 5-epoch warmup + cosine decay |
| Precision | FP16 mixed precision |
| Batch size | 16 |
| Epochs | 60 with early stopping (patience=12) |
| Hardware | Kaggle P100 GPU |
| Train time | ~35 minutes |

---

## Example Outputs

**inference_demo.png** — Road predictions with green overlay  
**edge_case_analysis.png** — Error maps (green=TP, red=FP, blue=FN)

Both files are in the `outputs/` folder of this repository.

---

## Edge Case Handling

| Edge Case | Strategy |
|-----------|----------|
| Water puddles | Rain augmentation + morphological post-processing |
| Night / low light | Gamma darkening augmentation (gamma 0.3-0.6) |
| Construction zones | Coarse dropout + over-sampling of construction scenes |
| Occlusion | Skip connections preserve boundary detail |
| Motion blur | Motion blur augmentation during training |

---

## File Structure

```
visionzero-drivable-segmentation/
├── phase1_pipeline.py       # nuScenes data pipeline + mask generation
├── phase2_model.py          # Model architecture + training loop
├── phase3_inference.py      # Edge case analysis + ONNX export + inference
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── outputs/
    ├── inference_demo.png   # Visual predictions
    └── edge_case_analysis.png  # Failure analysis
```

---

## Team

**VisionZero** — Computer Vision Challenge 2024  
Real-Time Drivable Area Segmentation for Level 4 Autonomous Vehicles
