# Generative Solar Panel Layout Designer

A supervised deep learning system that predicts optimal solar panel layouts for residential rooftops. Given an aerial image of a roof and a set of installation specifications, the model generates a spatial mask showing where panels should be placed.

---

## Overview

Solar panel layout design is currently a manual process performed by engineers who must balance electrical constraints, roof geometry, shading, and local regulations. This project trains a conditional image segmentation model to automate that spatial reasoning.

**Input:**
- Aerial/satellite image of an empty roof
- Installation metadata: number of panels, roof type, string count, connection type

**Output:**
- Binary spatial mask showing panel placement positions
- Predicted panel count (auxiliary output for constraint validation)

---

## How It Works

```
Empty Roof Image (aerial)
        +                   ──►  FiLM-Conditioned U-Net  ──►  Panel Layout Mask
Metadata (panels, roof type,
          strings, phase)
```

The model is a **U-Net** with a **ResNet-34 encoder** pretrained on ImageNet. Installation metadata is fused into the decoder at every scale using **FiLM (Feature-wise Linear Modulation)** — a conditioning technique that applies learned scale and shift to each feature map based on the metadata vector. This forces the model to respect electrical and physical constraints at every level of spatial reasoning.

---

## Dataset

Each training sample consists of:

| Component | Description |
|---|---|
| Raw layout image | Aerial photo with panel overlay drawn on roof |
| Clean roof image | Same photo with panels removed (via inpainting) |
| Binary mask | Panel footprint extracted from the overlay |
| Metadata row | CSV record with installation specifications |

### `data/metadata.csv` — Required Columns

| Column | Type | Example |
|---|---|---|
| `image_filename` | string | `house_001.jpg` |
| `mask_filename` | string | `house_001_mask.png` |
| `num_panels` | int | `13` |
| `roof_type` | string | `tile` / `metal` / `flat` / `other` |
| `num_strings` | int | `1` |
| `connection_type` | string | `single_phase` / `three_phase` |

> Raw images and masks are excluded from version control due to size. Store them locally under `data/raw/`, `data/roofs/`, and `data/masks/`.

---

## Domain Constraints the Model Learns

| Constraint | Rule |
|---|---|
| **String integrity** | `strings=1` → all panels on one roof face (single spatial cluster) |
| **Phase ceiling** | `single_phase` → max 2 strings; `three_phase` → up to 3 |
| **Tile roof mounting** | Portrait orientation preferred, 300mm setback from edges |
| **Roof angle** | 15°–35° pitch → dense packing, no tilt frames needed |
| **Flat roof** | Tilt frames required → inter-row spacing enforced |

---

## Project Structure

```
GenerativeSolarPanelLayoutDesigner/
├── data/
│   ├── raw/                    # original labeled images (panels overlaid)
│   ├── roofs/                  # clean roof images (panels removed)
│   ├── masks/                  # binary panel layout masks
│   └── metadata.csv            # tabular labels — versioned
├── src/
│   ├── dataset.py              # SolarLayoutDataset + DataLoader factory
│   ├── model.py                # FiLM-conditioned U-Net
│   ├── losses.py               # BCE + Dice + panel count loss
│   ├── metrics.py              # IoU, Dice, panel count MAE
│   ├── train.py                # training + validation loop
│   └── predict.py              # inference on new roof images
├── scripts/
│   ├── extract_masks.py        # extract binary masks from raw overlays
│   └── inpaint_roofs.py        # remove panels to produce clean roof images
├── configs/
│   └── default.yaml            # all hyperparameters and paths
├── notebooks/
│   └── 01_data_exploration.ipynb
├── checkpoints/                # saved model weights (not versioned)
├── outputs/
│   ├── predictions/
│   └── visualizations/
└── requirements.txt
```

---

## Setup

**1. Clone the repository**
```bash
git clone <repo-url>
cd GenerativeSolarPanelLayoutDesigner
```

**2. Create a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your data**

Place your files in the correct directories:
```
data/raw/          ← labeled images (roof + panel overlay)
data/metadata.csv  ← tabular labels
```

---

## Data Preparation

**Step 1 — Extract binary masks from raw labeled images:**
```bash
python scripts/extract_masks.py
```
This thresholds the panel overlay color in each raw image and saves a binary mask to `data/masks/`.

**Step 2 — Generate clean roof images (remove panels):**
```bash
python scripts/inpaint_roofs.py
```
This uses the extracted masks to inpaint the panel regions and saves clean roof images to `data/roofs/`.

---

## Training

```bash
python -m src.train
```

To use a custom config:
```bash
python -m src.train --config configs/default.yaml
```

Training will:
- Log loss, IoU, Dice score, and panel count MAE per epoch
- Save the best checkpoint to `checkpoints/best_model.pt`
- Apply early stopping based on validation IoU

### Key Hyperparameters (`configs/default.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `encoder` | `resnet34` | U-Net backbone |
| `epochs` | `100` | Maximum training epochs |
| `batch_size` | `8` | Samples per batch |
| `lr` | `0.0003` | Initial learning rate |
| `dice_weight` | `0.5` | Weight of Dice loss vs BCE |
| `count_loss_weight` | `0.3` | Weight of panel count auxiliary loss |
| `early_stopping_patience` | `15` | Epochs without improvement before stopping |

---

## Inference

```python
import torch
import yaml
from src.predict import load_model, predict
from src.dataset import SolarLayoutDataset

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build metadata vector matching your installation specs
meta = torch.tensor([
    13 / 30,   # num_panels normalised
    1 / 4,     # num_strings normalised
    0, 1, 0, 0,  # roof_type one-hot: [metal, tile, flat, other]
    1, 0,        # connection_type: [single_phase, three_phase]
], dtype=torch.float32)

model = load_model('checkpoints/best_model.pt', cfg, meta_dim=8, device=device)
mask = predict('data/roofs/house_new.jpg', meta, model, cfg, device)
```

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **IoU** | Intersection over Union on panel mask | > 0.75 |
| **Dice Score** | F1 equivalent for mask overlap | > 0.85 |
| **Panel Count MAE** | Mean absolute error on predicted panel count | ≤ 2 panels |

---

## Model Capabilities

- Predicts panel layout mask conditioned on roof image and installation specs
- Respects string count by learning single-cluster vs multi-cluster spatial patterns
- Respects panel count via auxiliary regression head
- Handles tile, metal, and flat roof types with distinct mounting rules
- Generalizes across roof orientations via augmentation (flips, 90° rotations)
- Runs inference on any clean aerial roof image at 512×512 resolution

---

## Roadmap

- [ ] Data collection and mask extraction pipeline
- [ ] Baseline U-Net without conditioning (benchmark)
- [ ] FiLM-conditioned U-Net (primary model)
- [ ] Panel count auxiliary loss integration
- [ ] Evaluation on held-out test set
- [ ] Post-processing: snap mask to panel grid dimensions
- [ ] Streamlit demo app for visual testing

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA-capable GPU recommended (tested on RTX 3060+)
- See `requirements.txt` for full dependency list
