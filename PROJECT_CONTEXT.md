# Generative Solar Panel Layout Designer — Project Context

## What This Project Does
A deep learning pipeline that takes an aerial/satellite image of a rooftop and metadata
(roof type, panel count, angle, strings, connection type) and predicts where solar panels
should be placed. Output is a mask of individual rectangular panel cells overlaid on the roof image.

---

## Architecture

### Model — `src/model.py`
- **Base**: `smp.Unet` with `resnet34` encoder (ImageNet pretrained), outputs 1-channel logit map
- **FiLM conditioning**: a small MLP takes the 8-dim metadata vector and produces γ/β
  scale-shift parameters applied to the 16-channel decoder output before the segmentation head
- Input: RGB image (3, 512, 512) + metadata tensor (8,)
- Output: logit map (1, 512, 512), thresholded at 0.5 for binary mask

### Metadata encoding — `src/dataset.py → encode_meta()`
8-dim vector:
- [0-2]: one-hot roof_type (tile / tin / flat)
- [3-4]: one-hot connection_type (single_phase / three_phase)
- [5]:   num_panels / 70.0   ← already normalised to [0,1]
- [6]:   angle / 90
- [7]:   num_strings / 10.0

### Loss — `src/loss.py → CombinedLoss`
```
total = dice_weight * (BCE + Dice)
      + count_loss_weight * CountLoss
      + setback_loss_weight * SetbackLoss
```
- **BCE**: `BCEWithLogitsLoss(pos_weight=10.0)` — upweights panel pixels (~5% of image)
- **Dice**: standard soft Dice loss
- **CountLoss**: MSE between `sigmoid(logits).sum() / (600 * 70)` and `meta[:,5]`
  — both sides normalised to [0,1]. pixels_per_panel=600 calibrated for 512×512 input.
- **SetbackLoss**: penalises predictions within 20px of image border

Current weights in `configs/default.yaml`:
```yaml
dice_weight:        0.7
count_loss_weight:  0.15
setback_loss_weight: 0.1
```

---

## Data Pipeline

### Dataset — `data/`
- ~78 labelled houses (growing)
- `data/roofs/`    — aerial roof images (original ~700×1424, various sizes)
- `data/masks/`    — binary masks (255 = panel, 0 = background)
- `data/metadata.csv` — columns: image_filename, mask_filename, num_panels, roof_type,
  angle, num_strings, connection_type

### Mask generation — `scripts/prepare_single.py`
Diffs a clean roof screenshot vs a screenshot with panels placed in Pylon browser app.
Key: blobs are fitted with `cv2.minAreaRect` before saving so masks are **rectangular**,
teaching the model to predict rectangular regions not amorphous blobs.

For dark roofs use the `--dark-roof` flag in `scripts/capture_pair.py`:
```bash
python scripts/capture_pair.py --id house_X --dark-roof
```
Equivalent to: `--enhance --diff-mode max-channel --threshold 15 --min-area 800`

Batch reprocess all masks:
```bash
python scripts/prepare_batch.py --reprocess-all --no-preview
```

### Augmentations — `src/dataset.py → get_transforms()`
Training only:
- RandomRotate90, HorizontalFlip, VerticalFlip
- Affine (translate 5%, scale ±10%, rotate ±15°)
- RandomBrightnessContrast, HueSaturationValue, GaussNoise
- CoarseDropout (1-4 holes, 16-32px)
- Resize to 512×512, ImageNet Normalize

---

## Training — `train.py`

Current config (`configs/default.yaml`):
```yaml
epochs:                   100
batch_size:               8
lr:                       0.0001
lr_scheduler:             cosine
weight_decay:             0.0001
early_stopping_patience:  20
```

Run:
```bash
python train.py
```
Saves only `checkpoint/best.pt` (lowest val loss). Per-epoch saves removed.

Train/val split: 85% / 15% (shuffled with random_state=42).
With ~78 samples: ~66 train, ~12 val.

---

## Inference — `inference.py`

Preprocessing matches training exactly:
```python
A.Resize(512, 512) → A.Normalize() → ToTensorV2()
```

Post-processing pipeline (no retraining needed):
1. Threshold sigmoid output at 0.5
2. Morphological close to clean small gaps
3. Find contours, filter < 500px area
4. Fit `cv2.minAreaRect` to each blob
5. Subdivide each rectangle into a `cols × rows` grid of individual panels
   — grid dimensions derived from region aspect ratio + `num_panels` from metadata
   — panels on multiple roof faces: distributed proportionally by area

Output: individual panel rectangles drawn on original image resolution.

Edit the bottom of `inference.py` to test:
```python
HOUSE_ID  = "house_001"
ROOF_PATH = f"data/roofs/{HOUSE_ID}_roof.jpg"
META = {
    "roof_type":       "tile",
    "connection_type": "single_phase",
    "num_panels":      13,
    "angle":           22,
    "num_strings":     1,
}
```

Run:
```bash
python inference.py
```

---

## Known Issues & What Has Been Fixed

### Fixed
| Issue | Fix |
|---|---|
| CountLoss exploding to ~22,000 at init (dominated training, model collapsed to all-zeros) | Normalised both sides to [0,1]; changed pixels_per_panel 800→600 for 512×512 |
| Inference preprocessing mismatch (no resize, no ImageNet normalisation) | Added albumentations transform matching training pipeline |
| pos_weight tensor on CPU, model on GPU → RuntimeError | Used `register_buffer` + `.to(logits.device)` |
| ShiftScaleRotate / CoarseDropout deprecated in newer albumentations | Replaced with Affine and updated CoarseDropout arg names |
| Model saving every epoch (disk space) | Removed per-epoch save, only best.pt saved |
| Blob predictions instead of rectangular panels | minAreaRect in mask generation + panel grid post-processing in inference |

### Remaining Challenges
1. **Small dataset (~78 samples)** — primary bottleneck. Target 150+ for meaningful improvement.
2. **Val set too small (~12 samples)** — val loss is noisy, early stopping may trigger prematurely.
3. **Model predicts wrong roof face** — seen predicting light-coloured sections instead of actual panels.
   Root cause: training data bias. Fix by adding more diverse dark-roof examples.
4. **Panel grid orientation** — post-processing assumes panels align with the longest rectangle axis.
   Could be wrong if the roof face is oriented unusually.
5. **No test set** — only train/val. Consider holding out 10 houses as a permanent test set.

---

## Suggested Next Improvements

### High impact
- **More data**: use `prepare_batch.py` with `--dark-roof` on new captures. Target 150 total samples.
- **Encoder upgrade**: change `encoder: resnet34` → `efficientnet-b3` in `configs/default.yaml`.
  Better feature extraction, similar training time.
- **Tune inference threshold**: try 0.3–0.4 if predictions are sparse; 0.6 if over-predicting.

### Medium impact
- **Focal loss instead of BCE**: better handles extreme class imbalance than pos_weight alone.
  Replace BCE with `FocalLoss(alpha=0.25, gamma=2.0)`.
- **k-fold cross-validation**: with only ~78 samples, a single val split is unreliable.
  5-fold CV gives a more honest estimate of generalisation.
- **Test-time augmentation (TTA)**: run inference on 4 rotations + flips, average predictions.
  Free accuracy gain, no retraining.

### Lower impact
- **Warmup LR schedule**: add 5-epoch linear warmup before cosine decay to stabilise early training.
- **Increase film_hidden_dim**: from 256 → 512 to give FiLM more capacity to condition on metadata.
- **Panel aspect ratio in grid**: current assumption is 2:1 (w:h). Make this a config parameter
  to match actual panel dimensions used in the Pylon app.

---

## File Reference

| File | Purpose |
|---|---|
| `train.py` | Training loop |
| `inference.py` | Prediction + panel grid drawing |
| `src/model.py` | UNet + FiLM architecture |
| `src/dataset.py` | Dataset, augmentations, encode_meta |
| `src/loss.py` | BCE + Dice + CountLoss + SetbackLoss |
| `src/utils.py` | load_config, save/load checkpoint, EarlyStopping |
| `configs/default.yaml` | All hyperparameters |
| `scripts/capture_pair.py` | Screenshot capture (use --dark-roof for dark roofs) |
| `scripts/prepare_single.py` | Single mask extraction (outputs rectangles) |
| `scripts/prepare_batch.py` | Batch mask extraction |
| `data/metadata.csv` | Ground truth labels |
