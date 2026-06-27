import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model   import SolarUNet
from src.utils   import load_config, load_checkpoint
from src.dataset import encode_meta

cfg    = load_config("configs/default.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SolarUNet(cfg).to(device)
load_checkpoint("checkpoint/best.pt", model)
model.eval()
print(f"Model loaded from checkpoint/best.pt  ({device})")


_infer_transform = A.Compose([
    A.Resize(*cfg.data.img_size),
    A.Normalize(),
    ToTensorV2(),
])


def _draw_panel_grid(canvas: np.ndarray, rect, num_panels: int, gap: int = 4):
    """Fill canvas with a grid of num_panels individual rectangles inside rect."""
    (cx, cy), (rw, rh), angle = rect

    # ensure width is the long axis
    if rw < rh:
        rw, rh = rh, rw
        angle += 90

    # work out cols × rows that best fits the region's aspect ratio
    # solar panels are ~2× wider than tall
    panel_aspect = 2.0
    ratio = (rw / max(rh, 1)) / panel_aspect
    cols  = max(1, round((num_panels * ratio) ** 0.5))
    rows  = max(1, round(num_panels / cols))
    while cols * rows < num_panels:
        cols += 1

    cell_w = (rw - gap * (cols + 1)) / cols
    cell_h = (rh - gap * (rows + 1)) / rows
    if cell_w < 2 or cell_h < 2:
        return

    cos_a = np.cos(np.deg2rad(angle))
    sin_a = np.sin(np.deg2rad(angle))

    drawn = 0
    for r in range(rows):
        for c in range(cols):
            if drawn >= num_panels:
                break
            # local (unrotated) corner offsets from rect centre
            lx = -rw / 2 + gap + c * (cell_w + gap)
            ly = -rh / 2 + gap + r * (cell_h + gap)
            local = np.array([
                [lx,          ly         ],
                [lx + cell_w, ly         ],
                [lx + cell_w, ly + cell_h],
                [lx,          ly + cell_h],
            ])
            # rotate and translate to image space
            rot       = np.zeros_like(local)
            rot[:, 0] = cx + local[:, 0] * cos_a - local[:, 1] * sin_a
            rot[:, 1] = cy + local[:, 0] * sin_a + local[:, 1] * cos_a
            cv2.drawContours(canvas, [rot.astype(np.int32)], 0, 1, -1)
            drawn += 1


def _build_panel_mask(mask: np.ndarray, num_panels: int) -> np.ndarray:
    """Convert raw sigmoid mask into individual drawn panels."""
    kernel  = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = [c for c in contours if cv2.contourArea(c) >= 500]
    result = np.zeros_like(mask)
    if not valid:
        return result

    # distribute num_panels across regions proportionally to their area
    total_area = sum(cv2.contourArea(c) for c in valid)
    remaining  = num_panels
    for i, cnt in enumerate(valid):
        n = remaining if i == len(valid) - 1 else max(1, round(num_panels * cv2.contourArea(cnt) / total_area))
        remaining -= n
        _draw_panel_grid(result, cv2.minAreaRect(cnt), n)

    return result


def predict(roof_path: str, meta_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(roof_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {roof_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    img_t = _infer_transform(image=img)["image"].unsqueeze(0).to(device)
    meta  = encode_meta(pd.Series(meta_dict)).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(img_t, meta)

    mask = torch.sigmoid(logit).squeeze().cpu().numpy() > cfg.inference.threshold
    mask = _build_panel_mask(mask.astype(np.uint8), meta_dict["num_panels"])
    return img, cv2.resize(mask, (w, h))


def save_and_show(house_id: str, img_rgb: np.ndarray, mask: np.ndarray):
    os.makedirs(cfg.inference.output_dir, exist_ok=True)

    # green overlay on original image
    overlay = img_rgb.copy()
    overlay[mask > 0] = (0, 200, 80)
    preview = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    mask_path   = os.path.join(cfg.inference.output_dir, f"{house_id}_mask.png")
    preview_path = os.path.join(cfg.inference.output_dir, f"{house_id}_preview.png")
    cv2.imwrite(mask_path,   mask * 255)
    cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    print(f"Saved: {mask_path}")
    print(f"Saved: {preview_path}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{house_id}  |  panels drawn: {cv2.connectedComponents(mask)[0] - 1}",
                 fontsize=13)
    axes[0].imshow(img_rgb);          axes[0].set_title("Roof");        axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Pred Mask"); axes[1].axis("off")
    axes[2].imshow(preview);           axes[2].set_title("Overlay");    axes[2].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a roof image.")
    parser.add_argument("--roof",            required=True,  help="Path to the roof image (any format cv2 can open)")
    parser.add_argument("--id",              default=None,   help="Label used for output filenames (defaults to image stem)")
    parser.add_argument("--roof-type",       required=True,  choices=["tile", "tin", "flat"])
    parser.add_argument("--connection-type", required=True,  choices=["single_phase", "three_phase"])
    parser.add_argument("--num-panels",      required=True,  type=int,   help="Expected number of panels")
    parser.add_argument("--angle",           required=True,  type=float, help="Roof tilt angle in degrees")
    parser.add_argument("--num-strings",     required=True,  type=int,   help="Number of strings")
    args = parser.parse_args()

    house_id = args.id or os.path.splitext(os.path.basename(args.roof))[0]
    meta = {
        "roof_type":       args.roof_type,
        "connection_type": args.connection_type,
        "num_panels":      args.num_panels,
        "angle":           args.angle,
        "num_strings":     args.num_strings,
    }

    img_rgb, mask = predict(args.roof, meta)
    save_and_show(house_id, img_rgb, mask)
