import math
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model   import SolarUNet
from src.utils   import load_config, load_checkpoint
from src.dataset import encode_meta

cfg    = load_config("configs/default.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = A.Compose([
    A.Resize(*cfg.data.img_size),
    A.Normalize(),
    ToTensorV2(),
])


# ---------------------------------------------------------------------------
# Roof detection  (geometric / no-model path)
# ---------------------------------------------------------------------------

def _detect_roof(img_rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 binary mask of the roof region."""
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mx, my = max(10, w // 10), max(10, h // 10)
    rect   = (mx, my, w - 2 * mx, h - 2 * my)
    gc     = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        roof = np.where((gc == 1) | (gc == 3), 255, 0).astype(np.uint8)
        k    = np.ones((15, 15), np.uint8)
        roof = cv2.morphologyEx(roof, cv2.MORPH_CLOSE, k, iterations=3)
        roof = cv2.morphologyEx(roof, cv2.MORPH_OPEN,  k, iterations=1)
        if cv2.countNonZero(roof) >= int(0.05 * h * w):
            return roof
    except cv2.error:
        pass

    # Fallback: largest edge contour
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    k       = np.ones((9, 9), np.uint8)
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= int(0.05 * h * w):
            peri  = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            roof  = np.zeros((h, w), np.uint8)
            cv2.fillPoly(roof, [approx], 255)
            return roof

    # Final fallback: centre crop
    roof = np.zeros((h, w), np.uint8)
    roof[my: h - my, mx: w - mx] = 255
    return roof


# ---------------------------------------------------------------------------
# Geometric panel layout
# ---------------------------------------------------------------------------

# Panel rect type: ((cx, cy), (cw, ch), angle) — same format as cv2.minAreaRect.
PanelRect = tuple


def _draw_panel_grid(canvas: np.ndarray, rect, num_panels: int, gap: int = 4,
                     min_cell_px: int = 0,
                     fixed_cell: tuple[float, float] | None = None,
                     ) -> tuple[list[PanelRect], float, float]:
    """Fill canvas with portrait-oriented panel cells inside rect.

    Returns (panel_rects, cell_w, cell_h).
    panel_rects: list of ((cx, cy), (cw, ch), angle) for every drawn cell,
    suitable for passing directly to _stamp_panel_at_rect.  This avoids
    re-deriving positions from the mask (which fails when cells touch).
    """
    if num_panels <= 0:
        return [], 0.0, 0.0

    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90

    if fixed_cell is not None:
        cell_w, cell_h = fixed_cell
        cols = max(1, int((rw - gap) / (cell_w + gap)))
        rows = max(1, int((rh - gap) / (cell_h + gap)))
    else:
        TARGET   = 1.7
        floor_px = max(min_cell_px, 2)
        best_cols, best_rows, best_err = 1, num_panels, float("inf")
        for c in range(1, num_panels + 1):
            r  = math.ceil(num_panels / c)
            cw = (rw - gap * (c + 1)) / c
            ch = (rh - gap * (r + 1)) / r
            if cw < floor_px or ch < floor_px:
                continue
            err = abs((ch / cw) - TARGET)
            if err < best_err:
                best_err, best_cols, best_rows = err, c, r
        cols, rows = best_cols, best_rows
        cell_w = (rw - gap * (cols + 1)) / cols
        cell_h = (rh - gap * (rows + 1)) / rows
        if cell_w < 2 or cell_h < 2:
            return [], 0.0, 0.0

    cos_a = np.cos(np.deg2rad(angle))
    sin_a = np.sin(np.deg2rad(angle))

    panel_rects: list[PanelRect] = []
    for r in range(rows):
        for c in range(cols):
            if len(panel_rects) >= num_panels:
                break
            lx = -rw / 2 + gap + c * (cell_w + gap)
            ly = -rh / 2 + gap + r * (cell_h + gap)

            # Centre of this cell in world coordinates
            lcx, lcy = lx + cell_w / 2, ly + cell_h / 2
            gcx = cx + lcx * cos_a - lcy * sin_a
            gcy = cy + lcx * sin_a + lcy * cos_a
            panel_rects.append(((gcx, gcy), (cell_w, cell_h), angle))

            local = np.array([
                [lx,          ly         ],
                [lx + cell_w, ly         ],
                [lx + cell_w, ly + cell_h],
                [lx,          ly + cell_h],
            ])
            rot       = np.zeros_like(local)
            rot[:, 0] = cx + local[:, 0] * cos_a - local[:, 1] * sin_a
            rot[:, 1] = cy + local[:, 0] * sin_a + local[:, 1] * cos_a
            cv2.drawContours(canvas, [rot.astype(np.int32)], 0, 255, -1)

    return panel_rects, cell_w, cell_h


def _build_panel_mask(roof_mask: np.ndarray, num_panels: int,
                      roof_direction: float | None = None,
                      min_cell_px: int = 0,
                      ) -> tuple[np.ndarray, list[PanelRect]]:
    """Place num_panels rectangular cells inside the roof_mask region.

    Returns (binary_mask, panel_rects).  panel_rects tracks each cell's
    exact position so callers don't have to re-derive it from the mask
    (which breaks when cells share an edge after integer rounding).
    """
    k       = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = sorted([c for c in contours if cv2.contourArea(c) >= 500],
                   key=cv2.contourArea, reverse=True)
    result = np.zeros_like(roof_mask)
    if not valid:
        return result, []

    total_area      = sum(cv2.contourArea(c) for c in valid)
    remaining       = num_panels
    shared_cell: tuple[float, float] | None = None
    all_panel_rects: list[PanelRect] = []

    for i, cnt in enumerate(valid):
        if remaining <= 0:
            break
        area_frac = cv2.contourArea(cnt) / total_area
        n = remaining if i == len(valid) - 1 else max(1, round(num_panels * area_frac))
        n = min(n, remaining)

        rect = cv2.minAreaRect(cnt)
        if roof_direction is not None:
            (rcx, rcy), (rw, rh), _ = rect
            rect = ((rcx, rcy), (rw, rh), -roof_direction)

        rects_i, cw, ch = _draw_panel_grid(result, rect, n,
                                            min_cell_px=min_cell_px,
                                            fixed_cell=shared_cell)
        all_panel_rects.extend(rects_i)
        if shared_cell is None and cw > 0:
            shared_cell = (cw, ch)
        remaining -= len(rects_i)

    roof_binary = (cleaned > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(result, roof_binary), all_panel_rects


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model() -> SolarUNet:
    m = SolarUNet(cfg).to(device)
    load_checkpoint("checkpoint/best.pt", m)
    m.eval()
    print(f"Model loaded from checkpoint/best.pt  ({device})")
    return m


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(roof_path: str, meta_dict: dict,
            use_model: bool = False,
            roof_direction: float | None = None,
            panel_size: int = 0,
            ) -> tuple[np.ndarray, np.ndarray, list[PanelRect]]:
    """Return (img_rgb, panel_mask_255, panel_rects).

    Both paths detect a roof region then fill it with a geometric panel grid.
    use_model=True  — model (FiLM-conditioned on all metadata) detects which
                      roof face to use; smarter than GrabCut on complex roofs.
    use_model=False — GrabCut detects the roof region (no checkpoint needed).
    """
    img_bgr = cv2.imread(roof_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open: {roof_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    if use_model:
        model  = _load_model()
        img_t  = _transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        meta_t = encode_meta(meta_dict).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img_t, meta_t))[0]       # (3, H, W) in [0,1]
        # Model output is a blob indicating WHERE panels should go.
        # Convert to a binary roof-region mask, then fill it with a geometric
        # panel grid — same as the no-model path but using a smarter region.
        roof_region = (pred.max(dim=0).values > cfg.inference.threshold).cpu().numpy()
        roof_region = cv2.resize(roof_region.astype(np.uint8) * 255, (w, h),
                                 interpolation=cv2.INTER_NEAREST)
        mask, panel_rects = _build_panel_mask(roof_region, meta_dict["num_panels"],
                                              roof_direction=roof_direction,
                                              min_cell_px=panel_size)
    else:
        roof         = _detect_roof(img_rgb)
        mask, panel_rects = _build_panel_mask(roof, meta_dict["num_panels"],
                                              roof_direction=roof_direction,
                                              min_cell_px=panel_size)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return img_rgb, mask, panel_rects


# ---------------------------------------------------------------------------
# Template stamping
# ---------------------------------------------------------------------------

def _stamp_panel_at_rect(canvas_bgr: np.ndarray, template_bgr: np.ndarray,
                         rect: PanelRect) -> None:
    """Warp template_bgr onto canvas_bgr at the given cell rect in portrait orientation."""
    (cx, cy), (rw, rh), angle = rect
    tw, th = max(2, int(rw)), max(2, int(rh))
    if tw > th:
        tw, th = th, tw
        angle += 90

    tpl = template_bgr
    if tpl.shape[1] > tpl.shape[0]:
        tpl = cv2.rotate(tpl, cv2.ROTATE_90_CLOCKWISE)

    H, W      = canvas_bgr.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    def _rot(x, y):
        rx, ry = x - tw / 2.0, y - th / 2.0
        return [cx + rx * cos_a - ry * sin_a,
                cy + rx * sin_a + ry * cos_a]

    src_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    dst_pts = np.array([_rot(*p) for p in src_pts], dtype=np.float32)
    M       = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped  = cv2.warpPerspective(cv2.resize(tpl, (tw, th)), M, (W, H))
    stamp   = cv2.warpPerspective(np.ones((th, tw), np.uint8) * 255, M, (W, H)) > 0
    canvas_bgr[stamp] = warped[stamp]


def _render_with_template(roof_rgb: np.ndarray, panel_rects: list[PanelRect],
                          template_bgr: np.ndarray) -> np.ndarray:
    """Stamp template at each panel rect.  Uses explicit rects rather than
    findContours so adjacent cells that share an edge after rounding are still
    stamped individually.
    """
    canvas = cv2.cvtColor(roof_rgb, cv2.COLOR_RGB2BGR)
    for rect in panel_rects:
        (cx, cy), (rw, rh), _ = rect
        if rw * rh < 4:
            continue
        _stamp_panel_at_rect(canvas, template_bgr, rect)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Output / visualisation
# ---------------------------------------------------------------------------

def save_and_show(house_id: str, img_rgb: np.ndarray, mask: np.ndarray,
                  panel_rects: list[PanelRect],
                  template_bgr: np.ndarray | None = None):
    os.makedirs(cfg.inference.output_dir, exist_ok=True)

    if template_bgr is not None:
        rendered     = _render_with_template(img_rgb, panel_rects, template_bgr)
        render_label = "Panel Render"
    else:
        overlay  = img_rgb.copy()
        overlay[mask > 0] = (0, 200, 80)
        rendered     = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
        render_label = "Overlay"

    preview_path = os.path.join(cfg.inference.output_dir, f"{house_id}_preview.png")
    cv2.imwrite(preview_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    print(f"Saved: {preview_path}")

    num_drawn = len(panel_rects)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{house_id}  |  panels drawn: {num_drawn}", fontsize=13)
    axes[0].imshow(img_rgb);  axes[0].set_title("Roof");        axes[0].axis("off")
    axes[1].imshow(rendered); axes[1].set_title(render_label);  axes[1].axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Solar panel layout inference.\n\n"
                    "Without --use-model: roof is detected with GrabCut and panels "
                    "are placed geometrically. Only --num-panels affects the output.\n\n"
                    "With --use-model: the trained model runs on the image + all "
                    "metadata via FiLM conditioning. All metadata args affect the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--roof",       required=True, help="Path to the roof image")
    parser.add_argument("--id",         default=None,  help="Label for output files (defaults to image stem)")
    parser.add_argument("--num-panels", required=True, type=int, help="Number of solar panels")
    parser.add_argument("--use-model",  action="store_true",
                        help="Use the trained model; all metadata then affects panel placement")

    meta = parser.add_argument_group("metadata (used with --use-model)")
    meta.add_argument("--roof-type",       default="tile",         choices=["tile", "tin", "flat"])
    meta.add_argument("--connection-type", default="single_phase", choices=["single_phase", "three_phase"])
    meta.add_argument("--angle",           default=20.0, type=float, help="Roof tilt in degrees")
    meta.add_argument("--num-strings",     default=1,    type=int)

    geom = parser.add_argument_group("geometric mode options (ignored with --use-model)")
    geom.add_argument("--roof-direction", default=None, type=float,
                      help="Direction the slope faces, degrees CW from image-top "
                           "(0=up, 90=right, 180=down, 270=left). Auto-detected when omitted.")
    geom.add_argument("--panel-size",     default=0,    type=int,
                      help="Minimum panel cell width in pixels (0 = auto-scale)")

    parser.add_argument("--panel-template", default=None,
                        help="Reference panel image for stamped rendering; "
                             "omit for a green overlay instead")
    args = parser.parse_args()

    house_id = args.id or os.path.splitext(os.path.basename(args.roof))[0]
    meta_dict = {
        "roof_type":       args.roof_type,
        "connection_type": args.connection_type,
        "num_panels":      args.num_panels,
        "angle":           args.angle,
        "num_strings":     args.num_strings,
    }

    template_bgr = cv2.imread(args.panel_template) if args.panel_template else None
    img_rgb, mask, panel_rects = predict(
        args.roof, meta_dict,
        use_model=args.use_model,
        roof_direction=args.roof_direction,
        panel_size=args.panel_size,
    )
    save_and_show(house_id, img_rgb, mask, panel_rects, template_bgr=template_bgr)
