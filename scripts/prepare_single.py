import argparse
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _nms(boxes, scores, iou_threshold=0.3):
    """Non-maximum suppression for overlapping template matches."""
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    kept   = []
    while order.size > 0:
        i = order[0]
        kept.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_threshold]
    return kept


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees, expanding canvas so no corners are clipped."""
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h))


def _sample_template_color(template_bgr: np.ndarray) -> np.ndarray:
    """Return the median BGR colour of the template's inner panel area (ignores borders)."""
    h, w = template_bgr.shape[:2]
    ch, cw = max(1, h // 4), max(1, w // 4)
    center = template_bgr[ch: h - ch, cw: w - cw]
    if center.size == 0:
        center = template_bgr
    return np.median(center.reshape(-1, 3), axis=0).astype(np.float32)


def _color_match_mask(layout_bgr: np.ndarray, panel_color: np.ndarray,
                      color_tol: int, cell_min_area: int) -> tuple:
    """Find layout regions whose per-channel colour is within tol of panel_color."""
    diff  = np.abs(layout_bgr.astype(np.float32) - panel_color)
    raw   = (diff.max(axis=2) < color_tol).astype(np.uint8) * 255
    k5    = np.ones((5, 5), np.uint8)
    k3    = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(raw,   cv2.MORPH_CLOSE, k5, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  k3, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(layout_bgr.shape[:2], dtype=np.uint8)
    kept   = 0
    for cnt in contours:
        if cv2.contourArea(cnt) >= cell_min_area:
            box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int32)
            cv2.drawContours(result, [box], 0, 255, -1)
            kept += 1
    return result, kept


def detect_by_template(layout_bgr, template_bgr, match_thresh=0.45, nms_iou=0.3, cell_min_area=300):
    """
    Detect panels using the template image.

    Primary path  — colour match: samples the panel colour from the template
                    centre and finds all matching pixels in the layout.
                    Rotation- and scale-invariant; works for any Pylon rendering.

    Fallback path — multi-scale + multi-angle sliding-window template match,
                    used only when the colour match finds nothing.
    """
    # ── Primary: colour-based detection (rotation-invariant) ─────────────────
    panel_color = _sample_template_color(template_bgr)
    for tol in (35, 50, 65):
        mask, kept = _color_match_mask(layout_bgr, panel_color, tol, cell_min_area)
        if kept > 0:
            print(f"  Colour match — panels detected: {kept}  (tolerance={tol})")
            return mask, kept

    # ── Fallback: multi-scale, multi-angle template matching ──────────────────
    print("  Colour match found 0 panels; trying rotated template matching …")
    gray_layout = cv2.cvtColor(layout_bgr, cv2.COLOR_BGR2GRAY)
    gray_tmpl   = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    all_boxes, all_scores = [], []
    scales = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0]
    angles = list(range(-60, 65, 15))

    for angle in angles:
        rot      = _rotate_image(gray_tmpl, angle) if angle != 0 else gray_tmpl
        rth, rtw = rot.shape[:2]
        for scale in scales:
            rw = max(2, int(rtw * scale))
            rh = max(2, int(rth * scale))
            if rh > gray_layout.shape[0] or rw > gray_layout.shape[1]:
                continue
            tpl_s  = cv2.resize(rot, (rw, rh))
            result = cv2.matchTemplate(gray_layout, tpl_s, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= match_thresh)
            for x, y in zip(xs, ys):
                all_boxes.append([x, y, x + rw, y + rh])
                all_scores.append(float(result[y, x]))

    if not all_boxes:
        return np.zeros(layout_bgr.shape[:2], dtype=np.uint8), 0

    kept_idx   = _nms(all_boxes, all_scores, nms_iou)
    kept_boxes = [
        b for b in np.array(all_boxes)[kept_idx].astype(int)
        if (b[2] - b[0]) * (b[3] - b[1]) >= cell_min_area
    ]
    mask = np.zeros(layout_bgr.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in kept_boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    print(f"  Template matching — panels detected: {len(kept_boxes)}")
    return mask, len(kept_boxes)


def crop_image(img, margins):
    """Trim pixels from each edge. margins = (top, right, bottom, left)."""
    top, right, bottom, left = margins
    h, w = img.shape[:2]
    return img[top: h - bottom if bottom else h,
               left: w - right  if right  else w]


def enhance_image(img):
    """Apply CLAHE on the L channel (LAB) to boost local contrast on dark roofs."""
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _cells_by_color(layout_bgr, coarse_mask, cell_min_area, color_tol):
    """
    Detect individual panel cells within a coarse panel region by colour.
    Returns (binary_mask, count).
    """
    panel_pixels = layout_bgr[coarse_mask > 0]
    if len(panel_pixels) == 0:
        return np.zeros(coarse_mask.shape[:2], dtype=np.uint8), 0

    panel_color = np.median(panel_pixels, axis=0).astype(np.float32)
    dist        = np.max(np.abs(layout_bgr.astype(np.float32) - panel_color), axis=2)
    cell_pixels = ((dist < color_tol) * 255).astype(np.uint8)
    cell_pixels = cv2.bitwise_and(cell_pixels, coarse_mask)
    cell_pixels = cv2.morphologyEx(
        cell_pixels, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(cell_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(cell_pixels)
    kept   = 0
    for cnt in contours:
        if cv2.contourArea(cnt) >= cell_min_area:
            rect = cv2.minAreaRect(cnt)
            box  = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(result, [box], 0, 255, thickness=cv2.FILLED)
            kept += 1

    return result, kept


# ---------------------------------------------------------------------------
# Panel-cut mode (NEW default): extract actual panel pixels from layout
# ---------------------------------------------------------------------------

def _detect_panel_region_from_layout(layout_bgr, cell_min_area=300, color_tol=35):
    """
    Detect panel pixels in the layout image using GrabCut + colour sampling.
    Returns a binary mask (255 = panel pixel).
    """
    h, w = layout_bgr.shape[:2]
    mx, my = max(10, w // 10), max(10, h // 10)
    rect   = (mx, my, w - 2 * mx, h - 2 * my)

    gc_mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(layout_bgr, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        coarse = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)
        k      = np.ones((15, 15), np.uint8)
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, k, iterations=3)
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,  k, iterations=1)
    except cv2.error:
        coarse = np.zeros((h, w), np.uint8)
        coarse[my: h - my, mx: w - mx] = 255

    panel_mask, kept = _cells_by_color(layout_bgr, coarse, cell_min_area, color_tol)
    if kept == 0:
        # Fallback: use the whole GrabCut foreground as the region
        panel_mask = coarse

    return panel_mask


def extract_panel_cut(layout_path, clean_path=None, crop_margins=(0, 0, 0, 0),
                      panel_template_path=None, match_thresh=0.45, nms_iou=0.3,
                      cell_min_area=300, cell_color_tol=35, diff_threshold=20):
    """
    Cut the actual panel pixels from the layout image.

    Detection priority:
      1. Diff vs clean roof — works regardless of panel colour or rotation.
      2. Template colour match + rotated sliding-window fallback.
      3. GrabCut + colour sampling (last resort).

    Returns (panel_cut_bgr, layout_bgr_cropped).
    panel_cut_bgr — BGR image; panel-region pixels kept, all others zeroed.
    """
    layout = cv2.imread(layout_path)
    if layout is None:
        raise FileNotFoundError(f"Cannot open layout image: {layout_path}")
    layout = crop_image(layout, crop_margins)

    panel_mask = None

    # ── 1. Diff-based detection (most reliable) ───────────────────────────────
    if clean_path is not None:
        clean = cv2.imread(clean_path)
        if clean is not None:
            clean = crop_image(clean, crop_margins)
            if clean.shape != layout.shape:
                clean = cv2.resize(clean, (layout.shape[1], layout.shape[0]))
            diff      = cv2.absdiff(clean, layout)
            diff_gray = np.max(diff, axis=2).astype(np.uint8)
            _, coarse = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)
            k         = np.ones((5, 5), np.uint8)
            coarse    = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, k, iterations=3)
            coarse    = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,  k, iterations=1)
            diff_area = cv2.countNonZero(coarse)
            if diff_area > 0:
                # Try to separate individual cells within the diff blob
                cell_mask, kept = _cells_by_color(layout, coarse, cell_min_area, cell_color_tol)
                if kept > 0:
                    print(f"  Diff + colour — individual cells: {kept}")
                    panel_mask = cell_mask
                else:
                    print(f"  Diff — coarse region ({diff_area} px); no cell separation, using blob")
                    panel_mask = coarse
            else:
                print(f"  Diff found nothing (threshold={diff_threshold}); trying template/colour")

    # ── 2. Template-based detection ────────────────────────────────────────────
    if panel_mask is None and panel_template_path is not None:
        template = cv2.imread(panel_template_path)
        if template is None:
            raise FileNotFoundError(f"Cannot open panel template: {panel_template_path}")
        panel_mask, kept = detect_by_template(
            layout, template,
            match_thresh=match_thresh, nms_iou=nms_iou, cell_min_area=cell_min_area,
        )
        if kept == 0:
            panel_mask = None
            print("  Template also found 0 panels; falling back to GrabCut")

    # ── 3. GrabCut + colour sampling (last resort) ────────────────────────────
    if panel_mask is None:
        print("  No clean roof or template — using GrabCut + colour sampling")
        panel_mask = _detect_panel_region_from_layout(layout, cell_min_area, cell_color_tol)
        detected   = cv2.connectedComponents(panel_mask)[0] - 1
        print(f"  Colour-based panel regions detected: {detected}")

    # Cut: preserve layout pixels inside panel region, zero everything else
    panel_cut = np.zeros_like(layout)
    panel_cut[panel_mask > 0] = layout[panel_mask > 0]

    return panel_cut, layout


# ---------------------------------------------------------------------------
# Legacy diff-based mode (kept for backward compatibility)
# ---------------------------------------------------------------------------

def extract_mask(clean_path, layout_path, threshold=25, min_area=1000,
                 crop_margins=(0, 0, 0, 0), enhance=False, diff_mode="gray",
                 cell_masks=False, cell_min_area=300, cell_color_tol=35,
                 panel_template_path=None, match_thresh=0.55, nms_iou=0.3):
    clean  = cv2.imread(clean_path)
    layout = cv2.imread(layout_path)

    if clean is None:
        raise FileNotFoundError(f"Cannot open clean roof image: {clean_path}")
    if layout is None:
        raise FileNotFoundError(f"Cannot open layout image: {layout_path}")

    clean  = crop_image(clean,  crop_margins)
    layout = crop_image(layout, crop_margins)

    if clean.shape != layout.shape:
        layout = cv2.resize(layout, (clean.shape[1], clean.shape[0]))

    layout_raw = layout.copy()

    if enhance:
        clean  = enhance_image(clean)
        layout = enhance_image(layout)

    if panel_template_path is not None:
        template_bgr = cv2.imread(panel_template_path)
        if template_bgr is None:
            raise FileNotFoundError(f"Cannot open panel template: {panel_template_path}")
        filtered, kept = detect_by_template(
            layout_raw, template_bgr,
            match_thresh=match_thresh, nms_iou=nms_iou, cell_min_area=cell_min_area,
        )
        print(f"  Template matching — panels detected: {kept}")
        if kept == 0:
            print(f"  Tip: try --match-thresh {max(0.35, match_thresh - 0.1):.2f}")
        return filtered, clean

    diff = cv2.absdiff(clean, layout)
    if diff_mode == "max-channel":
        diff_gray = np.max(diff, axis=2).astype(np.uint8)
    else:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask  = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    ck       = np.ones((5, 5), np.uint8)
    coarse   = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, ck, iterations=2)
    coarse   = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,  ck, iterations=1)
    coarse_contours, _ = cv2.findContours(coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cell_masks:
        coarse_mask  = np.zeros_like(coarse)
        valid_arrays = [c for c in coarse_contours if cv2.contourArea(c) >= min_area]
        cv2.drawContours(coarse_mask, valid_arrays, -1, 255, thickness=cv2.FILLED)
        filtered, kept = _cells_by_color(layout_raw, coarse_mask, cell_min_area, cell_color_tol)
        print(f"  Cell mode — arrays: {len(valid_arrays)}  |  panels kept: {kept}")
    else:
        filtered = np.zeros_like(coarse)
        kept     = 0
        for cnt in coarse_contours:
            if cv2.contourArea(cnt) >= min_area:
                rect = cv2.minAreaRect(cnt)
                box  = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(filtered, [box], 0, 255, thickness=cv2.FILLED)
                kept += 1
        print(f"  Contours found: {len(coarse_contours)}  |  kept: {kept}")

    return filtered, clean


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_cut(layout_bgr, panel_cut_bgr, house_id):
    layout_rgb    = cv2.cvtColor(layout_bgr,   cv2.COLOR_BGR2RGB)
    cut_rgb       = cv2.cvtColor(panel_cut_bgr, cv2.COLOR_BGR2RGB)

    # Blend cut over layout so you can see placement in context
    panel_mask    = (panel_cut_bgr.sum(axis=2) > 0).astype(np.uint8)
    overlay       = layout_rgb.copy()
    overlay[panel_mask > 0] = cut_rgb[panel_mask > 0]

    num_regions   = cv2.connectedComponents(panel_mask.astype(np.uint8))[0] - 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{house_id}  |  panel regions: {num_regions}", fontsize=13)
    axes[0].imshow(layout_rgb);  axes[0].set_title("Layout (with panels)"); axes[0].axis("off")
    axes[1].imshow(cut_rgb);     axes[1].set_title("Panel Cut (target)");   axes[1].axis("off")
    axes[2].imshow(overlay);     axes[2].set_title("Overlay");               axes[2].axis("off")
    plt.tight_layout()
    plt.show()


def preview(clean_cropped, mask, house_id):
    """Legacy preview for diff/binary-mask mode."""
    clean_rgb = cv2.cvtColor(clean_cropped, cv2.COLOR_BGR2RGB)
    overlay   = clean_rgb.copy()
    overlay[mask > 0] = (0, 200, 80)
    preview_img   = cv2.addWeighted(clean_rgb, 0.6, overlay, 0.4, 0)
    num_labels, _ = cv2.connectedComponents(mask)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{house_id}  |  detected regions: {num_labels - 1}", fontsize=13)
    axes[0].imshow(clean_rgb);        axes[0].set_title("Clean Roof");  axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray");axes[1].set_title("Binary Mask"); axes[1].axis("off")
    axes[2].imshow(preview_img);      axes[2].set_title("Overlay");     axes[2].axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roof",      required=True,  help="Path to clean roof image (copied as model input)")
    parser.add_argument("--layout",    required=True,  help="Path to layout image with panels visible")
    parser.add_argument("--id",        required=True,  help="House ID e.g. house_001")

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--layout-cut", action="store_true", default=True,
                      help="(default) Extract RGB panel pixels from layout image as training target")
    mode.add_argument("--diff",        action="store_true",
                      help="Legacy: create binary mask from diff(clean, layout)")

    # Panel detection options (used by both modes)
    parser.add_argument("--panel-template",  default=None,
                        help="Path to a single panel reference image for template-matching detection")
    parser.add_argument("--match-thresh",    type=float, default=0.55)
    parser.add_argument("--nms-iou",         type=float, default=0.3)
    parser.add_argument("--cell-min-area",   type=int,   default=300)
    parser.add_argument("--cell-color-tol",  type=int,   default=35,
                        help="Max per-channel colour distance for colour-based panel detection")

    # Crop / enhance (shared)
    parser.add_argument("--crop",     type=int, nargs=4, default=[0, 0, 0, 0],
                        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"))
    parser.add_argument("--enhance",  action="store_true")

    # Legacy diff options
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--min-area",  type=int, default=1000)
    parser.add_argument("--diff-mode", choices=["gray", "max-channel"], default="gray")
    parser.add_argument("--cell-masks", action="store_true")

    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    # --diff explicitly given → override the default --layout-cut
    use_diff = args.diff

    os.makedirs("data/roofs", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)

    _, ext   = os.path.splitext(args.roof)
    roof_out = os.path.join("data/roofs", f"{args.id}_roof{ext}")
    mask_out = os.path.join("data/masks", f"{args.id}_mask.png")

    crop_margins = tuple(args.crop)

    print(f"[1/3] Copying clean roof  → {roof_out}")
    shutil.copy2(args.roof, roof_out)

    if use_diff:
        # ── Legacy binary-mask mode ──────────────────────────────────────────
        print(f"[2/3] Extracting binary mask (diff mode)  → {mask_out}")
        mask, clean_cropped = extract_mask(
            args.roof, args.layout,
            threshold=args.threshold, min_area=args.min_area,
            crop_margins=crop_margins, enhance=args.enhance,
            diff_mode=args.diff_mode,
            cell_masks=args.cell_masks,
            cell_min_area=args.cell_min_area,
            cell_color_tol=args.cell_color_tol,
            panel_template_path=args.panel_template,
            match_thresh=args.match_thresh, nms_iou=args.nms_iou,
        )
        cv2.imwrite(mask_out, mask)
        if not args.no_preview:
            print("[3/3] Opening preview")
            preview(clean_cropped, mask, args.id)
        else:
            print("[3/3] Preview skipped")
    else:
        # ── New RGB panel-cut mode ───────────────────────────────────────────
        print(f"[2/3] Extracting panel cut (layout-cut mode)  → {mask_out}")
        panel_cut, layout_bgr = extract_panel_cut(
            args.layout,
            clean_path=args.roof,
            crop_margins=crop_margins,
            panel_template_path=args.panel_template,
            match_thresh=args.match_thresh, nms_iou=args.nms_iou,
            cell_min_area=args.cell_min_area, cell_color_tol=args.cell_color_tol,
        )
        cv2.imwrite(mask_out, panel_cut)   # saved as colour (3-channel) PNG
        if not args.no_preview:
            print("[3/3] Opening preview")
            preview_cut(layout_bgr, panel_cut, args.id)
        else:
            print("[3/3] Preview skipped")

    print(f"\nDone. Add this row to data/metadata.csv:")
    print(f"  {args.id}_roof{ext},{args.id}_mask.png,"
          f"<num_panels>,<roof_type>,<angle>,<num_strings>,<connection_type>")


if __name__ == "__main__":
    main()
