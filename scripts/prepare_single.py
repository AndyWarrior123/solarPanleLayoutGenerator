import argparse
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def extract_mask(clean_path, layout_path, threshold=25, min_area=1000,
                 crop_margins=(0, 0, 0, 0), enhance=False, diff_mode="gray"):
    clean  = cv2.imread(clean_path)
    layout = cv2.imread(layout_path)

    if clean is None:
        raise FileNotFoundError(f"Cannot open clean roof image: {clean_path}")
    if layout is None:
        raise FileNotFoundError(f"Cannot open layout image: {layout_path}")

    # Crop both images identically before diffing to remove browser UI
    clean  = crop_image(clean,  crop_margins)
    layout = crop_image(layout, crop_margins)

    if clean.shape != layout.shape:
        layout = cv2.resize(layout, (clean.shape[1], clean.shape[0]))

    if enhance:
        clean  = enhance_image(clean)
        layout = enhance_image(layout)

    diff = cv2.absdiff(clean, layout)

    if diff_mode == "max-channel":
        diff_gray = np.max(diff, axis=2).astype(np.uint8)
    else:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Drop blobs too small to be a panel array
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    kept = 0
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            # Fit a minimum-area rectangle so the mask teaches the model
            # to predict clean rectangular regions, not amorphous blobs
            rect = cv2.minAreaRect(contour)
            box  = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(filtered, [box], 0, 255, thickness=cv2.FILLED)
            kept += 1
    print(f"  Contours found: {len(contours)}  |  kept after area filter: {kept}")

    return filtered, clean  # return cropped clean so preview stays aligned


def preview(clean_cropped, mask, house_id):
    clean_rgb = cv2.cvtColor(clean_cropped, cv2.COLOR_BGR2RGB)

    overlay = clean_rgb.copy()
    overlay[mask > 0] = (0, 200, 80)
    preview_img = cv2.addWeighted(clean_rgb, 0.6, overlay, 0.4, 0)

    num_labels, _ = cv2.connectedComponents(mask)
    panel_count   = num_labels - 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{house_id}  |  detected regions: {panel_count}", fontsize=13)

    axes[0].imshow(clean_rgb);         axes[0].set_title("Clean Roof");  axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Binary Mask"); axes[1].axis("off")
    axes[2].imshow(preview_img);       axes[2].set_title("Overlay");     axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roof",      required=True,  help="Path to clean roof image")
    parser.add_argument("--layout",    required=True,  help="Path to layout image with panels")
    parser.add_argument("--id",        required=True,  help="House ID e.g. house_001")
    parser.add_argument("--threshold", type=int, default=25,
                        help="Diff sensitivity 0-255 (lower = more sensitive)")
    parser.add_argument("--min-area",  type=int, default=1000,
                        help="Minimum blob area in pixels to keep")
    parser.add_argument("--crop",      type=int, nargs=4, default=[0, 0, 0, 0],
                        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
                        help="Pixels to trim from each edge to remove browser UI")
    parser.add_argument("--enhance",    action="store_true",
                        help="Apply CLAHE contrast boost before diffing (helps dark roofs)")
    parser.add_argument("--diff-mode",  choices=["gray", "max-channel"], default="gray",
                        help="'gray' averages channels; 'max-channel' uses the channel "
                             "with the largest diff (better for colour-contrasting panels)")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    os.makedirs("data/roofs", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)

    _, ext   = os.path.splitext(args.roof)
    roof_out = os.path.join("data/roofs", f"{args.id}_roof{ext}")
    mask_out = os.path.join("data/masks", f"{args.id}_mask.png")

    crop_margins = tuple(args.crop)

    print(f"[1/3] Copying clean roof  → {roof_out}")
    shutil.copy2(args.roof, roof_out)

    print(f"[2/3] Extracting mask     → {mask_out}")
    mask, clean_cropped = extract_mask(
        args.roof, args.layout,
        threshold=args.threshold,
        min_area=args.min_area,
        crop_margins=crop_margins,
        enhance=args.enhance,
        diff_mode=args.diff_mode,
    )
    cv2.imwrite(mask_out, mask)

    if not args.no_preview:
        print("[3/3] Opening preview — verify the green overlay covers the panels")
        preview(clean_cropped, mask, args.id)
    else:
        print("[3/3] Preview skipped")

    print(f"\nDone. Add this row to data/metadata.csv:")
    print(f"  {args.id}_roof{ext},{args.id}_mask.png,"
          f"<num_panels>,<roof_type>,<angle>,<num_strings>,<connection_type>")


if __name__ == "__main__":
    main()
