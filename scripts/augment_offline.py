"""
Generate offline augmented copies of every labelled house and write a combined
metadata CSV that includes both originals and synthetic samples.

Augmented files are saved alongside real data using the naming convention:
  house_001__aug0_roof.jpg / house_001__aug0_mask.png

Usage:
    python scripts/augment_offline.py
    python scripts/augment_offline.py --copies 4 --seed 7
    python scripts/augment_offline.py --input-csv data/metadata.csv --output-csv data/metadata_augmented.csv
    python scripts/augment_offline.py --dry-run      # preview counts only, no files written

The output CSV keeps all original rows first, then appended synthetic rows.
Re-running skips files that already exist (safe to resume after interruption).
"""

import argparse
import os
import sys

import albumentations as A
import cv2
import numpy as np
import pandas as pd


# ── augmentation pipeline ────────────────────────────────────────────────────
# Heavier than the training-time pipeline — we want permanently distinct images
# so the model sees genuinely different perspectives, not just in-batch variation.

def build_pipeline(seed: int, copy_index: int) -> A.Compose:
    rng = np.random.default_rng(seed + copy_index * 1000)
    random_state = int(rng.integers(0, 2**31))

    return A.Compose(
        [
            # — geometric (applied to image + mask together) —
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                scale=(0.80, 1.20),
                rotate=(-30, 30),
                shear=(-8, 8),
                p=0.8,
            ),
            A.Perspective(scale=(0.03, 0.08), p=0.3),
            # — colour (image only — mask is uint8 label, not touched) —
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.GaussNoise(p=0.3),
        ],
        additional_targets={"mask": "mask"},
        seed=random_state,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def load_pair(row: pd.Series, roof_dir: str, mask_dir: str):
    img_path  = os.path.join(roof_dir, row["image_filename"])
    mask_path = os.path.join(mask_dir, row["mask_filename"])

    img  = cv2.imread(img_path)
    mask = cv2.imread(mask_path)  # 3-channel BGR — RGB panel cut

    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot open mask:  {mask_path}")

    return img, mask


def aug_filename(original_filename: str, copy_index: int, suffix: str) -> str:
    base, ext = os.path.splitext(original_filename)
    return f"{base}__aug{copy_index}_{suffix}"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline augmentation — multiply training data without Pylon captures."
    )
    parser.add_argument(
        "--input-csv",  default="data/metadata.csv",
        help="Source metadata CSV (default: data/metadata.csv)",
    )
    parser.add_argument(
        "--output-csv", default="data/metadata_augmented.csv",
        help="Output CSV with originals + synthetics (default: data/metadata_augmented.csv)",
    )
    parser.add_argument(
        "--copies", type=int, default=3,
        help="Augmented copies per original image (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--roof-dir", default="data/roofs",
        help="Directory containing roof images (default: data/roofs)",
    )
    parser.add_argument(
        "--mask-dir", default="data/masks",
        help="Directory containing mask images (default: data/masks)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be created without writing any files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"ERROR: input CSV not found: {args.input_csv}")
        sys.exit(1)

    meta = pd.read_csv(args.input_csv)
    n_originals = len(meta)

    print("=" * 60)
    print("  Offline Augmentation")
    print("=" * 60)
    print(f"\n  Input CSV    : {args.input_csv}  ({n_originals} originals)")
    print(f"  Copies/image : {args.copies}")
    print(f"  Total output : {n_originals * (args.copies + 1)} rows")
    print(f"  Output CSV   : {args.output_csv}")
    if args.dry_run:
        print("\n  [DRY RUN] No files will be written.\n")

    new_rows = []
    skipped  = 0
    written  = 0

    for idx, row in meta.iterrows():
        house_id = row["image_filename"].replace("_roof.jpg", "").replace("_roof.png", "")

        try:
            img, mask = load_pair(row, args.roof_dir, args.mask_dir)
        except FileNotFoundError as e:
            print(f"  [warn] {e} — skipping")
            continue

        for copy_i in range(args.copies):
            img_fname  = aug_filename(row["image_filename"],  copy_i, "roof.jpg")
            mask_fname = aug_filename(row["mask_filename"],   copy_i, "mask.png")

            img_out  = os.path.join(args.roof_dir, img_fname)
            mask_out = os.path.join(args.mask_dir, mask_fname)

            if not args.dry_run:
                if os.path.exists(img_out) and os.path.exists(mask_out):
                    skipped += 1
                else:
                    pipeline = build_pipeline(args.seed, copy_i)
                    result   = pipeline(image=img, mask=mask)
                    cv2.imwrite(img_out,  result["image"])
                    cv2.imwrite(mask_out, result["mask"])
                    written += 1

            new_rows.append({
                "image_filename":  img_fname,
                "mask_filename":   mask_fname,
                "num_panels":      row["num_panels"],
                "roof_type":       row["roof_type"],
                "angle":           row["angle"],
                "num_strings":     row["num_strings"],
                "connection_type": row["connection_type"],
            })

        progress = idx + 1
        if progress % 10 == 0 or progress == n_originals:
            print(f"  [{progress:3d}/{n_originals}] {house_id}")

    combined = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)

    if not args.dry_run:
        combined.to_csv(args.output_csv, index=False)

    print(f"\n  Originals  : {n_originals}")
    print(f"  Synthetic  : {len(new_rows)}")
    print(f"  Total rows : {len(combined)}")
    if not args.dry_run:
        print(f"  Written    : {written}  |  Already existed (skipped): {skipped}")
        print(f"\n  Saved → {args.output_csv}")
        print("\n  To use during training, update configs/default.yaml:")
        print("    data:")
        print(f"      metadata_csv: {args.output_csv}")
    else:
        print("\n  [DRY RUN] Nothing written.")


if __name__ == "__main__":
    main()
