"""
Batch-process all unprocessed house pairs in data/raw/.

A "pair" is a file matching *_roof.{jpg,png} that has a corresponding
*_layout.{jpg,png} in the same directory. Any pair whose mask already
exists in data/masks/ is skipped.

Usage (new default — RGB panel-cut mode):
    python scripts/prepare_batch.py --panel-template data/panel_template.jpg

Legacy binary-mask mode:
    python scripts/prepare_batch.py --diff --no-preview
"""

import argparse
import os
import re
import subprocess
import sys

RAW_DIR   = os.path.join("data", "raw")
MASKS_DIR = os.path.join("data", "masks")
IMG_EXTS  = (".jpg", ".jpeg", ".png")


def find_pairs(skip_existing: bool = True) -> list[tuple[str, str, str]]:
    files = os.listdir(RAW_DIR)
    roof_pattern = re.compile(r"^(.+)_roof(\.(jpg|jpeg|png))$", re.IGNORECASE)

    pairs = []
    for fname in sorted(files):
        m = roof_pattern.match(fname)
        if not m:
            continue

        house_id = m.group(1)

        layout_file = None
        for ext in IMG_EXTS:
            candidate = f"{house_id}_layout{ext}"
            if candidate in files:
                layout_file = candidate
                break

        if layout_file is None:
            print(f"  [skip] {house_id}: no matching layout image in {RAW_DIR}")
            continue

        mask_file = os.path.join(MASKS_DIR, f"{house_id}_mask.png")
        if skip_existing and os.path.exists(mask_file):
            print(f"  [skip] {house_id}: mask already exists")
            continue

        pairs.append((
            house_id,
            os.path.join(RAW_DIR, fname),
            os.path.join(RAW_DIR, layout_file),
        ))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Batch panel-cut / mask extraction for all unprocessed pairs.")

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diff",       action="store_true",
                      help="Legacy: binary diff-based mask instead of RGB panel cut")

    # Panel detection
    parser.add_argument("--panel-template",  default=None,
                        help="Path to a single panel reference image for template-matching detection "
                             "(recommended — used in layout-cut mode and diff/cell-masks mode)")
    parser.add_argument("--match-thresh",    type=float, default=0.55)
    parser.add_argument("--nms-iou",         type=float, default=0.3)
    parser.add_argument("--cell-min-area",   type=int,   default=300)
    parser.add_argument("--cell-color-tol",  type=int,   default=35)

    # Shared
    parser.add_argument("--crop",     type=int, nargs=4, default=[0, 0, 0, 0],
                        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"))
    parser.add_argument("--enhance",  action="store_true")
    parser.add_argument("--no-preview",    action="store_true")
    parser.add_argument("--reprocess-all", action="store_true",
                        help="Re-run even for houses that already have a mask")

    # Legacy diff options
    parser.add_argument("--threshold",     type=int, default=25)
    parser.add_argument("--min-area",      type=int, default=1000)
    parser.add_argument("--diff-mode",     choices=["gray", "max-channel"], default="gray")
    parser.add_argument("--cell-masks",    action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        print(f"Raw directory not found: {RAW_DIR}")
        sys.exit(1)

    pairs = find_pairs(skip_existing=not args.reprocess_all)

    if not pairs:
        print("Nothing to process.")
        return

    print(f"\nProcessing {len(pairs)} pair(s):\n")

    for i, (house_id, roof_path, layout_path) in enumerate(pairs, 1):
        print(f"── [{i}/{len(pairs)}] {house_id} ──")

        cmd = [
            sys.executable, "scripts/prepare_single.py",
            "--roof",    roof_path,
            "--layout",  layout_path,
            "--id",      house_id,
            "--crop",    *[str(v) for v in args.crop],
            "--cell-min-area",  str(args.cell_min_area),
            "--cell-color-tol", str(args.cell_color_tol),
        ]

        if args.diff:
            cmd += [
                "--diff",
                "--threshold", str(args.threshold),
                "--min-area",  str(args.min_area),
                "--diff-mode", args.diff_mode,
            ]
            if args.cell_masks:
                cmd.append("--cell-masks")

        if args.panel_template:
            cmd += ["--panel-template", args.panel_template,
                    "--match-thresh",   str(args.match_thresh),
                    "--nms-iou",        str(args.nms_iou)]
        if args.enhance:
            cmd.append("--enhance")
        if args.no_preview:
            cmd.append("--no-preview")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  ERROR: prepare_single.py exited with code {result.returncode}")
            print("  Continuing with next pair...\n")
        else:
            print()

    print("Batch complete.")


if __name__ == "__main__":
    main()
