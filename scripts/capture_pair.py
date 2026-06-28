"""
Capture a layout+roof screenshot pair from the Pylon browser app
and immediately run mask extraction.

Usage (first time):
    python scripts/capture_pair.py --id house_006

    1. A full screenshot opens — click the TOP-LEFT then BOTTOM-RIGHT
       corner of the map area. That region is saved to data/capture_config.json
       and reused on every future run.

    2. Switch to your browser (panels visible). The script waits --delay seconds,
       then captures the layout screenshot.

    3. Press Enter in the terminal when ready to delete panels.
       The script clicks the map centre to focus the window, then sends Ctrl+A → Del.

    4. After a short pause the clean-roof screenshot is captured automatically.

    5. Mask extraction runs and the preview opens (pass --no-preview to skip).

To re-draw the capture region:
    python scripts/capture_pair.py --id house_006 --reset-region
"""

import argparse
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pyautogui

REGION_CONFIG = "data/capture_config.json"


# ── region helpers ──────────────────────────────────────────────────────────

def select_region(delay: int) -> dict:
    print(f"\nSwitch to your browser now — taking a full screenshot in {delay}s...")
    time.sleep(delay)

    shot = pyautogui.screenshot()
    img = np.array(shot)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(img)
    ax.set_title(
        "Click TOP-LEFT corner of the map area, then BOTTOM-RIGHT corner.\n"
        "Close the window or wait 60 s to cancel.",
        fontsize=11,
    )
    plt.tight_layout()

    pts = plt.ginput(2, timeout=60)
    plt.close()

    if len(pts) < 2:
        raise RuntimeError("Region selection cancelled — no points received.")

    x1, y1 = int(pts[0][0]), int(pts[0][1])
    x2, y2 = int(pts[1][0]), int(pts[1][1])

    region = {
        "left":   min(x1, x2),
        "top":    min(y1, y2),
        "width":  abs(x2 - x1),
        "height": abs(y2 - y1),
    }
    os.makedirs("data", exist_ok=True)
    with open(REGION_CONFIG, "w") as f:
        json.dump(region, f, indent=2)

    print(f"Region saved to {REGION_CONFIG}: {region}")
    return region


def load_region() -> dict | None:
    if not os.path.exists(REGION_CONFIG):
        return None
    with open(REGION_CONFIG) as f:
        return json.load(f)


def capture_region(region: dict):
    return pyautogui.screenshot(
        region=(region["left"], region["top"], region["width"], region["height"])
    )


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture a Pylon screenshot pair and build the mask.")
    parser.add_argument("--id",           required=True,  help="House ID e.g. house_006")
    parser.add_argument("--delay",        type=int, default=4,
                        help="Seconds to switch to browser before each capture (default 4)")
    parser.add_argument("--delete-pause", type=int, default=4,
                        help="Seconds to wait after Ctrl+A+Del for panels to disappear (default 2)")
    parser.add_argument("--reset-region", action="store_true",
                        help="Ignore saved region and select a new one")
    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diff", action="store_true",
                      help="Use legacy binary diff-based mask instead of RGB panel cut")

    # Panel detection
    parser.add_argument("--panel-template",  default=None,
                        help="Path to a panel reference image for template-matching detection "
                             "(recommended for layout-cut mode)")
    parser.add_argument("--match-thresh",    type=float, default=0.55)
    parser.add_argument("--nms-iou",         type=float, default=0.3)
    parser.add_argument("--cell-min-area",   type=int, default=300)
    parser.add_argument("--cell-color-tol",  type=int, default=35)

    # Dark-roof preset (still useful for diff mode)
    parser.add_argument("--dark-roof",    action="store_true",
                        help="Preset for dark roofs: enables CLAHE, max-channel diff, "
                             "threshold 15, min-area 800.")
    parser.add_argument("--threshold",    type=int, default=25)
    parser.add_argument("--min-area",     type=int, default=1000)
    parser.add_argument("--enhance",      action="store_true")
    parser.add_argument("--diff-mode",    choices=["gray", "max-channel"], default="gray")
    parser.add_argument("--cell-masks",   action="store_true")
    parser.add_argument("--no-preview",   action="store_true")
    args = parser.parse_args()

    if args.dark_roof:
        args.enhance    = True
        args.diff_mode  = "max-channel"
        args.threshold  = 15
        args.min_area   = 800

    os.makedirs("data/raw", exist_ok=True)

    # ── 1. region setup ──────────────────────────────────────────────────
    region = load_region()
    if region is None or args.reset_region:
        region = select_region(args.delay)
    else:
        print(f"Using saved region: {region}  (use --reset-region to change)")

    # centre of the region — used to click-focus the browser map before key input
    cx = region["left"] + region["width"]  // 2
    cy = region["top"]  + region["height"] // 2

    # ── 2. layout screenshot (panels visible) ────────────────────────────
    print(f"\n[1/4] Switch to Pylon in your browser with panels visible.")
    print(f"      Capturing layout in {args.delay}s...")
    time.sleep(args.delay)

    layout_shot = capture_region(region)
    layout_path = os.path.join("data", "raw", f"{args.id}_layout.jpg")
    layout_shot.save(layout_path, quality=95)
    print(f"      Saved: {layout_path}")

    # ── 3. delete panels ─────────────────────────────────────────────────
    print("\n[2/4] Press Enter here when ready to delete panels.")
    print("      (Make sure the Pylon tab is in focus.)")
    input("      > ")

    # click the map canvas to ensure browser has focus, not terminal
    pyautogui.click(cx, cy)
    time.sleep(0.3)

    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.3)
    pyautogui.press("delete")
    print(f"      Ctrl+A + Del sent. Waiting {args.delete_pause}s for panels to disappear...")
    time.sleep(args.delete_pause)

    # ── 4. clean roof screenshot ─────────────────────────────────────────
    print("[3/4] Capturing clean roof...")
    roof_shot = capture_region(region)
    roof_path = os.path.join("data", "raw", f"{args.id}_roof.jpg")
    roof_shot.save(roof_path, quality=95)
    print(f"      Saved: {roof_path}")

    # ── 5. run mask extraction ───────────────────────────────────────────
    print("\n[4/4] Running panel extraction...")
    cmd = [
        sys.executable, "scripts/prepare_single.py",
        "--roof",   roof_path,
        "--layout", layout_path,
        "--id",     args.id,
        "--cell-min-area",  str(args.cell_min_area),
        "--cell-color-tol", str(args.cell_color_tol),
    ]
    if args.diff:
        cmd += ["--diff",
                "--threshold", str(args.threshold),
                "--min-area",  str(args.min_area),
                "--diff-mode", args.diff_mode]
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

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
