"""
Automated road annotation for Ghanaian dashcam images.

Uses HSV color thresholding + spatial priors + morphological cleanup
to generate binary road masks (ground truth for IoU evaluation).

Usage:
    python auto_annotate.py --input sample_test/ --output sample_test_labels/
    python auto_annotate.py --input sample_test/ --output sample_test_labels/ --preview
"""

import os
import glob
import argparse

import cv2
import numpy as np
from PIL import Image


def annotate_road(img_bgr):
    """
    Generate a binary road mask from a dashcam image.

    Strategy:
      1. HSV color filter: road is gray (low saturation), medium brightness.
      2. Texture filter: road is smooth (low local gradient variance).
      3. Trapezoidal spatial prior: road fans out from vanishing point.
      4. Combine color + texture + spatial, then flood-fill from bottom-center.
      5. Morphological cleanup.

    Returns:
        Binary mask (H, W) with 255 = road, 0 = not-road.
    """
    h, w = img_bgr.shape[:2]

    # --- Step 1: HSV color filter ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)

    # Road asphalt: low saturation, mid-range value
    # Tighter thresholds to exclude walls/buildings
    color_mask = (sat < 50) & (val > 50) & (val < 180)

    # Exclude greenish pixels (vegetation has hue ~35-85 in OpenCV)
    green_mask = (hue > 25) & (hue < 90) & (sat > 20)
    color_mask = color_mask & ~green_mask

    # --- Step 2: Texture filter (smooth regions) ---
    # Road surface is smoother than walls, vegetation, gravel
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Local variance in a 15x15 window
    gray_f = gray.astype(np.float32)
    blur = cv2.GaussianBlur(gray_f, (15, 15), 0)
    blur_sq = cv2.GaussianBlur(gray_f ** 2, (15, 15), 0)
    local_var = blur_sq - blur ** 2
    local_var = np.clip(local_var, 0, None)

    # Smooth regions have low variance — road is smoother than textured walls
    smooth_mask = local_var < 400

    # --- Step 3: Trapezoidal spatial prior ---
    # Road in dashcam: wide at bottom, narrows toward vanishing point ~center-top
    spatial = np.zeros((h, w), dtype=np.uint8)
    vanish_y = int(h * 0.40)  # approximate vanishing point height
    vanish_x = int(w * 0.50)
    bot_y = int(h * 0.92)     # above dashboard

    # Trapezoid: narrow at top, wide at bottom
    trap_top_half = int(w * 0.12)   # narrow opening at vanishing point
    trap_bot_half = int(w * 0.50)   # full width at bottom

    pts = np.array([
        [vanish_x - trap_top_half, vanish_y],
        [vanish_x + trap_top_half, vanish_y],
        [vanish_x + trap_bot_half, bot_y],
        [vanish_x - trap_bot_half, bot_y],
    ], dtype=np.int32)
    cv2.fillConvexPoly(spatial, pts, 255)

    # --- Step 4: Combine all three cues ---
    combined = (color_mask.astype(np.uint8) * 255) & (smooth_mask.astype(np.uint8) * 255) & spatial

    # --- Step 5: Morphological cleanup ---
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    # --- Step 6: Keep largest component touching bottom-center ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)

    if num_labels <= 1:
        return combined

    # Seed from bottom-center strip
    seed_top = int(h * 0.78)
    seed_left = int(w * 0.30)
    seed_right = int(w * 0.70)
    seed_region = labels[seed_top:bot_y, seed_left:seed_right]
    seed_labels = set(np.unique(seed_region)) - {0}

    if not seed_labels:
        # Fallback: largest component
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = np.argmax(areas) + 1
        seed_labels = {best}

    final_mask = np.zeros_like(combined)
    for lbl in seed_labels:
        final_mask[labels == lbl] = 255

    # Final smoothing
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_smooth)

    return final_mask


def create_preview(img_bgr, mask, alpha=0.4):
    """Overlay road mask on the original image (green tint)."""
    overlay = img_bgr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) +
                          np.array([0, 200, 0]) * alpha).astype(np.uint8)

    # Draw mask contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate roads in dashcam images.")
    parser.add_argument("--input", type=str, default="sample_test",
                        help="Input image directory")
    parser.add_argument("--output", type=str, default="sample_test_labels",
                        help="Output mask directory")
    parser.add_argument("--preview", action="store_true",
                        help="Save overlay previews alongside masks")
    parser.add_argument("--preview_dir", type=str, default="sample_test_preview",
                        help="Directory for preview images")
    args = parser.parse_args()

    # Collect images
    paths = sorted(
        glob.glob(os.path.join(args.input, "*.jpg")) +
        glob.glob(os.path.join(args.input, "*.jpeg")) +
        glob.glob(os.path.join(args.input, "*.png"))
    )
    if not paths:
        print(f"No images found in {args.input}/")
        return

    os.makedirs(args.output, exist_ok=True)
    if args.preview:
        os.makedirs(args.preview_dir, exist_ok=True)

    print(f"Auto-annotating {len(paths)} images ...\n")

    for path in paths:
        fname = os.path.basename(path)
        name, _ = os.path.splitext(fname)

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"  Skipping {fname} (cannot read)")
            continue

        mask = annotate_road(img_bgr)

        # Road pixel percentage
        h, w = mask.shape
        road_pct = (mask > 0).sum() / (h * w) * 100

        # Save mask as PNG (black/white)
        mask_path = os.path.join(args.output, f"{name}.png")
        cv2.imwrite(mask_path, mask)

        print(f"  {fname}: road = {road_pct:.1f}% of image")

        # Save preview
        if args.preview:
            preview = create_preview(img_bgr, mask)
            preview_path = os.path.join(args.preview_dir, f"{name}_preview.jpg")
            cv2.imwrite(preview_path, preview)

    print(f"\nMasks saved to: {args.output}/")
    if args.preview:
        print(f"Previews saved to: {args.preview_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
