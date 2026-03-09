"""
Augment Cityscapes training images with Ghana-specific transforms and save to disk.

Usage:
    python augment_ghana.py [--out ghana_augmented] [--comparisons 5]

Output structure:
    ghana_augmented/
    ├── train/
    │   ├── img/          # augmented training images
    │   └── label/        # labels copied as-is (resized)
    ├── val/
    │   ├── img/          # val images (resized, no augmentation)
    │   └── label/        # val labels (resized)
    └── comparisons/      # side-by-side original|augmented for inspection
"""

import os
import glob
import random
import argparse
import shutil

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from dotenv import load_dotenv
import kagglehub

# ── Constants ────────────────────────────────────────────────────────────────

ROAD_COLOR = (128, 64, 128)
COLOR_TOLERANCE = 5
IMG_SIZE = (512, 1024)  # H, W


# ── GhanaAugmentor ──────────────────────────────────────────────────────────

class GhanaAugmentor:
    """Domain-specific augmentations to bridge the Cityscapes → Ghana gap."""

    def warm_color_shift(self, img_np):
        """Shift color temperature warmer: boost red, reduce blue."""
        img = img_np.astype(np.float32)
        red_boost = random.uniform(5, 20)
        blue_reduce = random.uniform(5, 20)
        img[:, :, 0] = np.clip(img[:, :, 0] + red_boost, 0, 255)   # R
        img[:, :, 2] = np.clip(img[:, :, 2] - blue_reduce, 0, 255)  # B
        return img.astype(np.uint8)

    def saturation_boost(self, img_np):
        """Boost saturation to simulate vivid tropical vegetation."""
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = random.uniform(1.2, 1.6)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def road_marking_degradation(self, img_np, road_mask):
        """Blur road pixels to fade lane markings."""
        ksize = random.choice(range(5, 16, 2))  # odd kernel 5–15
        blurred = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
        mask_3c = road_mask[:, :, None].astype(bool)
        img_np = np.where(mask_3c, blurred, img_np)
        return img_np

    def simulated_potholes(self, img_np, road_mask):
        """Stamp dark ellipses on road regions to simulate potholes/patches."""
        num = random.randint(3, 8)
        road_ys, road_xs = np.where(road_mask > 0)
        if len(road_ys) == 0:
            return img_np
        img_out = img_np.copy()
        for _ in range(num):
            idx = random.randint(0, len(road_ys) - 1)
            cy, cx = int(road_ys[idx]), int(road_xs[idx])
            axes = (random.randint(10, 40), random.randint(10, 40))
            angle = random.randint(0, 180)
            intensity = random.randint(30, 80)
            color = (intensity, intensity, intensity)
            cv2.ellipse(img_out, (cx, cy), axes, angle, 0, 360, color, -1)
        # Only keep changes inside road mask
        mask_3c = road_mask[:, :, None].astype(bool)
        img_np = np.where(mask_3c, img_out, img_np)
        return img_np

    def edge_clutter(self, img_np, road_mask):
        """Add noise along road mask boundaries to simulate edge clutter."""
        band_width = random.randint(20, 40)
        kernel = np.ones((band_width, band_width), np.uint8)
        dilated = cv2.dilate(road_mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(road_mask.astype(np.uint8), kernel, iterations=1)
        edge_band = ((dilated - eroded) > 0)
        sigma = random.uniform(20, 40)
        noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
        img_float = img_np.astype(np.float32)
        edge_3c = edge_band[:, :, None]
        img_float = np.where(edge_3c, img_float + noise, img_float)
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def __call__(self, img_np, road_mask):
        """Apply Ghana-specific augmentations randomly.

        Global color transforms (1-3): 50% probability each.
        Road-aware transforms (4-6): 40% probability each.
        """
        # 1. Brightness/contrast via ColorJitter
        if random.random() < 0.5:
            pil_img = Image.fromarray(img_np)
            jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4)
            pil_img = jitter(pil_img)
            img_np = np.array(pil_img)

        # 2. Warm color temperature shift
        if random.random() < 0.5:
            img_np = self.warm_color_shift(img_np)

        # 3. Saturation boost
        if random.random() < 0.5:
            img_np = self.saturation_boost(img_np)

        # 4. Road marking degradation
        if random.random() < 0.4:
            img_np = self.road_marking_degradation(img_np, road_mask)

        # 5. Simulated potholes/patches
        if random.random() < 0.4:
            img_np = self.simulated_potholes(img_np, road_mask)

        # 6. Edge clutter
        if random.random() < 0.4:
            img_np = self.edge_clutter(img_np, road_mask)

        return img_np


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_road_mask(seg_np):
    """Binary road mask from an RGB segmentation array."""
    seg16 = seg_np.astype(np.int16)
    diff = np.abs(seg16 - np.array(ROAD_COLOR, dtype=np.int16))
    return (diff.max(axis=2) <= COLOR_TOLERANCE).astype(np.uint8)


def resolve_dirs(dataset_path):
    """Locate train/val img/label dirs with fallback search."""
    train_img_dir = os.path.join(dataset_path, "train", "img")
    train_seg_dir = os.path.join(dataset_path, "train", "label")
    val_img_dir = os.path.join(dataset_path, "val", "img")
    val_seg_dir = os.path.join(dataset_path, "val", "label")

    for split, dirs in [("train", [train_img_dir, train_seg_dir]),
                        ("val", [val_img_dir, val_seg_dir])]:
        if not os.path.isdir(dirs[0]):
            for root, subdirs, _ in os.walk(dataset_path):
                if split in subdirs:
                    candidate = os.path.join(root, split)
                    sub = os.listdir(candidate)
                    if "img" in sub:
                        dirs[0] = os.path.join(candidate, "img")
                        dirs[1] = os.path.join(candidate, "label")
                        if split == "train":
                            train_img_dir, train_seg_dir = dirs
                        else:
                            val_img_dir, val_seg_dir = dirs
                        break
            print(f"Resolved {split} dirs → images: {dirs[0]}, labels: {dirs[1]}")

    return train_img_dir, train_seg_dir, val_img_dir, val_seg_dir


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Augment Cityscapes images with Ghana-specific transforms and save to disk."
    )
    parser.add_argument("--out", type=str, default="ghana_augmented",
                        help="Output directory (default: ghana_augmented)")
    parser.add_argument("--comparisons", type=int, default=5,
                        help="Number of side-by-side comparison images to save (default: 5)")
    args = parser.parse_args()

    # ── Download / locate dataset ────────────────────────────────────────
    load_dotenv()
    api_key = os.environ.get("KAGGLE_API_KEY") or os.environ.get("KAGGLE_KEY")
    if api_key:
        os.environ["KAGGLE_KEY"] = api_key
        if not os.environ.get("KAGGLE_USERNAME"):
            os.environ["KAGGLE_USERNAME"] = "__token__"

    print("Downloading Cityscapes dataset …")
    dataset_path = kagglehub.dataset_download("shuvoalok/cityscapes")
    print(f"Dataset path: {dataset_path}")

    train_img_dir, train_seg_dir, val_img_dir, val_seg_dir = resolve_dirs(dataset_path)

    for d in [train_img_dir, train_seg_dir, val_img_dir, val_seg_dir]:
        assert os.path.isdir(d), f"Directory not found: {d}"

    # ── Create output directories ────────────────────────────────────────
    out_train_img = os.path.join(args.out, "train", "img")
    out_train_lbl = os.path.join(args.out, "train", "label")
    out_val_img = os.path.join(args.out, "val", "img")
    out_val_lbl = os.path.join(args.out, "val", "label")
    out_cmp = os.path.join(args.out, "comparisons")

    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl, out_cmp]:
        os.makedirs(d, exist_ok=True)

    augmentor = GhanaAugmentor()

    # ── Augment training images ──────────────────────────────────────────
    train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))
    train_labels = sorted(glob.glob(os.path.join(train_seg_dir, "*.*")))
    assert len(train_images) == len(train_labels), "Train image/label count mismatch"

    comparison_indices = set(random.sample(range(len(train_images)),
                                           min(args.comparisons, len(train_images))))

    print(f"\nAugmenting {len(train_images)} training images …")
    for i, (img_path, lbl_path) in enumerate(tqdm(zip(train_images, train_labels),
                                                   total=len(train_images),
                                                   desc="Augmenting train")):
        img = Image.open(img_path).convert("RGB")
        seg = Image.open(lbl_path).convert("RGB")

        # Random horizontal flip (applied consistently to both)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        seg = seg.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)

        img_np = np.array(img)
        seg_np = np.array(seg)
        road_mask = extract_road_mask(seg_np)

        original = img_np.copy()
        augmented = augmentor(img_np, road_mask)

        # Save augmented image and label
        fname = os.path.basename(img_path)
        name, _ = os.path.splitext(fname)
        Image.fromarray(augmented).save(os.path.join(out_train_img, f"{name}.png"))
        seg.save(os.path.join(out_train_lbl, f"{name}.png"))

        # Save comparison if selected
        if i in comparison_indices:
            combined = np.concatenate([original, augmented], axis=1)
            Image.fromarray(combined).save(os.path.join(out_cmp, f"compare_{name}.png"))

    # ── Copy val images (resize only, no augmentation) ───────────────────
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.*")))
    val_labels = sorted(glob.glob(os.path.join(val_seg_dir, "*.*")))
    assert len(val_images) == len(val_labels), "Val image/label count mismatch"

    print(f"\nResizing {len(val_images)} val images (no augmentation) …")
    for img_path, lbl_path in tqdm(zip(val_images, val_labels),
                                    total=len(val_images), desc="Resizing val"):
        img = Image.open(img_path).convert("RGB")
        seg = Image.open(lbl_path).convert("RGB")

        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        seg = seg.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)

        fname = os.path.basename(img_path)
        name, _ = os.path.splitext(fname)
        img.save(os.path.join(out_val_img, f"{name}.png"))
        seg.save(os.path.join(out_val_lbl, f"{name}.png"))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDone!")
    print(f"  Augmented train: {out_train_img} ({len(train_images)} images)")
    print(f"  Val (resized):   {out_val_img} ({len(val_images)} images)")
    print(f"  Comparisons:     {out_cmp} ({len(comparison_indices)} side-by-side images)")
    print(f"\nInspect comparisons, then run:  python train_ghana.py --data {args.out}")


if __name__ == "__main__":
    main()
