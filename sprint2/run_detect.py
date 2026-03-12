"""
Run inference with the trained detection model + optional road segmentation filtering.

Usage:
    # Single image
    python run_detect.py --image path/to/image.jpg

    # Directory of images
    python run_detect.py --image_dir path/to/images/

    # Without road-aware filtering
    python run_detect.py --image photo.jpg --no_road_filter

    # Custom thresholds
    python run_detect.py --image photo.jpg --score_thresh 0.4 --iou_thresh 0.45

    # Print detections to terminal (no image saving)
    python run_detect.py --image photo.jpg --print_only

Prerequisites:
    1. Detection checkpoint (from Modal training):
       modal volume get road-detection-vol checkpoints/best_detector.pth .
    2. (Optional) Road segmentation checkpoint: best_road_model_ghana.pth
    3. (Optional) DeepLabV3Plus-Pytorch/ repo cloned locally (for road seg model)
"""

import os
import sys
import argparse
import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_detect import (
    RoadAwareDetector, road_aware_filter,
    DETECT_CLASSES, NUM_DETECT_CLASSES,
)

# ── Per-class colors (BGR for cv2) ──────────────────────────────────────────

CLASS_COLORS = {
    "car":           (0, 200, 0),
    "bus":           (0, 165, 255),
    "truck":         (255, 100, 0),
    "pedestrian":    (0, 0, 255),
    "rider":         (0, 255, 255),
    "motorcycle":    (128, 0, 128),
    "bicycle":       (255, 255, 0),
    "traffic sign":  (0, 128, 255),
    "traffic light": (255, 0, 255),
}


# ── Model loading ───────────────────────────────────────────────────────────

def load_detection_model(ckpt_path, device):
    """Load detection model from checkpoint. Returns (model, img_h, img_w)."""
    print(f"Loading detection checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Read image dimensions from checkpoint (support both old and new format)
    img_h = ckpt.get("img_h", 448)
    img_w = ckpt.get("img_w", ckpt.get("img_size", 800))

    model = RoadAwareDetector(num_classes=NUM_DETECT_CLASSES, pretrained_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # Print checkpoint info
    epoch = ckpt.get("epoch", "?")
    mAP = ckpt.get("mAP_50", ckpt.get("val_loss", "?"))
    print(f"  Epoch {epoch}, mAP@0.5={mAP}")
    print(f"  Input size: {img_h}x{img_w}")

    ap50 = ckpt.get("per_class_ap50", {})
    if ap50:
        parts = [f"{DETECT_CLASSES[c]}={ap:.3f}" for c, ap in sorted(ap50.items())]
        print(f"  Per-class AP@0.5: {', '.join(parts)}")

    return model, img_h, img_w


def load_segmentation_model(ckpt_path, device):
    """Load the road segmentation DeepLabV3+ model."""
    repo_dir = "DeepLabV3Plus-Pytorch"
    if not os.path.isdir(repo_dir):
        print(f"  Warning: {repo_dir}/ not found, cannot load segmentation model")
        return None

    sys.path.insert(0, repo_dir)
    from network.modeling import deeplabv3plus_resnet101
    import torch.nn as nn

    print(f"Loading segmentation checkpoint: {ckpt_path}")
    model = deeplabv3plus_resnet101(num_classes=19, output_stride=16)
    model.classifier.classifier[3] = nn.Conv2d(256, 2, kernel_size=1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  Val IoU: {ckpt.get('val_iou', '?')}")
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def detect(det_model, img_pil, img_h, img_w, device):
    """Run detection on a single PIL image. Returns {boxes, scores, labels}."""
    det_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = det_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        results = det_model(inp)
    return results[0]


def get_road_mask(seg_model, img_pil, device):
    """Run segmentation and return binary road mask (H, W) bool tensor."""
    seg_transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = seg_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = seg_model(inp)
    return (out.argmax(dim=1) == 1).squeeze(0).cpu()


# ── Visualization ────────────────────────────────────────────────────────────

def draw_detections(img_bgr, boxes, scores, labels, img_h, img_w):
    """Draw bounding boxes on the image. Scales boxes from model coords to image coords."""
    h, w = img_bgr.shape[:2]
    sx = w / img_w
    sy = h / img_h

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].cpu().numpy()
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)

        cls_id = labels[i].item()
        score = scores[i].item()
        cls_name = DETECT_CLASSES[cls_id]
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        label_text = f"{cls_name} {score:.2f}"

        # Box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img_bgr


def print_detections(boxes, scores, labels, img_h, img_w, orig_w, orig_h):
    """Print detections to terminal in a readable format."""
    sx = orig_w / img_w
    sy = orig_h / img_h

    if len(boxes) == 0:
        print("    No detections.")
        return

    print(f"    {'Class':<16} {'Score':>6}   {'Box (x1,y1,x2,y2)'}")
    print(f"    {'─'*16} {'─'*6}   {'─'*30}")
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].cpu().numpy()
        cls_name = DETECT_CLASSES[labels[i].item()]
        score = scores[i].item()
        print(f"    {cls_name:<16} {score:>6.3f}   "
              f"({int(x1*sx)}, {int(y1*sy)}, {int(x2*sx)}, {int(y2*sy)})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run detection with optional road-aware filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detect.py --image photo.jpg
  python run_detect.py --image_dir test_images/ --score_thresh 0.4
  python run_detect.py --image photo.jpg --no_road_filter --print_only
        """,
    )
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Directory of images")
    parser.add_argument("--out", type=str, default="detections_out",
                        help="Output directory (default: detections_out)")
    parser.add_argument("--det_ckpt", type=str, default="best_detector.pth",
                        help="Detection checkpoint path")
    parser.add_argument("--seg_ckpt", type=str, default="best_road_model_ghana.pth",
                        help="Road segmentation checkpoint (for road-aware filtering)")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="Min confidence score to display (default: 0.3)")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--no_road_filter", action="store_true",
                        help="Disable road-aware filtering")
    parser.add_argument("--print_only", action="store_true",
                        help="Only print detections to terminal, don't save images")
    args = parser.parse_args()

    # ── Collect image paths ──
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_paths = sorted(
            glob.glob(os.path.join(args.image_dir, "*.jpg")) +
            glob.glob(os.path.join(args.image_dir, "*.jpeg")) +
            glob.glob(os.path.join(args.image_dir, "*.png"))
        )
    else:
        parser.error("Provide --image or --image_dir")

    if not image_paths:
        print("No images found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load detection model ──
    if not os.path.isfile(args.det_ckpt):
        print(f"Error: Detection checkpoint not found: {args.det_ckpt}")
        print("Download it from Modal volume:")
        print("  modal volume get road-detection-vol checkpoints/best_detector.pth .")
        return

    det_model, img_h, img_w = load_detection_model(args.det_ckpt, device)

    # ── Load segmentation model (optional) ──
    seg_model = None
    if not args.no_road_filter:
        if os.path.isfile(args.seg_ckpt):
            seg_model = load_segmentation_model(args.seg_ckpt, device)
        else:
            print(f"Segmentation checkpoint not found ({args.seg_ckpt}), "
                  f"running without road-aware filtering.\n")

    if not args.print_only:
        os.makedirs(args.out, exist_ok=True)

    # ── Process images ──
    print(f"\nProcessing {len(image_paths)} image(s) …\n")
    total_detections = 0

    for path in image_paths:
        img_pil = Image.open(path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Run detection
        result = detect(det_model, img_pil, img_h, img_w, device)
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]

        # Road-aware filtering
        if seg_model is not None and boxes.shape[0] > 0:
            road_mask = get_road_mask(seg_model, img_pil, device)
            boxes, scores, labels = road_aware_filter(
                boxes, scores, labels, road_mask, img_h, img_w,
            )

        # Score threshold
        keep = scores > args.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        total_detections += len(boxes)
        fname = os.path.basename(path)

        print(f"  {fname}: {len(boxes)} detections")
        print_detections(boxes, scores, labels, img_h, img_w, orig_w, orig_h)

        # Save annotated image
        if not args.print_only:
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_bgr = draw_detections(img_bgr, boxes, scores, labels, img_h, img_w)

            # Add info bar at top
            info = (f"Detections: {len(boxes)} | "
                    f"Thresh: {args.score_thresh} | "
                    f"Road filter: {'ON' if seg_model else 'OFF'}")
            cv2.putText(img_bgr, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            out_path = os.path.join(args.out, fname)
            cv2.imwrite(out_path, img_bgr)

        print()

    # ── Summary ──
    print("=" * 50)
    print(f"Total: {total_detections} detections across {len(image_paths)} image(s)")
    if not args.print_only:
        print(f"Output: ./{args.out}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
