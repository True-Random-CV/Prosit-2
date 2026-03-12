"""
Train the anchor-based road detector on BDD100K using Modal.

Setup (one-time):
    pip install modal
    modal token new
    modal secret create kaggle-secret KAGGLE_USERNAME=__token__ KAGGLE_KEY=<your_key>

Deploy (persistent web endpoint — trigger training anytime):
    modal deploy train_detect_modal.py

Run directly (blocks until done):
    modal run train_detect_modal.py

Run single subject-style (spawns and returns immediately):
    curl -X POST https://<your-modal-url>/run_training

Monitor:
    modal app logs road-detection
"""

import modal

app = modal.App("road-detection")

vol = modal.Volume.from_name("road-detection-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "tqdm",
        "kagglehub",
        "fastapi[standard]==0.115.0",
    )
    .add_local_file("model_detect.py", remote_path="/root/model_detect.py")
)

MOUNT_DIR = "/data"
CKPT_DIR = f"{MOUNT_DIR}/checkpoints"
DATA_DIR = f"{MOUNT_DIR}/bdd100k"


# ── Training function ───────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    volumes={MOUNT_DIR: vol},
    image=image,
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("kaggle-secret")],
)
def train(
    num_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    max_train_images: int = 0,
    img_h: int = 448,
    img_w: int = 800,
):
    import os
    import sys
    import json
    import random

    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    from model_detect import (
        RoadAwareDetector, BDD100K_CLASS_MAP, NUM_DETECT_CLASSES, DETECT_CLASSES,
        evaluate_detections,
    )

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # ── 1. Download BDD100K into persistent volume ─────────────────────
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    def find_file(root, name):
        for dirpath, _, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)
        return None

    def find_dir_with_images(root):
        """Find a directory that actually contains image files."""
        for dirpath, _, files in os.walk(root):
            if any(f.endswith((".jpg", ".png")) for f in files[:10]):
                return dirpath
        return None

    # Check if data already exists in the volume
    train_labels_file = find_file(DATA_DIR, "bdd100k_labels_images_train.json")

    if train_labels_file:
        print(f"BDD100K already in volume at {DATA_DIR}")
    else:
        import kagglehub
        import shutil

        print("Downloading BDD100K from Kaggle …")
        cache_path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
        print(f"Downloaded to cache: {cache_path}")

        # Copy from ephemeral cache into persistent volume
        print(f"Copying to persistent volume at {DATA_DIR} …")
        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(DATA_DIR, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        vol.commit()
        print("Dataset saved to volume.")

    # ── 2. Locate files ──────────────────────────────────────────────────
    train_labels_file = find_file(DATA_DIR, "bdd100k_labels_images_train.json")
    val_labels_file = find_file(DATA_DIR, "bdd100k_labels_images_val.json")

    # Find image dirs (may be nested like bdd100k/images/100k/train/)
    train_img_dir = find_dir_with_images(os.path.join(DATA_DIR, "images", "100k", "train")) \
        if os.path.isdir(os.path.join(DATA_DIR, "images")) else None
    val_img_dir = find_dir_with_images(os.path.join(DATA_DIR, "images", "100k", "val")) \
        if os.path.isdir(os.path.join(DATA_DIR, "images")) else None

    # Fallback: search the whole DATA_DIR
    if not train_img_dir:
        for candidate in ["train", "images"]:
            d = find_file(DATA_DIR, "")  # dummy — just search broadly
            break
        # Broad search for any dir named "train" containing images
        for dirpath, dirnames, _ in os.walk(DATA_DIR):
            if "train" in dirnames:
                found = find_dir_with_images(os.path.join(dirpath, "train"))
                if found:
                    train_img_dir = found
                    break
    if not val_img_dir:
        for dirpath, dirnames, _ in os.walk(DATA_DIR):
            if "val" in dirnames:
                found = find_dir_with_images(os.path.join(dirpath, "val"))
                if found:
                    val_img_dir = found
                    break

    assert train_labels_file, f"Could not find train labels JSON in {DATA_DIR}"
    assert val_labels_file, f"Could not find val labels JSON in {DATA_DIR}"
    assert train_img_dir, f"Could not find train image directory in {DATA_DIR}"
    assert val_img_dir, f"Could not find val image directory in {DATA_DIR}"

    print(f"Train labels: {train_labels_file}")
    print(f"Train images: {train_img_dir}")
    print(f"Val labels:   {val_labels_file}")
    print(f"Val images:   {val_img_dir}")

    # ── 3. Parse labels ──────────────────────────────────────────────────

    def parse_bdd100k_labels(json_path, max_images=None):
        with open(json_path) as f:
            data = json.load(f)

        annotations = {}
        for frame in data:
            name = frame["name"]
            boxes = []
            for label in frame.get("labels", []):
                cat = label.get("category", "")
                if cat not in BDD100K_CLASS_MAP:
                    continue
                b = label.get("box2d")
                if b is None:
                    continue
                class_id = BDD100K_CLASS_MAP[cat]
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"], class_id])
            if boxes:
                annotations[name] = boxes

        if max_images and len(annotations) > max_images:
            keys = random.sample(list(annotations.keys()), max_images)
            annotations = {k: annotations[k] for k in keys}

        return annotations

    print("Parsing train labels …")
    train_annots = parse_bdd100k_labels(
        train_labels_file, max_images=max_train_images if max_train_images > 0 else None
    )
    print(f"  {len(train_annots)} train images with annotations")

    print("Parsing val labels …")
    val_annots = parse_bdd100k_labels(val_labels_file, max_images=2000)
    print(f"  {len(val_annots)} val images with annotations")

    # ── 4. Dataset ───────────────────────────────────────────────────────

    class BDD100KDetection(Dataset):
        def __init__(self, img_dir, annotations, target_h, target_w):
            self.img_dir = img_dir
            self.target_h = target_h
            self.target_w = target_w
            self.samples = []
            for fname, boxes in annotations.items():
                path = os.path.join(img_dir, fname)
                if os.path.isfile(path):
                    self.samples.append((path, boxes))
            print(f"  Dataset: {len(self.samples)} images (of {len(annotations)} annotated)")

            self.transform = transforms.Compose([
                transforms.Resize((target_h, target_w)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, raw_boxes = self.samples[idx]
            img = Image.open(path).convert("RGB")
            orig_w, orig_h = img.size

            img = self.transform(img)

            sx = self.target_w / orig_w
            sy = self.target_h / orig_h
            boxes = []
            labels = []
            for x1, y1, x2, y2, cls_id in raw_boxes:
                boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                labels.append(cls_id)

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
                "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            }
            return img, target

    def collate_fn(batch):
        imgs, targets = zip(*batch)
        return torch.stack(imgs), list(targets)

    train_ds = BDD100KDetection(train_img_dir, train_annots, img_h, img_w)
    val_ds = BDD100KDetection(val_img_dir, val_annots, img_h, img_w)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # ── 5. Model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RoadAwareDetector(num_classes=NUM_DETECT_CLASSES, pretrained_backbone=True)

    # Freeze early backbone (stem, layer1, layer2) — keep frozen
    # Unfreeze later backbone (layer3, layer4) — fine-tune with low LR
    for name, param in model.named_parameters():
        if name.startswith(("stem.", "layer1.", "layer2.")):
            param.requires_grad = False

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── 6. Optimizer (differential LR) ────────────────────────────────────
    # Backbone layer3/4: low LR (1/10th) — fine-tune pretrained features
    # FPN + head: full LR — train from scratch
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("layer3.", "layer4.")):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )

    best_mAP = 0.0
    save_path = os.path.join(CKPT_DIR, "best_detector.pth")

    # ── 7. Training loop ─────────────────────────────────────────────────
    print(f"\nStarting training for {num_epochs} epochs …\n")

    for epoch in range(num_epochs):
        # -- Train --
        model.train()
        # Keep frozen layers in eval mode (BatchNorm)
        model.stem.eval()
        model.layer1.eval()
        model.layer2.eval()

        train_loss = 0.0
        train_cls = 0.0
        train_obj = 0.0
        train_reg = 0.0
        train_steps = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            imgs = imgs.to(device)
            losses = model(imgs, targets)

            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += losses["total_loss"].item()
            train_cls += losses["cls_loss"].item()
            train_obj += losses["obj_loss"].item()
            train_reg += losses["reg_loss"].item()
            train_steps += 1

        n = max(train_steps, 1)
        avg_train = train_loss / n

        print(
            f"Epoch {epoch+1}/{num_epochs} train: "
            f"loss={avg_train:.4f} "
            f"(cls={train_cls/n:.4f} obj={train_obj/n:.4f} reg={train_reg/n:.4f})"
        )

        # -- Evaluate (mAP via NMS decode) --
        print(f"  Running mAP evaluation on val set …")
        metrics = evaluate_detections(model, val_loader, device,
                                      num_classes=NUM_DETECT_CLASSES)

        mAP_50 = metrics["mAP_50"]
        mAP_50_95 = metrics["mAP_50_95"]

        scheduler.step()
        cur_lr = optimizer.param_groups[1]["lr"]  # head LR

        print(
            f"  val: mAP@0.5={mAP_50:.4f}, mAP@0.5:0.95={mAP_50_95:.4f}, lr={cur_lr:.6f}"
        )

        # Per-class AP breakdown
        ap50 = metrics["per_class_ap50"]
        if ap50:
            parts = [f"{DETECT_CLASSES[c]}={ap:.3f}" for c, ap in sorted(ap50.items())]
            print(f"  AP@0.5 per class: {', '.join(parts)}")

        if mAP_50 > best_mAP:
            best_mAP = mAP_50
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mAP_50": mAP_50,
                "mAP_50_95": mAP_50_95,
                "per_class_ap50": ap50,
                "classes": DETECT_CLASSES,
                "img_h": img_h,
                "img_w": img_w,
            }, save_path)
            vol.commit()
            print(f"  → Saved new best model (mAP@0.5={mAP_50:.4f})")

    # ── 8. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Best mAP@0.5:     {best_mAP:.4f}")
    print(f"  Checkpoint:       {save_path}")
    print("=" * 50)

    return {
        "best_mAP_50": float(best_mAP),
        "checkpoint": save_path,
        "epochs": num_epochs,
    }


# ── Web endpoint (deploy once, trigger anytime) ─────────────────────────────

@app.function(image=image)
@modal.asgi_app()
def training_endpoint():
    """Web endpoint to spawn training jobs."""
    import fastapi

    web_app = fastapi.FastAPI()

    @web_app.post("/run_training")
    async def trigger_training(
        epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-3,
        max_images: int = 0,
        img_h: int = 448,
        img_w: int = 800,
    ):
        """Spawn a training job that runs in the background."""
        call = train.spawn(
            num_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_train_images=max_images,
            img_h=img_h,
            img_w=img_w,
        )
        return {
            "status": "training job spawned",
            "job_id": call.object_id,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_images": max_images,
                "img_h": img_h,
                "img_w": img_w,
            },
            "monitor": "modal app logs road-detection",
        }

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    return web_app


# ── Local entrypoint (direct run) ───────────────────────────────────────────

@app.local_entrypoint()
def main(
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    max_images: int = 0,
    img_h: int = 448,
    img_w: int = 800,
):
    """
    Run training directly.

    Examples:
        modal run train_detect_modal.py                          # defaults
        modal run train_detect_modal.py --epochs 50 --lr 5e-4   # custom
    """
    result = train.remote(
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_train_images=max_images,
        img_h=img_h,
        img_w=img_w,
    )
    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)
    for key, value in result.items():
        print(f"  {key}: {value}")
