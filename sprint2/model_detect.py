"""
Anchor-based object detection model for Ghana road scenes.

Custom implementation from scratch:
- FPN neck on ResNet101 backbone
- Anchor generation, encoding/decoding
- Focal loss, Smooth L1, BCE objectness loss
- Non-maximum suppression (pure PyTorch, no library shortcuts)
- Road-aware detection filtering using segmentation mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

# ── Classes ──────────────────────────────────────────────────────────────────

DETECT_CLASSES = [
    "car", "bus", "truck", "pedestrian", "rider",
    "motorcycle", "bicycle", "traffic sign", "traffic light",
]
NUM_DETECT_CLASSES = len(DETECT_CLASSES)

# BDD100K category name → our class index
BDD100K_CLASS_MAP = {
    "car": 0, "bus": 1, "truck": 2, "pedestrian": 3, "rider": 4,
    "motorcycle": 5, "bicycle": 6, "traffic sign": 7, "traffic light": 8,
}

# Classes expected on/near the road (used for road-aware filtering)
ROAD_RELEVANT_CLASSES = {0, 1, 2, 3, 4, 5, 6}  # vehicles + pedestrian + rider


# ── IoU ──────────────────────────────────────────────────────────────────────

def compute_iou_matrix(boxes1, boxes2):
    """IoU between two sets of boxes in x1y1x2y2 format.
    boxes1: (N, 4), boxes2: (M, 4) → returns (N, M).
    """
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=1e-6)


# ── Bbox Encoding / Decoding ────────────────────────────────────────────────

def encode_boxes(anchors, gt_boxes):
    """Encode GT boxes as deltas relative to anchors. Both x1y1x2y2."""
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1)
    a_h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1)

    g_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    g_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    g_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1)
    g_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1)

    dx = (g_cx - a_cx) / a_w
    dy = (g_cy - a_cy) / a_h
    dw = torch.log(g_w / a_w)
    dh = torch.log(g_h / a_h)

    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(anchors, deltas):
    """Decode predicted deltas relative to anchors → x1y1x2y2."""
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1)
    a_h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1)

    pred_cx = deltas[:, 0] * a_w + a_cx
    pred_cy = deltas[:, 1] * a_h + a_cy
    pred_w = torch.exp(deltas[:, 2].clamp(max=4.0)) * a_w
    pred_h = torch.exp(deltas[:, 3].clamp(max=4.0)) * a_h

    return torch.stack([
        pred_cx - pred_w / 2, pred_cy - pred_h / 2,
        pred_cx + pred_w / 2, pred_cy + pred_h / 2,
    ], dim=1)


# ── Anchor Generator ────────────────────────────────────────────────────────

class AnchorGenerator:
    def __init__(self, base_sizes=(32, 64, 128),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3))):
        self.base_sizes = base_sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales  # sub-octave scales per level (RetinaNet-style)

    @property
    def num_anchors_per_cell(self):
        return len(self.aspect_ratios) * len(self.scales)

    def _make_cell_anchors(self, base_size, device):
        anchors = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                size = base_size * scale
                h = size / np.sqrt(ratio)
                w = size * np.sqrt(ratio)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def __call__(self, feature_sizes, strides, device):
        """Generate anchors for all FPN levels.
        Returns concatenated (total_anchors, 4) tensor.
        """
        all_anchors = []
        for (fh, fw), stride, base_size in zip(feature_sizes, strides, self.base_sizes):
            cell_anchors = self._make_cell_anchors(base_size, device)  # (A, 4)

            shift_x = (torch.arange(fw, device=device, dtype=torch.float32) + 0.5) * stride
            shift_y = (torch.arange(fh, device=device, dtype=torch.float32) + 0.5) * stride
            sy, sx = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack([sx, sy, sx, sy], dim=-1).reshape(-1, 4)  # (fh*fw, 4)

            level_anchors = (shifts[:, None, :] + cell_anchors[None, :, :]).reshape(-1, 4)
            all_anchors.append(level_anchors)

        return torch.cat(all_anchors, dim=0)


# ── FPN Neck ─────────────────────────────────────────────────────────────────

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        self.output = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list]
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        """features: [C3, C4, C5] low-to-high stride."""
        lats = [l(f) for l, f in zip(self.lateral, features)]
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i + 1], size=lats[i].shape[2:], mode="nearest")
        return [o(l) for o, l in zip(self.output, lats)]


# ── Detection Head ───────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3, num_classes=NUM_DETECT_CLASSES):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Classification subnet (4 conv + ReLU)
        cls_layers = []
        for _ in range(4):
            cls_layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(True)]
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.obj_score = nn.Conv2d(in_channels, num_anchors, 3, padding=1)

        # Regression subnet (4 conv + ReLU)
        reg_layers = []
        for _ in range(4):
            reg_layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(True)]
        self.reg_subnet = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Prior probability ~0.01 for foreground
        prior_bias = -np.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_score.bias, prior_bias)
        nn.init.constant_(self.obj_score.bias, prior_bias)

    def forward(self, fpn_features):
        """Returns concatenated predictions across all FPN levels.
        cls: (B, total_anchors, num_classes)
        obj: (B, total_anchors, 1)
        reg: (B, total_anchors, 4)
        """
        all_cls, all_obj, all_reg = [], [], []
        for feat in fpn_features:
            B, _, H, W = feat.shape
            cf = self.cls_subnet(feat)
            cls = self.cls_score(cf).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            obj = self.obj_score(cf).permute(0, 2, 3, 1).reshape(B, -1, 1)
            rf = self.reg_subnet(feat)
            reg = self.reg_pred(rf).permute(0, 2, 3, 1).reshape(B, -1, 4)
            all_cls.append(cls)
            all_obj.append(obj)
            all_reg.append(reg)
        return torch.cat(all_cls, 1), torch.cat(all_obj, 1), torch.cat(all_reg, 1)


# ── Loss Functions ───────────────────────────────────────────────────────────

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss.
    logits: (N, C)  targets: (N,) class indices (all ≥ 0).
    """
    if logits.numel() == 0:
        return logits.sum() * 0.0

    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
    p_t = p * one_hot + (1 - p) * (1 - one_hot)
    alpha_t = alpha * one_hot + (1 - alpha) * (1 - one_hot)
    loss = alpha_t * (1 - p_t) ** gamma * ce

    return loss.sum() / max(logits.shape[0], 1)


def smooth_l1_loss(preds, targets, beta=1.0 / 9):
    """Smooth L1 / Huber loss for bbox regression."""
    if preds.numel() == 0:
        return preds.sum() * 0.0
    diff = torch.abs(preds - targets)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.sum() / max(preds.shape[0], 1)


def objectness_loss(logits, targets):
    """BCE objectness. logits: (N,1), targets: (N,) in {0,1,-1}."""
    valid = targets >= 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[valid].squeeze(-1), targets[valid].float(), reduction="mean"
    )


# ── Anchor–Target Matching ───────────────────────────────────────────────────

def match_anchors_to_targets(anchors, gt_boxes, gt_labels, pos_iou=0.5, neg_iou=0.4):
    """Assign each anchor a target.
    Returns:
        cls_targets (N,): class id for positives, -1 otherwise
        obj_targets (N,): 1=pos, 0=neg, -1=ignore
        reg_targets (N,4): encoded deltas for positives
    """
    N = anchors.shape[0]
    device = anchors.device
    cls_targets = torch.full((N,), -1, dtype=torch.long, device=device)
    obj_targets = torch.full((N,), -1, dtype=torch.long, device=device)
    reg_targets = torch.zeros((N, 4), dtype=torch.float32, device=device)

    if gt_boxes.shape[0] == 0:
        obj_targets[:] = 0
        return cls_targets, obj_targets, reg_targets

    iou = compute_iou_matrix(anchors, gt_boxes)  # (N, M)
    max_iou, max_idx = iou.max(dim=1)

    # Negatives
    obj_targets[max_iou < neg_iou] = 0

    # Positives
    pos = max_iou >= pos_iou
    obj_targets[pos] = 1
    cls_targets[pos] = gt_labels[max_idx[pos]]
    reg_targets[pos] = encode_boxes(anchors[pos], gt_boxes[max_idx[pos]])

    # Ensure every GT box has at least one positive anchor
    best_anchor = iou.argmax(dim=0)  # (M,)
    for gi, ai in enumerate(best_anchor):
        obj_targets[ai] = 1
        cls_targets[ai] = gt_labels[gi]
        reg_targets[ai] = encode_boxes(anchors[ai:ai + 1], gt_boxes[gi:gi + 1]).squeeze(0)

    return cls_targets, obj_targets, reg_targets


# ── Custom NMS ───────────────────────────────────────────────────────────────

def nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS — pure PyTorch, no library calls.
    boxes: (N,4) x1y1x2y2, scores: (N,). Returns kept indices.
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    suppressed = torch.zeros(len(order), dtype=torch.bool, device=boxes.device)

    for i in range(len(order)):
        if suppressed[i]:
            continue
        idx = order[i]
        keep.append(idx)

        rest = order[i + 1:]
        if len(rest) == 0:
            break

        xx1 = torch.max(boxes[idx, 0], boxes[rest, 0])
        yy1 = torch.max(boxes[idx, 1], boxes[rest, 1])
        xx2 = torch.min(boxes[idx, 2], boxes[rest, 2])
        yy2 = torch.min(boxes[idx, 3], boxes[rest, 3])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[idx] + areas[rest] - inter).clamp(min=1e-6)

        for j in range(len(rest)):
            if iou[j] > iou_threshold:
                suppressed[i + 1 + j] = True

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def multiclass_nms(boxes, cls_scores, obj_scores,
                   score_thresh=0.05, iou_thresh=0.5, max_det=100):
    """Per-class NMS. Combines objectness × class confidence."""
    scores = cls_scores * obj_scores.unsqueeze(1)  # (N, C)
    num_classes = scores.shape[1]

    det_boxes, det_scores, det_labels = [], [], []
    for c in range(num_classes):
        cs = scores[:, c]
        mask = cs > score_thresh
        if not mask.any():
            continue
        kept = nms(boxes[mask], cs[mask], iou_thresh)
        det_boxes.append(boxes[mask][kept])
        det_scores.append(cs[mask][kept])
        det_labels.append(torch.full((len(kept),), c, dtype=torch.long, device=boxes.device))

    if len(det_boxes) == 0:
        z = boxes.device
        return torch.zeros(0, 4, device=z), torch.zeros(0, device=z), torch.zeros(0, dtype=torch.long, device=z)

    det_boxes = torch.cat(det_boxes)
    det_scores = torch.cat(det_scores)
    det_labels = torch.cat(det_labels)

    if len(det_scores) > max_det:
        topk = det_scores.argsort(descending=True)[:max_det]
        det_boxes, det_scores, det_labels = det_boxes[topk], det_scores[topk], det_labels[topk]

    return det_boxes, det_scores, det_labels


# ── Road-Aware Filtering ────────────────────────────────────────────────────

def road_aware_filter(boxes, scores, labels, road_mask, img_h, img_w):
    """Suppress vehicle/pedestrian detections whose bottom-center is off-road.
    Traffic signs/lights are always kept (they're above the road).
    road_mask: (H_mask, W_mask) binary, 1 = road.
    """
    if boxes.shape[0] == 0:
        return boxes, scores, labels

    mask_h, mask_w = road_mask.shape
    keep = []

    for i in range(boxes.shape[0]):
        if labels[i].item() not in ROAD_RELEVANT_CLASSES:
            keep.append(i)
            continue

        # Bottom-center of the detection box
        bcx = (boxes[i, 0] + boxes[i, 2]) / 2
        bcy = boxes[i, 3]

        mx = int((bcx / img_w * mask_w).clamp(0, mask_w - 1).item())
        my = int((bcy / img_h * mask_h).clamp(0, mask_h - 1).item())

        # Check a small neighborhood around the bottom-center
        r = 5
        patch = road_mask[max(0, my - r):my + r + 1, max(0, mx - r):mx + r + 1]
        if patch.any():
            keep.append(i)

    if not keep:
        d = boxes.device
        return torch.zeros(0, 4, device=d), torch.zeros(0, device=d), torch.zeros(0, dtype=torch.long, device=d)

    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[keep], scores[keep], labels[keep]


# ── Full Detector ────────────────────────────────────────────────────────────

class RoadAwareDetector(nn.Module):
    """Anchor-based FPN detector on ResNet101 backbone."""

    STRIDES = [8, 16, 32]
    BASE_SIZES = [32, 64, 128]
    ASPECT_RATIOS = [0.5, 1.0, 2.0]
    SCALES = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]  # RetinaNet sub-octave scales

    def __init__(self, num_classes=NUM_DETECT_CLASSES, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone (ResNet101) ──
        weights = "IMAGENET1K_V1" if pretrained_backbone else None
        resnet = models.resnet101(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # stride 4,  256 ch
        self.layer2 = resnet.layer2   # stride 8,  512 ch
        self.layer3 = resnet.layer3   # stride 16, 1024 ch
        self.layer4 = resnet.layer4   # stride 32, 2048 ch

        # ── FPN ──
        self.fpn = FPN([512, 1024, 2048], out_channels=256)

        # ── Anchor generator (9 anchors per cell: 3 ratios × 3 scales) ──
        self.anchor_gen = AnchorGenerator(
            self.BASE_SIZES, self.ASPECT_RATIOS, self.SCALES
        )
        self.num_anchors = self.anchor_gen.num_anchors_per_cell

        # ── Detection head ──
        self.head = DetectionHead(256, self.num_anchors, num_classes)

        self._anchor_cache = {}

    def _get_anchors(self, feat_sizes, device):
        key = tuple(feat_sizes)
        if key not in self._anchor_cache:
            self._anchor_cache[key] = self.anchor_gen(feat_sizes, self.STRIDES, device)
        return self._anchor_cache[key]

    def backbone_features(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)   # stride 8
        c4 = self.layer3(c3)   # stride 16
        c5 = self.layer4(c4)   # stride 32
        return [c3, c4, c5]

    def forward(self, images, targets=None):
        """
        Training:  forward(images, targets) → dict of losses
        Inference: forward(images)           → list of {boxes, scores, labels}

        targets: list[dict] with keys 'boxes' (N,4 x1y1x2y2) and 'labels' (N,)
        """
        feats = self.backbone_features(images)
        fpn_out = self.fpn(feats)

        feat_sizes = [(f.shape[2], f.shape[3]) for f in fpn_out]
        anchors = self._get_anchors(feat_sizes, images.device)

        cls_logits, obj_logits, reg_preds = self.head(fpn_out)

        if targets is not None:
            return self._losses(cls_logits, obj_logits, reg_preds, anchors, targets)
        return self._predict(cls_logits, obj_logits, reg_preds, anchors, images.shape[2:])

    # ── Training losses ──

    def _losses(self, cls_logits, obj_logits, reg_preds, anchors, targets):
        B = cls_logits.shape[0]
        dev = cls_logits.device

        loss_cls = torch.tensor(0.0, device=dev)
        loss_obj = torch.tensor(0.0, device=dev)
        loss_reg = torch.tensor(0.0, device=dev)

        for i in range(B):
            gt_boxes = targets[i]["boxes"].to(dev)
            gt_labels = targets[i]["labels"].to(dev)

            ct, ot, rt = match_anchors_to_targets(anchors, gt_boxes, gt_labels)
            pos = ot == 1

            if pos.any():
                loss_cls = loss_cls + focal_loss(cls_logits[i][pos], ct[pos])
                loss_reg = loss_reg + smooth_l1_loss(reg_preds[i][pos], rt[pos])
            loss_obj = loss_obj + objectness_loss(obj_logits[i], ot)

        n = max(B, 1)
        return {
            "cls_loss": loss_cls / n,
            "obj_loss": loss_obj / n,
            "reg_loss": loss_reg / n,
            "total_loss": (loss_cls + loss_obj + loss_reg) / n,
        }

    # ── Inference ──

    def _predict(self, cls_logits, obj_logits, reg_preds, anchors, img_size,
                 score_thresh=0.05, iou_thresh=0.5, max_det=100):
        results = []
        img_h, img_w = img_size
        for i in range(cls_logits.shape[0]):
            cls_p = torch.sigmoid(cls_logits[i])
            obj_p = torch.sigmoid(obj_logits[i]).squeeze(-1)
            decoded = decode_boxes(anchors, reg_preds[i])
            # Clamp to image bounds
            decoded[:, 0::2] = decoded[:, 0::2].clamp(0, img_w)
            decoded[:, 1::2] = decoded[:, 1::2].clamp(0, img_h)

            boxes, scores, labels = multiclass_nms(
                decoded, cls_p, obj_p, score_thresh, iou_thresh, max_det
            )
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results


# ── mAP Evaluation ──────────────────────────────────────────────────────────

def _compute_ap(recalls, precisions):
    """Compute AP using the 11-point interpolation (VOC-style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_above = precisions[recalls >= t]
        p = float(precisions_above.max()) if len(precisions_above) > 0 else 0.0
        ap += p / 11
    return ap


def compute_ap_per_class(pred_boxes, pred_scores, pred_labels,
                         gt_boxes, gt_labels, iou_threshold=0.5,
                         num_classes=NUM_DETECT_CLASSES):
    """Compute per-class Average Precision.

    Args:
        pred_boxes:  list[Tensor(N_i, 4)] per image
        pred_scores: list[Tensor(N_i,)]   per image
        pred_labels: list[Tensor(N_i,)]   per image
        gt_boxes:    list[Tensor(M_i, 4)] per image
        gt_labels:   list[Tensor(M_i,)]   per image
        iou_threshold: IoU threshold for a correct detection

    Returns:
        per_class_ap: dict {class_id: AP}
        mAP: mean AP across classes that have GT instances
    """
    aps = {}

    for c in range(num_classes):
        # Gather all predictions for this class across images, with image index
        all_scores = []
        all_tp_fp = []  # 1 = TP, 0 = FP
        n_gt_total = 0

        for img_idx in range(len(gt_boxes)):
            # GT boxes for this class in this image
            gt_mask = gt_labels[img_idx] == c
            gt_c = gt_boxes[img_idx][gt_mask]
            n_gt = gt_c.shape[0]
            n_gt_total += n_gt
            matched = torch.zeros(n_gt, dtype=torch.bool)

            # Predictions for this class in this image
            pred_mask = pred_labels[img_idx] == c
            p_boxes = pred_boxes[img_idx][pred_mask]
            p_scores = pred_scores[img_idx][pred_mask]

            if p_boxes.shape[0] == 0:
                continue

            # Sort by score descending
            order = p_scores.argsort(descending=True)
            p_boxes = p_boxes[order]
            p_scores = p_scores[order]

            for j in range(p_boxes.shape[0]):
                all_scores.append(p_scores[j].item())

                if n_gt == 0:
                    all_tp_fp.append(0)
                    continue

                ious = compute_iou_matrix(
                    p_boxes[j:j + 1], gt_c
                ).squeeze(0)  # (n_gt,)

                best_iou, best_gt = ious.max(dim=0)

                if best_iou >= iou_threshold and not matched[best_gt]:
                    all_tp_fp.append(1)
                    matched[best_gt] = True
                else:
                    all_tp_fp.append(0)

        if n_gt_total == 0:
            continue  # skip classes with no GT

        if len(all_scores) == 0:
            aps[c] = 0.0
            continue

        # Sort all detections by score
        sorted_indices = np.argsort(-np.array(all_scores))
        tp_fp = np.array(all_tp_fp)[sorted_indices]

        tp_cumsum = np.cumsum(tp_fp)
        fp_cumsum = np.cumsum(1 - tp_fp)

        recalls = tp_cumsum / n_gt_total
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        aps[c] = _compute_ap(recalls, precisions)

    mAP = float(np.mean(list(aps.values()))) if aps else 0.0
    return aps, mAP


def evaluate_detections(model, data_loader, device, iou_thresholds=None,
                        num_classes=NUM_DETECT_CLASSES):
    """Run full evaluation: mAP@0.5 and mAP@0.5:0.95.

    Args:
        model: RoadAwareDetector (set to eval mode before calling)
        data_loader: yields (images, targets)
        device: torch device
        iou_thresholds: list of IoU thresholds (default: 0.5:0.05:0.95)

    Returns dict with:
        mAP_50:         mAP at IoU=0.5
        mAP_50_95:      mAP averaged over IoU 0.5:0.05:0.95
        per_class_ap50: {class_id: AP} at IoU=0.5
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_gt_boxes, all_gt_labels = [], []

    model.eval()
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = imgs.to(device)
            results = model(imgs)  # inference mode (no targets)

            for i, res in enumerate(results):
                all_pred_boxes.append(res["boxes"].cpu())
                all_pred_scores.append(res["scores"].cpu())
                all_pred_labels.append(res["labels"].cpu())
                all_gt_boxes.append(targets[i]["boxes"])
                all_gt_labels.append(targets[i]["labels"])

    # mAP@0.5
    per_class_ap50, mAP_50 = compute_ap_per_class(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels,
        iou_threshold=0.5, num_classes=num_classes,
    )

    # mAP@0.5:0.95
    maps_at_thresholds = []
    for t in iou_thresholds:
        _, m = compute_ap_per_class(
            all_pred_boxes, all_pred_scores, all_pred_labels,
            all_gt_boxes, all_gt_labels,
            iou_threshold=t, num_classes=num_classes,
        )
        maps_at_thresholds.append(m)
    mAP_50_95 = float(np.mean(maps_at_thresholds))

    return {
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "per_class_ap50": per_class_ap50,
    }
