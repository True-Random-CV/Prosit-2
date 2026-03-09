# Sprint 3 — Vision-Language-Assisted Failure Mining for Domain Adaptation

## Overview

This notebook implements **Sprint 3** of a domain-adaptive perception pipeline for Ghanaian dashcam footage. The core challenge is that segmentation and detection models trained on Western datasets (Cityscapes, BDD100K) fail on Ghanaian roads due to **domain shift**: unpaved surfaces, laterite dust, dense informal markets, unique vehicle types (okada motorcycles), and unfamiliar road infrastructure.

Sprint 3 addresses this by using **Vision-Language Models (VLMs)** — CLIP and BLIP — to automatically discover _where_ the models fail, generate pseudo-labels for those failure cases, and retrain both the segmentation and detection models without any manual annotation.

### Key Results

| Model | Cityscapes Val IoU | Ghana Mean Entropy ↓ | Uncertain Pixels ↓ |
|---|---|---|---|
| Cityscapes-only (baseline) | 0.8446 | 0.1112 | 7.35% |
| Ghana-augmented (Sprint 2) | 0.8198 | 0.0956 | 5.07% |
| **Hard-neg retrained (Sprint 3)** | **0.8612** | **0.0526** | **1.96%** |

The retrained model achieves the highest Cityscapes validation IoU (no regression) while cutting uncertainty on Ghanaian frames by **73%** relative to the baseline.

---

## Pipeline Architecture

The notebook runs 8 sequential sections:

```
┌─────────────────────────────────────────────────────────┐
│ Section 1: Setup & Model Loading                        │
│   Load DeepLabV3+ (seg), RoadAwareDetector (det),       │
│   extract 3,797 Ghanaian dashcam frames                 │
├─────────────────────────────────────────────────────────┤
│ Section 2: Segmentation & Detection Inference           │
│   Run Cityscapes + Augmented models with TTA,           │
│   compute per-pixel disagreement + entropy maps         │
├─────────────────────────────────────────────────────────┤
│ Section 3: CLIP-Based Failure Mining                    │
│   Score all frames against 7 failure-category prompts,  │
│   select 72 balanced, diverse hard negatives            │
├─────────────────────────────────────────────────────────┤
│ Section 4: BLIP Captioning                              │
│   Generate short captions for failure cases             │
│   (used in report + class hinting for detection)        │
├─────────────────────────────────────────────────────────┤
│ Section 5: Pseudo-Labelling & Self-Training             │
│   Confidence-masked pseudo-labels from teacher ensemble │
│   2 rounds of staged self-training with hard crops      │
├─────────────────────────────────────────────────────────┤
│ Section 6: Grounding DINO Detection Pseudo-Labels       │
│   Zero-shot object detection → pseudo bboxes,           │
│   fine-tune detector on pseudo + source data            │
├─────────────────────────────────────────────────────────┤
│ Section 7: Before/After Evaluation                      │
│   Compare all models on Ghanaian frames + Cityscapes    │
│   val (regression check), detection comparison          │
├─────────────────────────────────────────────────────────┤
│ Section 8: Failure Analysis Report                      │
│   Generate markdown report + visualizations + catalogue │
└─────────────────────────────────────────────────────────┘
```

---

## Section-by-Section Explanation

### Section 1 — Setup & Model Loading

- **Dependencies**: `open_clip_torch`, `transformers`, `groundingdino-py`, `gdown`, `kagglehub`, `opencv-python`
- **Models loaded**:
  - **DeepLabV3+ ResNet-101** (2 checkpoints: Cityscapes-only and Ghana-augmented from Sprints 1–2)
  - **RoadAwareDetector** (FPN-based anchor detector, trained on BDD100K in Sprint 2, mAP@0.5 = 0.184)
- **Data**: 3,797 frames extracted from Ghanaian dashcam videos
- All models use ImageNet normalisation and are resized to 512×1024 (segmentation) or 448×800 (detection)

### Section 2 — Segmentation & Detection Inference with TTA

Both segmentation models are run on all 3,797 frames using **Test-Time Augmentation (TTA)**: each prediction is the average of the original frame and its horizontal flip, producing smoother probability maps.

From the two models' outputs, per-frame metrics are computed:
- **Disagreement**: fraction of pixels where the two models predict opposite classes
- **Uncertainty**: fraction of pixels with road probability between 0.3 and 0.7
- **Mean entropy**: pixel-level prediction entropy (lower = more confident)

The detection model is also run on all frames with road-aware spatial filtering (suppresses detections far from road regions).

### Section 3 — CLIP-Based Failure Mining

**What it does**: Uses CLIP (ViT-B/32) to score every frame against 7 categories of failure-pattern text prompts:

| Category | Example Prompt |
|---|---|
| `shadow_confusion` | "road with heavy shadow patterns confusing segmentation" |
| `dust_texture` | "dusty unpaved laterite road in Africa" |
| `road_degradation` | "heavily degraded asphalt road with potholes" |
| `vehicle_confusion` | "motorbike taxi on a busy road" |
| `pedestrian_occlusion` | "crowded roadside blocking the road boundary" |
| `low_light` | "dark underexposed dashcam frame" |
| `market_clutter` | "informal market stalls beside a road" |

**How hard negatives are selected**: The pipeline combines three signals into a composite score:
1. **CLIP category score** (text-image similarity to failure prompts)
2. **Segmentation uncertainty** (pixel-level entropy)
3. **Model disagreement** (Cityscapes vs Augmented predictions diverge)

A balanced quota-based system ensures representation from each failure category. Diversity is enforced by limiting frames per video clip and using CLIP embedding distance to avoid near-duplicates.

**Result**: 72 hard negatives selected across 7 categories:
- `dust_texture`: 30 (41.7%)
- `road_degradation`: 30 (41.7%)
- `vehicle_confusion`: 5 (6.9%)
- `pedestrian_occlusion`: 4 (5.6%)
- `low_light`, `market_clutter`, `speed_bumps`: 1 each

### Section 4 — BLIP Captioning

Each hard negative is captioned using **BLIP** (Salesforce/blip-image-captioning-base) with two prompts:
- General: "a dashcam photo of"
- Road-focused: "this road scene shows"

The captions serve two purposes:
1. Human-readable descriptions in the failure analysis report
2. Class hints for detection pseudo-labelling (e.g., if BLIP mentions "motorcycle", the detector's proposals for that class are prioritised)

### Section 5 — Confidence-Masked Pseudo-Labelling & Iterative Self-Training

This is the core training section. It executes in **2 rounds**:

#### Pseudo-Label Generation
For each hard negative frame, a teacher ensemble (Cityscapes + Augmented + previous round's model) generates pseudo-labels:
- **Consensus voting**: Only pixels where all teachers agree are labelled
- **Disagreement masking**: Pixels where teachers disagree by > 0.25 are set to `IGNORE_INDEX = 255`
- **Per-pixel weights**: Proportional to teacher confidence (higher agreement → higher weight)
- **Post-processing**: Morphological closing/opening + largest connected component selection

#### Training
- **Data mix**: Cityscapes supervised labels + Ghana pseudo-labels + focus crops around uncertain boundary regions
- **Loss function**: Weighted Cross-Entropy + Soft BCE + Weighted Dice Loss (combined for robust gradient flow)
- **Staged backbone unfreezing**:
  - Epochs 1–2 (Stage 0): Only classifier head, LR = 5×10⁻⁴
  - Epochs 3–6 (Stage 1): Head + backbone layer4, LR = [5×10⁻⁴, 1×10⁻⁴]
  - Epochs 7–10 (Stage 2): Head + layer4 + layer3, LR = [5×10⁻⁴, 1×10⁻⁴, 5×10⁻⁵]
- **Learning rate decay**: LR halved at epoch 6

**Training Results**:

| Round | Best Epoch | Best Val IoU |
|---|---|---|
| Round 1 | 8 | 0.8573 |
| Round 2 | 5 | 0.8485 |

Round 1 achieved the best validation IoU; the model from Round 1 Epoch 8 was selected as the final checkpoint.

### Section 6 — Grounding DINO Detection Pseudo-Labels & Detector Fine-Tuning

**Grounding DINO** is a state-of-the-art zero-shot, text-prompted object detector. Unlike the circular approach of using the Sprint 2 detector to generate its own training labels, Grounding DINO is an independent external teacher.

**How it works**:
1. Grounding DINO receives a text prompt: `"car . taxi . bus . truck . lorry . person . pedestrian . motorcycle rider . motorcycle . motorbike . bicycle . traffic sign . traffic light ."`
2. For each of the 72 hard negative frames, it detects objects matching these prompts
3. Detected phrases are mapped to the 9-class BDD100K detection taxonomy
4. Per-class NMS removes duplicate boxes (IoU threshold = 0.5)

**Compatibility Note**: The `groundingdino-py` package requires two runtime patches for compatibility with transformers ≥ 4.45:
- `BertModel.get_head_mask` (removed in newer transformers, shimmed back)
- `get_extended_attention_mask` (old code passes `device` where new API expects `dtype`)

**Result**: 64 frames received valid pseudo bounding boxes (785 total boxes).

**Detector Fine-Tuning**:
- Backbone (ResNet-101) is frozen; only FPN + detection head are trained
- 6 epochs on pseudo-labelled Ghana frames (oversampled 4×), no BDD100K source data was available in this run
- Training loss decreased from 1.17 → 0.20

### Section 7 — Before/After Evaluation

#### Segmentation Comparison

| Metric | Cityscapes-only | Ghana-augmented | Hard-neg retrained |
|---|---|---|---|
| Mean Road Probability | 0.1510 | 0.2197 | 0.2100 |
| Mean Entropy | 0.1112 | 0.0956 | **0.0526** |
| Road Ratio | 0.1540 | 0.2224 | 0.2114 |
| Uncertain Pixels | 7.35% | 5.07% | **1.96%** |
| Cityscapes Val IoU | 0.8446 | 0.8198 | **0.8612** |

The retrained model halves the entropy and nearly eliminates uncertain pixels on Ghanaian data, while simultaneously **improving** Cityscapes validation IoU (no regression).

#### Per-Category Entropy Improvement

| Category | Before (Augmented) | After (Retrained) | Improvement |
|---|---|---|---|
| speed_bumps | 0.0931 | 0.0344 | −63% |
| vehicle_confusion | 0.1044 | 0.0460 | −56% |
| road_degradation | 0.1092 | 0.0522 | −52% |
| dust_texture | 0.1031 | 0.0510 | −51% |
| pedestrian_occlusion | 0.0946 | 0.0425 | −55% |
| market_clutter | 0.1021 | 0.0618 | −39% |
| low_light | 0.0648 | 0.0870 | **+34%** ⚠️ |

The `low_light` category shows entropy _increase_ after retraining — this is because only 1 frame was selected for this category, providing insufficient training signal.

#### Detection Comparison

| Configuration | Total Dets | Dets/Frame | Mean Score |
|---|---|---|---|
| Original det + Cityscapes seg | 3,804 | 1.0 | 0.4039 |
| Original det + Augmented seg | 5,341 | 1.4 | 0.4166 |
| Original det + Retrained seg | 3,842 | 1.0 | 0.4066 |
| **Retrained det + Retrained seg** | **12,254** | **3.2** | 0.3518 |

The retrained detector produces 3.2× more detections per frame, but with a lower mean confidence score (0.35 vs 0.40). Without ground-truth annotations, it is not possible to determine whether these additional detections are true positives or false positives.

### Section 8 — Failure Analysis Report

Generates:
- `failure_analysis_report.md` — structured markdown report with all tables and findings
- `failure_catalogue.json` — machine-readable per-frame failure metadata
- Before/after comparison images saved to `sprint3_output/visualizations/`
- Pseudo-label previews saved to `sprint3_output/pseudo_labels/`

---

## Technical Components

### Models Used

| Model | Purpose | Source |
|---|---|---|
| DeepLabV3+ ResNet-101 | Road segmentation | Cityscapes-pretrained (Sprint 1) |
| RoadAwareDetector (FPN) | Object detection | BDD100K-trained (Sprint 2), mAP@0.5 = 0.184 |
| CLIP ViT-B/32 | Failure mining, crop verification | OpenAI / LAION2B weights |
| BLIP-base | Image captioning | Salesforce `blip-image-captioning-base` |
| Grounding DINO SwinT | Zero-shot detection pseudo-labels | IDEA Research |

### Key Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| `HARD_NEG_COUNT` | 300 (72 selected) | Target hard negatives |
| `SELF_TRAIN_ROUNDS` | 2 | Iterative self-training rounds |
| `ROUND_EPOCHS` | 10 | Epochs per round |
| `WARMUP_EPOCHS` | 2 | Head-only warmup epochs |
| `PSEUDO_DISAGREE_MARGIN` | 0.25 | Minimum margin for teacher agreement |
| `GDINO_BOX_THRESH` | 0.25 | Grounding DINO confidence threshold |
| `DET_EPOCHS` | 6 | Detection fine-tuning epochs |

---

## Limitations

### 1. No Ground-Truth Annotations for Ghanaian Data
The most fundamental limitation: there are **no annotated bounding boxes or segmentation masks** for the Ghanaian dashcam frames. All evaluation on the target domain is proxy-based (entropy, uncertainty, detection count) rather than direct accuracy/IoU metrics. The Cityscapes validation IoU confirms no regression on the source domain but does not measure target-domain accuracy.

### 2. Small Hard Negative Pool
Only **72 hard negatives** (out of 3,797 frames) were selected for retraining. This is a small fraction of the training data, and some failure categories are severely underrepresented:
- `low_light`: 1 frame — too few to learn from (entropy actually increased after training)
- `market_clutter`: 1 frame
- `speed_bumps`: 1 frame

A larger pool (200–500 frames) with more balanced category representation would likely improve results.

### 3. Pseudo-Label Noise
Despite confidence masking, the pseudo-labels are generated by models that themselves struggle on the target domain. Approximately **7% of pixels** are masked as uncertain in Round 1, but the remaining "confident" labels are not guaranteed to be correct. This ceiling limits how much the self-training loop can improve.

### 4. Detection Evaluation Without mAP
The retrained detector triples detection count (3.2 dets/frame vs 1.0), but the mean confidence drops from 0.40 to 0.35. Without ground-truth boxes, it is impossible to compute mAP or determine the precision/recall trade-off. The additional detections could be:
- **True positives** the original model missed (vehicles, pedestrians on laterite roads)
- **False positives** caused by overfitting to noisy Grounding DINO pseudo-labels

### 5. No BDD100K Source Data in Detection Training
The detector was fine-tuned on **pseudo-labels only** (64 Ghana frames, oversampled 4×) without any BDD100K source data mixing. This creates a risk of catastrophic forgetting of general detection capabilities. Ideally, a 50/50 mix of source and target data should be used.

### 6. Grounding DINO Compatibility
The `groundingdino-py` package requires runtime monkey-patches for compatibility with `transformers >= 4.45`. This is fragile and may break with future library updates. Two specific patches are applied:
- `BertModel.get_head_mask` (removed in newer transformers)
- `get_extended_attention_mask` dtype/device argument swap

### 7. Binary Segmentation Only
The segmentation model is reduced to a **2-class problem** (road vs non-road). This loses fine-grained Cityscapes class information (sidewalk, building, vegetation, etc.). Multi-class adaptation would require different pseudo-labelling strategies.

### 8. Single Dataset / Geography
The pipeline is tested on dashcam footage from **one city in Ghana**. Generalisation to other West African cities, rural roads, or different seasons/weather conditions is not evaluated.

### 9. CLIP Prompt Engineering
The failure categories and CLIP text prompts were manually designed. Different prompt formulations could yield substantially different hard negative selections. No systematic prompt search or optimisation was performed.

---

## File Structure

```
Prosit 2 Final/
├── sprint3_final (1).ipynb     # Full executed notebook with outputs
├── failure_analysis_report.md  # Generated failure analysis summary
└── README.md                   # This document
```

### Output Artifacts (on Google Drive)

```
sprint3_output/
├── hard_negative_retrained_model.pth   # Best segmentation checkpoint
├── retrained_detector.pth              # Fine-tuned detection checkpoint
├── failure_catalogue.json              # Per-frame failure metadata
├── groundingdino_swint_ogc.pth         # Cached GDINO weights
├── failure_report/
│   └── failure_analysis_report.md      # Full report
├── pseudo_labels/                      # Pseudo-label preview images
└── visualizations/                     # Before/after comparison images
```

---

## Runtime

Executed on **Google Colab with G4 GPU** (NVIDIA T4, 16 GB VRAM). Total runtime: approximately **40–50 minutes**.

| Section | Approximate Time |
|---|---|
| Setup & Model Loading | 2 min |
| Segmentation Inference (TTA) | 4.5 min |
| Detection Inference | 3 min |
| CLIP Mining | 3 min |
| BLIP Captioning | 2 min |
| Self-Training (2 rounds × 10 epochs) | 15 min |
| Grounding DINO Pseudo-Labels | 1 min |
| Detector Fine-Tuning (6 epochs) | 1 min |
| Evaluation | 12 min |
| Report Generation | < 1 min |

---

## Dependencies

```
torch >= 2.0
torchvision
open_clip_torch
transformers
groundingdino-py
gdown
kagglehub
opencv-python
Pillow
tqdm
python-dotenv
```

## References

- [DeepLabV3+](https://arxiv.org/abs/1802.02611) — Chen et al., 2018
- [CLIP](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [BLIP](https://arxiv.org/abs/2201.12086) — Li et al., 2022
- [Grounding DINO](https://arxiv.org/abs/2303.05499) — Liu et al., 2023
- [Cityscapes](https://www.cityscapes-dataset.com/) — Cordts et al., 2016
- [BDD100K](https://bdd-data.berkeley.edu/) — Yu et al., 2020
