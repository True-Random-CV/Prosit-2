# Sprint 1: Road Segmentation for Ghanaian Roads

## Objective
Evaluate how well a pretrained segmentation model (DeepLabV3+ ResNet101) can segment roads in Ghana, and whether fine-tuning on Cityscapes (with and without Ghana-specific augmentations) improves performance on local Ghanaian dashcam footage.

## Dataset
- **Cityscapes** — [Kaggle](https://www.kaggle.com/datasets/shuvoalok/cityscapes)
- **Local test set** — 50 frames sampled from a Ghanaian dashcam video, with auto-generated ground truth masks (see `auto_annotate.py`)

## Model
- **Architecture:** DeepLabV3+ with ResNet101 backbone
- **Pretrained checkpoint:** [best_deeplabv3plus_resnet101_cityscapes_os16.pth](https://github.com/VainF/DeepLabV3Plus-Pytorch) (19-class Cityscapes)
- **Fine-tuned variants:** Binary road/not-road (2-class head), backbone frozen, classifier head trained for 10 epochs

## Experiments & Results

### 1. Baseline — Pretrained model on Cityscapes validation set

Evaluating the off-the-shelf 19-class Cityscapes model on the Cityscapes validation split (road = class 0).

| Metric | Value |
|---|---|
| Mean IoU (road) | 0.5785 |
| AP (road) | 0.8681 |
| AP (background) | 0.9832 |
| mAP | 0.9256 |

Results: `segmentation_output/`

### 2. Baseline — Pretrained model on local Ghanaian test video

Same pretrained 19-class model evaluated on 50 frames from a Ghanaian dashcam video.

| Metric | Value |
|---|---|
| Mean IoU (road) | 0.5392 |
| Mean Pixel Acc | 0.8609 |
| Mean Precision | 0.6133 |
| Mean Recall | 0.8202 |
| Mean F1 | 0.6966 |
| mAP (road class) | 0.6299 |

Results: `local_base_model_eval/`

### 3. Fine-tuned — Trained on Cityscapes (original only)

Fine-tuned DeepLabV3+ with a 2-class head (road/not-road) on the Cityscapes training set.

| Metric | Value |
|---|---|
| Val IoU | 0.8268 |
| Epochs | 10 |

Model: [Google Drive](https://drive.google.com/file/d/17U5tCW8tz2VF-RhyiAJJroKgGx0M8GUQ/view?usp=drive_link)

### 4. Fine-tuned — Trained on Cityscapes + Ghana augmentations

Fine-tuned with Ghana-specific augmentations applied to the training set (warm color shift, saturation boost, road marking degradation, simulated potholes, edge clutter — see `augment_ghana.py`).

| Metric | Value |
|---|---|
| Val IoU | 0.8307 |
| Epochs | 10 |

Model: [Google Drive](https://drive.google.com/file/d/1gPkIfXmT12PDzg7zrzE3X-NG_Ii5KAG3/view?usp=drive_link)

### 5. Augmented model — Evaluated on local Ghanaian test video

The Ghana-augmented fine-tuned model evaluated on the same 50 local dashcam frames.

| Metric | Value |
|---|---|
| Mean IoU (road) | 0.4582 |
| Mean Pixel Acc | 0.8458 |
| Mean Precision | 0.6047 |
| Mean Recall | 0.6525 |
| Mean F1 | 0.6236 |
| mAP (road class) | 0.5579 |

Results: `local_augment_model_eval/`

## Summary Table

| Model | Evaluated On | Mean IoU | mAP | F1 |
|---|---|---|---|---|
| Pretrained (19-class) | Cityscapes val | 0.5785 | 0.9256 | — |
| Pretrained (19-class) | Ghana local test | 0.5392 | 0.6299 | 0.6966 |
| Fine-tuned (original) | Cityscapes val | 0.8268 | — | — |
| Fine-tuned (augmented) | Cityscapes val | 0.8307 | — | — |
| Fine-tuned (augmented) | Ghana local test | 0.4582 | 0.5579 | 0.6236 |

## Key Observations
- The pretrained 19-class model actually outperforms the fine-tuned augmented model on local Ghanaian roads (IoU 0.5392 vs 0.4582). This suggests the domain gap between Cityscapes and Ghanaian roads is significant, and the augmentations alone were not sufficient to bridge it.
- Fine-tuning improved Cityscapes val IoU substantially (0.5785 → 0.8268), confirming the binary head is effective when the domain matches.
- Ghana augmentations gave a marginal improvement on Cityscapes val (0.8268 → 0.8307) but did not help on the actual Ghanaian test set.
- The auto-annotated ground truth masks (from `auto_annotate.py`) are approximate, which may affect the absolute metric values.

## Scripts
| File | Description |
|---|---|
| `Sprint1.ipynb` | Main notebook with all experiments |
| `augment_ghana.py` | Ghana-specific augmentation pipeline |
| `auto_annotate.py` | Automated road annotation using HSV thresholding + spatial priors |
