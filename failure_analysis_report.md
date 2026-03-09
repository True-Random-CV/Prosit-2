# Failure Analysis Report \-- Sprint 3 (Improved)

## 1\. Mining Strategy

- Balanced hard-negative selection across CLIP failure categories  
- Diversity-aware selection using CLIP image embeddings  
- Confidence-masked pseudo-labels with ignored pixels  
- Iterative self-training with staged backbone unfreezing  
- Hard-crop training around uncertainty, disagreement, and boundary regions  
- Detector pseudo-label verification with CLIP crop matching

## 2\. Hard Negative Distribution

| Category | Count | Percentage |
| :---- | :---- | :---- |
| dust\_texture | 30 | 41.7% |
| road\_degradation | 30 | 41.7% |
| vehicle\_confusion | 5 | 6.9% |
| pedestrian\_occlusion | 4 | 5.6% |
| low\_light | 1 | 1.4% |
| market\_clutter | 1 | 1.4% |
| speed\_bumps | 1 | 1.4% |

## 3\. Pseudo-Label Quality

| Round | Stable Frames | Mean Valid Px | Mean Ignored Px | Mean Weight |
| :---- | :---- | :---- | :---- | :---- |
| 1 | 72 | 0.9292 | 0.0708 | 0.9400 |
| 2 | 72 | 0.9432 | 0.0568 | 0.9458 |

## 4\. Segmentation Results

| Model | Cityscapes Val IoU | Ghana Mean Entropy | Ghana Uncertain Px |
| :---- | :---- | :---- | :---- |
| Cityscapes-only | 0.8446 | 0.1112 | 0.0735 |
| Ghana-augmented | 0.8198 | 0.0956 | 0.0507 |
| Hard-neg retrained | 0.8612 | 0.0526 | 0.0196 |

### Per-Category Entropy Improvement

| Category | Before (Augmented) | After (Retrained) | Change |
| :---- | :---- | :---- | :---- |
| speed\_bumps | 0.0931 | 0.0344 | 0.0587 |
| vehicle\_confusion | 0.1044 | 0.0460 | 0.0584 |
| road\_degradation | 0.1092 | 0.0522 | 0.0570 |
| dust\_texture | 0.1031 | 0.0510 | 0.0522 |
| pedestrian\_occlusion | 0.0946 | 0.0425 | 0.0520 |
| market\_clutter | 0.1021 | 0.0618 | 0.0402 |
| low\_light | 0.0648 | 0.0870 | \-0.0222 |

### Round-by-Round Training

| Round | Epoch | Train Loss | Val IoU | Stage | LRs |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | 1 | 0.1579 | 0.8159 | 0 | \[0.0005\] |
| 1 | 2 | 0.1446 | 0.7870 | 0 | \[0.0005\] |
| 1 | 3 | 0.1659 | 0.8510 | 1 | \[0.0005, 0.0001\] |
| 1 | 4 | 0.1465 | 0.8420 | 1 | \[0.0005, 0.0001\] |
| 1 | 5 | 0.1370 | 0.8144 | 1 | \[0.0005, 0.0001\] |
| 1 | 6 | 0.1278 | 0.8304 | 1 | \[0.00025, 5e-05\] |
| 1 | 7 | 0.1525 | 0.8318 | 2 | \[0.0005, 0.0001, 5e-05\] |
| 1 | 8 | 0.1372 | 0.8573 | 2 | \[0.0005, 0.0001, 5e-05\] |
| 1 | 9 | 0.1295 | 0.8096 | 2 | \[0.0005, 0.0001, 5e-05\] |
| 1 | 10 | 0.1211 | 0.8474 | 2 | \[0.0005, 0.0001, 5e-05\] |
| 2 | 1 | 0.1165 | 0.8404 | 0 | \[0.0005\] |
| 2 | 2 | 0.1111 | 0.8354 | 0 | \[0.0005\] |
| 2 | 3 | 0.1151 | 0.8277 | 1 | \[0.0005, 0.0001\] |
| 2 | 4 | 0.1082 | 0.8241 | 1 | \[0.0005, 0.0001\] |
| 2 | 5 | 0.1035 | 0.8485 | 1 | \[0.0005, 0.0001\] |
| 2 | 6 | 0.1001 | 0.8414 | 1 | \[0.0005, 0.0001\] |
| 2 | 7 | 0.1163 | 0.8259 | 2 | \[0.0005, 0.0001, 5e-05\] |
| 2 | 8 | 0.1139 | 0.8300 | 2 | \[0.0005, 0.0001, 5e-05\] |
| 2 | 9 | 0.1033 | 0.8355 | 2 | \[0.0005, 0.0001, 5e-05\] |
| 2 | 10 | 0.1107 | 0.8329 | 2 | \[0.0005, 0.0001, 5e-05\] |

## 5\. Detection Results

| Configuration | Total Dets | Dets/Frame | Mean Score |
| :---- | :---- | :---- | :---- |
| Orig det \+ Cityscapes seg | 3804 | 1.0 | 0.4039 |
| Orig det \+ Augmented seg | 5341 | 1.4 | 0.4166 |
| Orig det \+ Retrained seg | 3842 | 1.0 | 0.4066 |
| Retrained det \+ Retrained seg | 12254 | 3.2 | 0.3518 |

- Verified pseudo-labeled Ghana frames: 64  
- Source detection frames used: 0

| Det Epoch | Loss |
| :---- | :---- |
| 1 | 1.1685 |
| 2 | 0.6811 |
| 3 | 0.5201 |
| 4 | 0.3733 |
| 5 | 0.2680 |
| 6 | 0.2011 |

## 6\. Visualizations

- Comparison images saved to `/content/drive/My Drive/Colab Notebooks/Computer Vision/sprint3_output/visualizations`  
- Pseudo-label previews saved to `/content/drive/My Drive/Colab Notebooks/Computer Vision/sprint3_output/pseudo_labels`

