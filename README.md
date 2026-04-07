# Chrono-Aware Embryo Phase Classification (16-Phase Ablation Study)

## Overview
This project studies **chronologically-aware classification** for human embryo development.

- **Task:** classify time-lapse embryo images into **16 ordered phases**: `tPB2 → … → tHB`.
- **Problem:** standard classifiers treat classes as independent ("chronologically blind").
- **Idea:** add an **ordinal/temporal notion of distance** via a hybrid loss that combines:
  - **Cross-Entropy (CE)** for exact phase classification
  - **Expected-value regression penalty (MSE)** for chronological proximity

We run an ablation "tournament" across five CNN backbones:
**MobileNet_v2, GoogLeNet, Inception_v3, VGG16, VGG19**, each trained twice:
- **Baseline:** `α = 1.0` (pure CE)
- **Hybrid:** `α = 0.5` (50% CE + 50% expected-index MSE)

## Custom Loss Function (Math)
Let $K=16$ be the number of phases, indexed $i \in \{0,\dots,15\}$.
For a sample with true class index $y$ and model logits $\mathbf{z} \in \mathbb{R}^K$:

1) **Cross-Entropy (baseline)**

$$
\mathcal{L}_{\text{CE}} = -\log p_y, \quad \text{where } \mathbf{p} = \text{softmax}(\mathbf{z})
$$

This optimizes exact classification but does not encode that confusing adjacent phases is less severe than confusing distant phases.

2) **Expected phase index** (continuous "timeline position")

$$
\mathbb{E}[\hat{y}] = \sum_{i=0}^{15} p_i \cdot i
$$

3) **Chronological distance penalty (MSE)**

$$
\mathcal{L}_{\text{MSE}} = \big(\mathbb{E}[\hat{y}] - y\big)^2
$$

This creates a **rubber-band** effect: an error of 1 phase yields penalty 1, while an error of 10 phases yields penalty 100.

4) **Final hybrid loss**

$$
\mathcal{L}_{\text{Hybrid}} = \alpha\,\mathcal{L}_{\text{CE}} + (1-\alpha)\,\mathcal{L}_{\text{MSE}}
$$

In this study:
- Baseline: $\alpha = 1.0$
- Hybrid: $\alpha = 0.5$

## Tournament Setup
- **Hardware:** RTX 3070 Ti Laptop GPU (8GB VRAM), trained locally in VS Code.
- **Data:** time-lapse embryo images + phase annotations (16 classes).
- **Input size (this run):** `299×299` for all backbones (the training notebook uses a single resize setting aligned with Inception v3).
- **Split:** 70/15/15 split at the **video level**.
- **Metric:**
  - **Exact Accuracy**
  - **Tolerance Accuracy (±1 phase)** (treats off-by-one as acceptable)

## Results & Insights
### Dashboard
![FINAL Tournament Dashboard](outputs/FINAL_TOURNAMENT_DASHBOARD.png)

### Quantitative Summary (from checkpoints)
The table below is generated from the saved `.pth` files in `outputs/` (see [tools/summarize_tournament.py](tools/summarize_tournament.py)).

| Model | Run | Best Val Exact (%) | Best Val Tol (±1) (%) | Overfit Gap (Train–Val, pp) | Final Train Loss | Epochs |
|---|---|---:|---:|---:|---:|---:|
| MobileNet | Baseline (CE) | 64.65 | 87.21 | 1.60 | 0.91 | 6 |
| MobileNet | Hybrid (50/50 CE+MSE) | 63.40 | 87.25 | 1.72 | 0.86 | 9 |
| GoogLeNet | Baseline (CE) | 60.08 | 85.20 | 3.08 | 1.04 | 10 |
| GoogLeNet | Hybrid (50/50 CE+MSE) | 54.76 | 80.08 | -1.30 | 1.70 | 10 |
| InceptionV3 | Baseline (CE) | **67.00** | **89.42** | 0.45 | 0.89 | 6 |
| InceptionV3 | Hybrid (50/50 CE+MSE) | 63.60 | 87.61 | 0.15 | 1.01 | 8 |
| VGG16 | Baseline (CE) | 28.66 | 47.35 | -2.51 | 2.27 | 10 |
| VGG16 | Hybrid (50/50 CE+MSE) | 26.10 | 46.45 | -1.86 | 7.90 | 10 |
| VGG19 | Baseline (CE) | 49.99 | 73.72 | -1.06 | 1.47 | 7 |
| VGG19 | Hybrid (50/50 CE+MSE) | 49.25 | 72.26 | -1.93 | 2.09 | 7 |

### Key Findings
- **Champion:** **InceptionV3 + Baseline CE** achieved the best validation performance (Tolerance ±1: **89.42%**, Exact: **67.00%**).
- **Regularization effect:** the Hybrid loss strongly suppressed overfitting, often shrinking (or even flipping) the train–val gap.
- **Gradient conflict / “lazy middle” behavior:** the MSE term can dominate optimization pressure when the model assigns probability mass far from the true phase, encouraging safer middle-ground predictions. This typically reduced **exact accuracy** and increased **final training loss** versus pure CE.
- **Model behavior:**
  - **MobileNet** was a consistent and efficient workhorse.
  - **GoogLeNet** degraded noticeably under the Hybrid objective.
  - **VGG16/19** underperformed strongly; VGG16 + Hybrid showed severe loss inflation.

### Per-Run Curves
Each run saves curves to `outputs/*_results.png`:
- `outputs/MobileNet_Baseline_CE_results.png`
- `outputs/MobileNet_Hybrid_MSE_results.png`
- `outputs/GoogLeNet_Baseline_CE_results.png`
- `outputs/GoogLeNet_Hybrid_MSE_results.png`
- `outputs/InceptionV3_Baseline_CE_results.png`
- `outputs/InceptionV3_Hybrid_MSE_results.png`
- `outputs/VGG16_Baseline_CE_results.png`
- `outputs/VGG16_Hybrid_MSE_results.png`
- `outputs/VGG19_Baseline_CE_results.png`
- `outputs/VGG19_Hybrid_MSE_results.png`


## Project Report
See the full academic-style report in [REPORT.md](REPORT.md).
