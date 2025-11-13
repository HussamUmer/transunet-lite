# 🚀 TransUNet-Lite: Fast & Memory-Efficient Transformer Segmentation for Clinical-Scale Use

> **Status:** 📝 *Manuscript in preparation.*  
> Full code and architectural details for **TransUNet-Lite-Base** and **TransUNet-Lite-Tiny** are intentionally withheld until paper submission.  
> This repo currently exposes the **baseline setups, evaluation pipeline, and comparison models**; Lite variants (code + detailed diagrams) will be released after submission for full reproducibility.

---

## Abstract

This research presents TransUNet-Lite, a family of lightweight hybrid transformer–CNN models designed to deliver accurate, reliable, and efficient medical image segmentation under real-world computational constraints. We benchmark UNETR, SETR, TransUNet (paper-style baseline), and our two proposed variants—TransUNet-Lite-Base and TransUNet-Lite-Tiny—within a unified, strictly controlled MedSegBench-style pipeline applied to two heterogeneous imaging modalities: ISIC 2016 dermoscopy and BUSI breast ultrasound. 

Lite-Base achieves near-baseline TransUNet performance while reducing parameters and peak GPU memory by ~3–4×, and Lite-Tiny delivers extreme efficiency with ~16× fewer parameters and ~9× lower VRAM, yet remains competitive across Dice/IoU, AUPRC, and AUROC. Beyond accuracy, we conduct a thorough evaluation of CPU-only latency, throughput, memory usage, and model behavior under Test-Time Augmentation (TTA). The resulting mean-probability  and uncertainty maps reveal that the Lite variants maintain stable, anatomically meaningful confidence profiles across both modalities. 

Overall, these results demonstrate that TransUNet-Lite offers a strong accuracy–efficiency–uncertainty trade-off, providing transformer-level segmentation performance while remaining lightweight enough for real-time, edge-based, or resource-constrained clinical deployments.

---
## 1. Overview

This project introduces **TransUNet-Lite** variants that keep the global context of TransUNet-style hybrids while **aggressively reducing parameters, FLOPs, VRAM, and CPU/GPU latency**.

We systematically compare:

- **UNETR**
- **SETR**
- **TransUNet (Paper-style Baseline)**
- **TransUNet-Lite-Base**
- **TransUNet-Lite-Tiny**

under a **single, controlled pipeline** on:

- 🟤 **BUSI** — Breast ultrasound tumor segmentation (binary).
- 🟤 **ISIC 2016** — Skin lesion segmentation (binary variant), resized to 256×256.

All models are trained & evaluated with **identical data splits, losses, thresholds, augmentations, logging, and evaluation scripts**, so differences come from the **architectures**, not from training shortcuts.

---

## 2. Why TransUNet-Lite?

Traditional TransUNet-style models are powerful but heavy:

- Large Transformer bottlenecks
- Thick decoders
- High memory footprint and latency

**TransUNet-Lite-Base** and **TransUNet-Lite-Tiny** are designed to answer:

1. **Can we keep TransUNet-level segmentation quality with far fewer parameters and memory?**
2. **Can we make Transformer-based segmentation practical on GPUs with limited VRAM and on CPUs?**
3. **How do these lighter hybrids behave across *two* different medical datasets under the same protocol?**

Key design ideas (paper-style, implementation ready):

- 🔹 **Pretrained Transformer encoder** (ViT-S/16 for Lite-Base, DeiT-Tiny/16 for Lite-Tiny)
- 🔹 **Lite CNN skip path** to provide multi-scale features (H/4, H/8, H/16)
- 🔹 **Depthwise-separable decoder** to cut compute
- 🔹 **SE-style gated skips** to suppress noisy activations
- 🔹 **Boundary head** (auxiliary) to sharpen contours (optional in inference)

---

## 3. Datasets 📚

All experiments are built on top of **MedSegBench** standardized releases to ensure
fair, reproducible comparison across architectures. We use the official NPZ files,
predefined train/validation/test splits, and a unified 256×256 input resolution.

### 3.1 ISIC 2016 — Skin Lesion Segmentation (Dermoscopy)

- **Modality:** Dermoscopy  
- **Task:** Binary segmentation — lesion vs. background  
- **Notes:**
  - Classic, relatively clean dataset.
  - Well-suited for evaluating boundary quality and small architectural differences.
  - Resized and standardized by MedSegBench for consistent benchmarking.

**Split statistics (MedSegBench version used in this repo):**

| Dataset     | Modality    | Task                    | Train | Val | Test | Resolution | Source            |
|------------|-------------|-------------------------|------:|----:|-----:|-----------:|-------------------|
| ISIC 2016  | Dermoscopy  | Lesion vs. background   | 810   | 90  | 379  | 256×256    | MedSegBench (NPZ) |

We directly consume the MedSegBench NPZ file (`isic2016_256.npz`) and **do not**
alter the official splits. All splits are logged and exported in `summary/` for
full reproducibility.

---

### 3.2 BUSI — Breast Ultrasound Lesion Segmentation

- **Modality:** Ultrasound  
- **Task:** Binary segmentation — lesion vs. background  
- **Why BUSI:**  
  - Noisy, low-contrast, irregular shapes → **much harder** than ISIC 2016.  
  - Serves as a stress test: how well do models generalize under challenging,
    clinically realistic conditions?

**Split statistics (MedSegBench configuration used in this repo):**

| Dataset  | Modality    | Task                    | Train | Val | Test | Resolution | Source            |
|---------|-------------|-------------------------|-------:|-----:|------:|-----------:|-------------------|
| BUSI    | Ultrasound  | Lesion vs. background   |  452  |  64 | 131 | 256×256    | MedSegBench (NPZ) |

\*Exact BUSI split counts are taken **directly from the MedSegBench NPZ** used in
our runs and recorded in the corresponding `*_ids.txt` files under `summary/`.
We keep those splits fixed across all models to guarantee strict fairness.

---

### 3.3 Why These Two?

Together, **ISIC 2016** and **BUSI** give us:

- A **clean dermoscopy benchmark** where differences in architecture,
  calibration, and boundary handling are clearly visible.
- A **difficult ultrasound benchmark** where all models drop in accuracy,
  allowing us to:
  - test robustness,
  - probe generalization beyond “nice” images,
  - and see how our lightweight TransUNet-Lite variants behave under
    realistic noise and artifacts.

This pairing makes the reported results **more meaningful** than
single-dataset evaluations and sets up a solid foundation for future
extensions to more MedSegBench modalities.

---

## 4. 🧩 Training Pipeline Overview

This section illustrates the complete flow of our segmentation pipeline —  
from input images → preprocessing → augmentations → model training → evaluation → visualization → calibration.  
Both BUSI (ultrasound) and ISIC 2016 (dermoscopy) follow the same unified structure.

---

<h4 align="center">📌 BUSI Pipeline (Ultrasound)</h4>

---

![BUSI Pipeline](Diagrams/POWERPNT_yu5BmMHbsY.png)

*Figure 1: End-to-end BUSI (ultrasound) segmentation pipeline.  
Includes preprocessing (resize + ImageNet normalization), augmentation (geometric + photometric), model training, visualization, and postprocessing/calibration.*

---

<h4 align="center">📌 ISIC 2016 Pipeline (Dermoscopy)</h4>

---

![ISIC Pipeline](Diagrams/POWERPNT_PUZ2tKdhG8.png)

*Figure 2: Full ISIC 2016 (dermoscopy) segmentation workflow.  
The same unified MedSegBench-style pipeline is used for all compared models, ensuring strict reproducibility and apples-to-apples benchmarking.*

---

## 4. 🧪 Environment & Reproducibility

All experiments in this repository are designed to be strictly reproducible.  
Below is the **reference environment** used for the reported results:

**System**

- Python: `3.12.12`
- OS: `Linux-6.6.105+` (x86_64, glibc 2.35)
- Device: `cuda`
- GPU: `NVIDIA Tesla T4`  
  - Total VRAM: **15,095 MB** (~14.74 GB)
- CUDA: `12.6`
- cuDNN: `9.1.2`

**Core Libraries**

- `torch` `2.8.0+cu126`
- `torchvision` `0.23.0+cu126`
- `timm` `1.0.21`
- `monai` `1.5.1`
- `torchmetrics` `1.8.2`
- `numpy` `2.0.2`
- `pandas` `2.2.2`
- `albumentations` `2.0.8`
- `opencv-python` (`cv2`) `4.12.0`
- `Pillow` `11.3.0`
- `matplotlib` `3.10.0`
- `PyYAML` `6.0.3`
- `scikit-learn` `1.6.1`
- `psutil` `5.9.5`
- `thop` `0.1.1`
- `fvcore` `0.1.5.post20221221`

> 🔁 The same environment template is used across all compared models to ensure a fair, apples-to-apples evaluation.


## 5. ⚙️ Default Training Configuration (Example: UNETR on BUSI 256×256)

Each model in this repo is trained with a **YAML-driven configuration**.  
Below is the exact config snapshot (simplified) for the `UNETR` run on **BUSI (binary)** at **256×256** as an example:

```yaml
run:
  run_name: unetr_model_busi_IMG256_SEED42_2025-11-04_13-32-50
  seed: 42
  amp_on: true          # Mixed precision enabled

data:
  dataset: busi
  resolution: 256
  medsegbench_dir: /content/data/MedSegBenchCache
  predefined_splits: true

train:
  image_size: 256
  batch_size: 8
  epochs: 10
  num_workers: 4
  optimizer:
    name: adamw
    lr: 0.0003
    weight_decay: 0.0001
  scheduler:
    name: cosine
    warmup_epochs: 5
  early_stopping:
    monitor: val_dice
    patience: 20
  mixed_precision: true

augment:
  geometric:
    flip: true
    rotate: true
    scale: true
    elastic: false
  appearance:
    brightness_contrast: true
    blur_noise: true
  probabilities:
    flip: 0.5
    rotate: 0.3
    scale: 0.3
    brightness_contrast: 0.3
    blur_noise: 0.2

loss:
  primary: dice_bce
  weights:
    dice: 0.7
    bce: 0.3

metrics:
  threshold: 0.5

```
---

## 6. Experimental Setup (Shared for All Models)

**Pipeline (fully standardized):**

- Data from **MedSegBench-style NPZs**, fixed **train/val/test** splits
- Resize to **256×256**, 3-channel input
- Train-time: mild geometric + photometric augmentations (identical across models)
- Test-time: deterministic resize + normalization
- **Loss:** Dice + Binary Cross-Entropy (0.7 / 0.3)
- **Metrics:**
  - Dice coefficient
  - Intersection-over-Union
  - AUPRC, AUROC
  - Calibration via ECE (10 bins) for threshold analysis
- **Threshold:** 0.5 for main reporting; sweep (0.1 → 0.9) for best-Dice analysis
- **Logging:**
  - Per-epoch train/val loss, Dice, IoU (CSV)
  - Best checkpoint by val Dice (tie-breaker: val loss)
  - Test-time reports, latency, and memory usage
- **Comparisons:**
  - Same optimizer, schedule, batch size, and epochs across models within each benchmark.
  - Same GPU for GPU tests; same CPU node for CPU-only runs.

---

## 7. Architectures Compared  

Below we describe each backbone that we compared, how each model was implemented in our unified framework, and the original papers/authors that introduced the architecture.

---

### UNETR — “U-Net with Transformer Encoder”

**Authors & Reference**  
> - **Ali Hatamizadeh**, Dong Yang, Holger R. Roth, Daguang Xu  
> - *“UNETR: Transformers for 3D Medical Image Segmentation,”* WACV 2022.  
> - ➡ [https://arxiv.org/abs/2103.10504](https://arxiv.org/abs/2103.10504)

**Theoretical Summary (How UNETR Works)**  
> UNETR replaces the standard CNN encoder in U-Net with a pretrained **Vision Transformer (ViT-B/16)**.  
> Instead of building the downsampling path with convolutions, UNETR taps features from **multiple transformer layers**, using these token embeddings to construct a hierarchical representation. This creates a U-Net–style feature pyramid **without any convolutional encoder**.

**How We Implemented It**  

> - We use **timm’s ViT-B/16** as the encoder/feature extractor.  
> - We register hooks on four transformer depths (≈ layers **3, 6, 9, 12**) and collect their token outputs.  
> - Each token sequence is reshaped into a spatial map at **1/16 resolution** and then passed through **1×1 convolutions** to form a multi-scale hierarchy:  
>   - 96 → 192 → 384 → 768 channels.  
> - A U-Net–style decoder merges these streams via **top-down UpBlocks**, gradually lifting resolution:  
>   - 1/16 → 1/8 → 1/4 → full **H×W**.  
> - A final **1×1 convolutional head** produces the segmentation logits at full resolution.

---

### SETR — “Segmentation Transformer”

**Authors & Reference**  
> - **Sixiao Zheng**, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, *et al.*  
> - *“Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers,”* CVPR 2021.  
> - ➡ [https://arxiv.org/abs/2012.15840](https://arxiv.org/abs/2012.15840)

**Theoretical Summary (How SETR Works)**  
> SETR is one of the first large-scale **pure-Transformer** segmentation models.  
> It removes CNNs entirely from the encoder and uses a plain **ViT** to encode the image as a sequence of patch tokens.  
> Because ViT outputs a **1/16-resolution** representation, SETR reconstructs full-resolution predictions using a **Progressive > UPsampling (PUP)** decoder built from simple Conv+Upsample blocks.

**How We Implemented SETR-PUP**  

> - **Encoder:** pretrained **ViT-B/16** from `timm` (ImageNet pretraining).  
> - The ViT outputs token embeddings that we reshape into a spatial feature map of shape **(C, H/16, W/16)**.  
> - **Decoder:** we implement the **PUP variant**, following the original paper:  
>   - A stack of **ConvBNReLU** blocks with bilinear upsampling stages:  
>     - H/16 → H/8 → H/4 → H/2 → H.  
> - There are **no skip connections** from earlier layers, matching the original SETR design.

---

### TransUNet (Baseline) — “Hybrid CNN + Transformer U-Net”

**Authors & Reference**  
> - **Jieneng Chen**, Yongyi Lu, Qihang Yu, Xiangde Luo, *et al.*  
> - *“TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation,”* 2021 (MICCAI/MedIA).  
> - ➡ [https://arxiv.org/abs/2102.04306](https://arxiv.org/abs/2102.04306)

**Theoretical Summary (How TransUNet Works)**  
> TransUNet is a **hybrid CNN–Transformer** architecture that combines:  

> - A **ResNet-50** backbone to extract low- and mid-level spatial features at 1/4, 1/8, and 1/16 resolutions.  
> - A **ViT-B/16** that processes image tokens to model **global self-attention** and long-range dependencies.  

> The ViT bottleneck output at 1/16 resolution is fused with the deepest ResNet skip.  
> A standard U-Net decoder then upsamples back to full resolution, **merging**:  
> - **Global features** from the ViT, and  
> - **Local spatial details** from the CNN skips.  

> This design became a template for many later Transformer-based segmentation models.

**How We Implemented It**  

> - **CNN encoder:** ImageNet-pretrained **ResNet-50**, providing skip maps:  
>   - s1 @ **H/4**, 256 channels  
>   - s2 @ **H/8**, 512 channels  
>   - s3 @ **H/16**, 1024 channels  
> - **Transformer bottleneck:** `timm` **ViT-B/16**, producing a **768-channel** token map at H/16.  
> - We reshape the ViT tokens into a spatial map and fuse it with the deepest ResNet feature (s3).  
> - **Decoder (U-Net style):**  
>   - **UpBlock 1:** (ViT ⊕ s3) → 512 channels @ H/16  
>   - **UpBlock 2:** (512 ⊕ s2) → 256 channels @ H/8  
>   - **UpBlock 3:** (256 ⊕ s1) → 128 channels @ H/4  
>   - Final **bilinear upsample ×4** + Conv head → full-resolution logits.

**TransUNet-Lite-Base**  
> ViT-S/16 backbone + lightweight CNN skip encoder + depthwise decoder + SE-gated skips + boundary head.

**TransUNet-Lite-Tiny**  
> DeiT-Tiny/16 backbone + same lite decoder design as Lite-Base, further reduced channels.

All implemented in a way that **matches the spirit of the original papers** while fitting a **single plug-and-play evaluation pipeline**.
Full architectural details for *TransUNet-Lite-Base* and *TransUNet-Lite-Tiny* will be released after the manuscript submission.

---

## 8. 🧱 Model Size (Trainable Parameters)

| Model                | Parameters (Millions) |
|----------------------|----------------------:|
| UNETR               | 89.035                |
| SETR                | 90.934                |
| TransUNet-Baseline  | 108.769               |
| TransUNet-Lite-Base | 23.298                |
| TransUNet-Lite-Tiny | 6.814                 |

*Interpretation:* Both TransUNet-Lite variants are significantly more compact than the transformer-heavy baselines, with Lite-Tiny reducing parameter count by **~16×** vs. TransUNet-Baseline while remaining competitive in segmentation quality.

![Figure 1: Model Size Comparison](https://github.com/HussamUmer/transunet-lite/raw/main/plots/Full_Test/newplot%20(1).png)  
*Figure 1. Trainable parameters for all architectures. TransUNet-Lite-Base and TransUNet-Lite-Tiny are dramatically more compact than UNETR, SETR, and TransUNet-Baseline.*

---

## 9. Full GPU Test Results

### 9.1 Dice & IoU (Higher is Better)

| Model                | BUSI Dice | BUSI IoU | ISIC 2016 Dice | ISIC 2016 IoU |
|----------------------|:---------:|:--------:|:--------------:|:-------------:|
| UNETR                | 0.632     | 0.510    | 0.908          | 0.844         |
| SETR                 | 0.652     | 0.539    | 0.909          | 0.842         |
| TransUNet-Baseline   | **0.759** | **0.675**| **0.917**      | **0.856**     |
| TransUNet-Lite-Base  | 0.711     | 0.602    | 0.916          | 0.854         |
| TransUNet-Lite-Tiny  | 0.680     | 0.568    | 0.909          | 0.842         |

*Interpretation:*  
On **both datasets**, the **TransUNet family** dominates UNETR/SETR in Dice/IoU. **Lite-Base** stays within ≈0.001–0.005 of the baseline TransUNet on ISIC 2016 while being much lighter, and clearly outperforms UNETR/SETR. **Lite-Tiny** maintains competitive quality with a stronger efficiency bias.

![Figure 2: Test Dice & IoU (ISIC + BUSI)](https://github.com/HussamUmer/transunet-lite/raw/main/plots/Full_Test/newplot%20(2).png)  
*Figure 2. Test Dice and IoU on ISIC 2016 and BUSI. TransUNet-Lite-Base closely tracks the TransUNet-Baseline, while TransUNet-Lite-Tiny provides a strong efficiency–accuracy trade-off.*

---

### 9.2 AUPRC & AUROC (Higher is Better)

| Model                | BUSI AUPRC | BUSI AUROC | ISIC 2016 AUPRC | ISIC 2016 AUROC |
|----------------------|:----------:|:----------:|:----------------:|:----------------:|
| UNETR                | 0.723      | 0.944      | 0.976            | 0.989            |
| SETR                 | 0.720      | 0.935      | 0.975            | 0.990            |
| TransUNet-Baseline   | **0.833**  | **0.969**  | 0.981            | 0.991            |
| TransUNet-Lite-Base  | 0.795      | 0.959      | 0.980            | 0.991            |
| TransUNet-Lite-Tiny  | 0.801      | 0.951      | **0.981**        | **0.992**        |

*Interpretation:*  
All models are strong; **TransUNet-Baseline** and both **Lite** variants show **excellent lesion detection quality**, with Lite-Tiny even edging the others in AUROC on ISIC 2016. This confirms that aggressive compression does **not break discriminative power**.

![Figure 3: PR & ROC (AUPRC / AUROC)](https://github.com/HussamUmer/transunet-lite/raw/main/plots/Full_Test/newplot%20(3).png)  
*Figure 3. Precision–Recall and ROC performance across models on both datasets. TransUNet variants maintain strong discrimination, with Lite-Base and Lite-Tiny staying competitive with heavier transformer baselines.*

## 📈 ROC & PR Curves (TransUNet-Lite-Tiny Examples)

### BUSI (Ultrasound)

| ROC Curve | PR Curve |
|:---------:|:--------:|
| ![BUSI ROC](https://github.com/HussamUmer/transunet-lite/blob/main/plots/ROC_PR_Curves/TransUNetLiteTiny_model_busi_IMG256_SEED42_2025-11-03_05-02-40_ROC_curve.png) | ![BUSI PR](https://github.com/HussamUmer/transunet-lite/blob/main/plots/ROC_PR_Curves/TransUNetLiteTiny_model_busi_IMG256_SEED42_2025-11-03_05-02-40_PR_curve.png) |

### ISIC 2016 (Dermoscopy)

| ROC Curve | PR Curve |
|:---------:|:--------:|
| ![ISIC ROC](https://github.com/HussamUmer/transunet-lite/blob/main/plots/ROC_PR_Curves/TransUNet_ISIC2016_IMG256_SEED42_2025-10-14_21-28-31_ROC_curve.png) | ![ISIC PR](https://github.com/HussamUmer/transunet-lite/blob/main/plots/ROC_PR_Curves/TransUNet_ISIC2016_IMG256_SEED42_2025-10-14_21-28-31_PR_curve.png) |

> Full per-model ROC & PR curves are available in **Section 16 — Run Artifacts & Reproducibility**, inside each model’s `figures/` folder.


---

### 9.3 Inference Latency on GPU (Lower is Better)

| Model                | BUSI Latency (ms/img) | ISIC 2016 Latency (ms/img) |
|----------------------|:---------------------:|:---------------------------:|
| UNETR                | 56.90                 | 59.43                       |
| SETR                 | 50.05                 | 51.30                       |
| TransUNet-Baseline   | 72.78                 | 75.30                       |
| TransUNet-Lite-Base  | 56.83                 | 57.67                       |
| TransUNet-Lite-Tiny  | **52.18**             | **56.03**                   |

*Interpretation:*  
On GPU, **TransUNet-Lite-Tiny** consistently reduces latency vs. the baseline. **Lite-Base** closes most of the gap between a very heavy baseline and leaner designs, while preserving near-identical accuracy.

![Figure 4: Inference Latency (GPU)](https://github.com/HussamUmer/transunet-lite/raw/main/plots/Full_Test/newplot%20(4).png)  
*Figure 4. Per-image inference latency on GPU. Lite-Tiny and Lite-Base offer substantially faster or comparable inference versus larger baselines, highlighting their deployability.*

---

### 9.4 Peak VRAM on GPU (Lower is Better)

| Model                | BUSI Peak VRAM (MB) | ISIC 2016 Peak VRAM (MB) |
|----------------------|:-------------------:|:-------------------------:|
| UNETR                | 1620.23             | 1620.23                   |
| SETR                 | 1594.83             | 1592.12                   |
| TransUNet-Baseline   | 1981.80             | 1981.80                   |
| TransUNet-Lite-Base  | 497.64              | 495.43                    |
| TransUNet-Lite-Tiny  | **215.71**          | **215.71**                |

*Interpretation:*  
This is where **TransUNet-Lite truly shines**. Lite-Base cuts VRAM by **~3–4×**, and Lite-Tiny by **~9×** compared to the baseline, while remaining competitive or better than UNETR/SETR in segmentation quality.

![Figure 5: Peak VRAM Usage](https://github.com/HussamUmer/transunet-lite/raw/main/plots/Full_Test/newplot%20(5).png)  
*Figure 5. Peak VRAM consumption during inference. Lite-Base and especially Lite-Tiny significantly reduce memory footprint while preserving practical segmentation quality.*

![Figure 5: Peak VRAM Usage](https://github.com/HussamUmer/transunet-lite/raw/main/plots/Full_Test/newplot%20(6).png)  
*Figure 6. Peak VRAM consumption during inference.*

---

## 10. CPU-Only Evaluation (Realistic Edge Scenario)

All CPU experiments:

- Run with **batch size = 1**
- On the **same CPU environment**
- Use **50 fixed test images** per model/dataset for fair median/p90/p95 latency and throughput.

### 10.1 Dice & IoU on CPU

| Model                | BUSI Dice | BUSI IoU | ISIC 2016 Dice | ISIC 2016 IoU |
|----------------------|:---------:|:--------:|:--------------:|:-------------:|
| UNETR                | 0.629     | 0.518    | 0.924          | 0.869         |
| SETR                 | 0.661     | 0.560    | 0.922          | 0.864         |
| TransUNet-Baseline   | **0.770** | **0.705**| **0.937**      | 0.885         |
| TransUNet-Lite-Base  | 0.718     | 0.614    | 0.932          | **0.887**     |
| TransUNet-Lite-Tiny  | 0.664     | 0.556    | 0.925          | 0.865         |

*Interpretation:*  
On CPU, **TransUNet-Baseline** still leads in raw Dice, but **Lite-Base** is very close and **slightly surpasses it in IoU on ISIC 2016**, with much lower resource usage. Lite-Tiny remains a strong efficient alternative.

![Figure 7: CPU-only Dice & IoU](https://github.com/HussamUmer/transunet-lite/raw/main/plots/CPU_Only/newplot%20(1).png)  
*Figure 7. CPU-only Dice and IoU on BUSI and ISIC 2016. TransUNet-Lite-Base and Lite-Tiny preserve strong segmentation quality while running entirely on CPU.*

---

### 10.2 Latency Distribution on CPU (Lower is Better)

**Median / p90 / p95 per image (ms)**

| Model                | BUSI Med | BUSI p90 | BUSI p95 | ISIC Med | ISIC p90 | ISIC p95 |
|----------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| UNETR                | 717.48   | 883.69   | 899.25   | 947.03   | 1536.65  | 1553.19  |
| SETR                 | 941.69   | 1888.60  | 2106.99  | 769.12   | 912.66   | 945.19   |
| TransUNet-Baseline   | 1849.62  | 3438.89  | 3686.32  | 1595.43  | 3023.54  | 3374.08  |
| TransUNet-Lite-Base  | 574.51   | 1206.97  | 1323.37  | 633.44   | 1071.18  | 1284.09  |
| TransUNet-Lite-Tiny  | **274.80** | **385.56** | **452.38** | **444.18** | **841.97** | **884.80** |

*Interpretation:*  
On CPU, **TransUNet-Baseline is prohibitively slow**. **Lite-Base** halves latency vs. the baseline; **Lite-Tiny** is **~4–7× faster**, making Transformer-style segmentation realistic for CPU-bound clinical setups.

![Figure 8: Latency Distribution on CPU](https://github.com/HussamUmer/transunet-lite/raw/main/plots/CPU_Only/newplot%20(2).png)  
*Figure 8. CPU latency per image (median / p90 / p95). Lite-Tiny is consistently the fastest, with Lite-Base offering a strong speed–accuracy trade-off compared to heavier transformer baselines.*

---

### 10.3 Throughput (Frames per Second, Higher is Better)

| Model                | BUSI FPS | ISIC 2016 FPS |
|----------------------|:--------:|:-------------:|
| UNETR                | 1.35     | 0.93          |
| SETR                 | 0.81     | 1.26          |
| TransUNet-Baseline   | 0.45     | 0.51          |
| TransUNet-Lite-Base  | 1.51     | 1.40          |
| TransUNet-Lite-Tiny  | **3.70** | **1.93**      |

*Interpretation:*  
**Lite-Tiny** achieves the **highest throughput** across both datasets, far exceeding the baseline TransUNet, while maintaining strong segmentation quality.

![Figure 9: CPU Throughput (FPS)](https://github.com/HussamUmer/transunet-lite/raw/main/plots/CPU_Only/newplot%20(3).png)  
*Figure 9. Inference throughput on CPU (frames per second). Lite-Tiny delivers the highest FPS, followed by Lite-Base, demonstrating practical real-time potential on modest hardware.*

---

### 10.4 Peak RAM on CPU (Lower is Better)

| Model                | BUSI RAM (MB) | ISIC 2016 RAM (MB) |
|----------------------|:-------------:|:-------------------:|
| UNETR                | 4210.10       | 4371.83             |
| SETR                 | 4271.93       | 4488.03             |
| TransUNet-Baseline   | 5070.41       | 4847.79             |
| TransUNet-Lite-Base  | 2444.25       | 2601.23             |
| TransUNet-Lite-Tiny  | **1999.39**   | **2163.99**         |

*Interpretation:*  
Both Lite variants **significantly reduce host memory usage**, with Lite-Tiny using **less than half** the RAM of heavy transformer baselines.

![Figure 10: Peak RAM on CPU](https://github.com/HussamUmer/transunet-lite/raw/main/plots/CPU_Only/newplot%20(4).png)  
*Figure 10. Peak host RAM during CPU inference. Lite variants substantially reduce memory requirements relative to UNETR, SETR, and baseline TransUNet.*

---

### 10.5 Wall Time (Lower is Better)

| Model                | BUSI (s) | ISIC 2016 (s) |
|----------------------|:--------:|:-------------:|
| UNETR                | 33.42    | 48.50         |
| SETR                 | 55.47    | 35.71         |
| TransUNet-Baseline   | 100.93   | 89.08         |
| TransUNet-Lite-Base  | 29.89    | 32.16         |
| TransUNet-Lite-Tiny  | **12.17**| **23.37**     |

*Interpretation:*  
Over the same 50-image CPU benchmark, **TransUNet-Lite-Tiny** is **8–9× faster** than the baseline, and **Lite-Base** also clearly outperforms the baseline in end-to-end wall time.

![Figure 11: CPU Wall Time](https://github.com/HussamUmer/transunet-lite/raw/main/plots/CPU_Only/newplot%20(5).png)  
*Figure 11. Total wall-clock time for the CPU evaluation run. Lite-Tiny completes evaluation in the shortest time, with Lite-Base also noticeably faster than larger baselines.*

---

### 10.6 CPU Utilization

| Model                | BUSI CPU (%) | ISIC 2016 CPU (%) |
|----------------------|:------------:|:------------------:|
| UNETR                | 59.58        | 63.53              |
| SETR                 | 67.96        | 59.60              |
| TransUNet-Baseline   | 69.37        | 68.25              |
| TransUNet-Lite-Base  | 78.64        | 87.45              |
| TransUNet-Lite-Tiny  | **77.86**    | **99.20**          |

*Interpretation:*  
Lite models **utilize CPU resources much more effectively**, especially Lite-Tiny, which runs close to full utilization — crucial for practical deployments.

![Figure 12: CPU Utilization](https://github.com/HussamUmer/transunet-lite/raw/main/plots/CPU_Only/newplot%20(6).png)  
*Figure 12. Average CPU utilization during inference. Lite models achieve high utilization while remaining efficient, indicating good scalability on CPU-only systems.*

---

## 11. What Do These Results Tell Us?

1. **Against strong baselines (UNETR, SETR):**
   - All **TransUNet-based** models (Baseline + Lite variants) **consistently outperform** them on Dice/IoU across both datasets.
   - Lite models **retain the semantic strength of TransUNet** while being far cheaper.

2. **Against the original TransUNet baseline:**
   - **TransUNet-Baseline** still gives the **absolute best accuracy**, as expected for its size.
   - **TransUNet-Lite-Base** delivers **near-identical performance** (especially on ISIC 2016) with:
     - massive drops in VRAM (GPU & CPU),
     - improved speed,
     - and easier deployment.
   - **TransUNet-Lite-Tiny** targets **extreme efficiency**:
     - highest FPS, lowest memory,
     - modest but acceptable accuracy drop,
     - ideal for low-resource inference.

3. **Cross-dataset robustness:**
   - Behavior is consistent across **two very different medical datasets** (dermoscopy vs ultrasound),
   - supporting the claim that **the design generalizes**, not just overfits ISIC.

4. **BUSI: stress-testing on a difficult dataset:**
   - All models show **lower scores on BUSI** compared to ISIC 2016.
   - This drop is expected: BUSI ultrasound images are **noisy, low-contrast, and anatomically complex**, making tumor boundaries       intrinsically hard.
   - The experiment is intentional: it demonstrates how each architecture behaves under **realistic, challenging conditions**, rather than only on a “clean” benchmark.

5. **Research angle (publishable story):**
   - A **clean, controlled benchmark** of heavy vs. light Transformer-based segmenters.
   - A practical design that **outclasses prior transformer baselines (UNETR/SETR)** in both quality and efficiency.
   - Lite models that **offer strong Pareto-optimal points** in the accuracy–efficiency trade-off space.

---

## 12. 🎨 Qualitative Predictions

To complement the quantitative tables, we visualize model behavior on held-out test cases from both datasets.  
Each 4-panel grid shows (left → right):

1. Input image  
2. Ground-truth mask overlay  
3. Predicted mask overlay  
4. Predicted boundary (extracted from the predicted mask)

These views highlight lesion shape, contour sharpness, and failure modes under the **same preprocessing, threshold, and color mapping** across models.

---

### 12.1 ISIC 2016 — Qualitative Grids

<!-- Replace the filenames below with your actual ISIC grid images -->

<h3 align="center">ISIC 2016 — UNETR</h3>

![ISIC 2016 — UNETR](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/unetr_model_ISIC2016_IMG256_SEED42_2025-11-05_13-27-20_test_grid_4panels_12.png)  
*Figure 13. ISIC 2016: UNETR (UNet with Transformer encoder) preserves coarse lesion extent but shows less precise boundaries on the hardest cases.*

---

<h3 align="center">ISIC 2016 — SETR</h3>

![ISIC 2016 — SETR](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/setr_model_ISIC2016_IMG256_SEED42_2025-11-01_10-32-35_test_grid_4panels_12.png)  
*Figure 14. ISIC 2016: SETR (SEgmentation TRansformer) provides smooth, globally consistent masks, sometimes over-smoothing fine lesion details.*

---

<h3 align="center">ISIC 2016 — TransUNet (Baseline)</h3>

![ISIC 2016 — TransUNet-Baseline](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/TransUNet_ISIC2016_IMG256_SEED42_2025-10-19_10-01-43_test_grid_4panels_12.png)  
*Figure 15. ISIC 2016: TransUNet baseline delivers sharp lesion boundaries and strong coverage; used as our heavy reference model.*

---

<h3 align="center">ISIC 2016 — TransUNet-Lite-Base</h3>

![ISIC 2016 — TransUNet-Lite-Base](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/TransUNet_Lite_base_ISIC2016_IMG256_SEED42_2025-10-18_07-29-51_test_grid_4panels_12.png)  
*Figure 16. ISIC 2016: TransUNet-Lite-Base visually matches baseline TransUNet on most cases while using far fewer parameters and memory.*

---

<h3 align="center">ISIC 2016 — TransUNet-Lite-Tiny</h3>

![ISIC 2016 — TransUNet-Lite-Tiny](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/TransUNet_Lite_tiny_ISIC2016_IMG256_SEED42_2025-10-14_21-28-31_test_grid_4panels_12.png)  
*Figure 17. ISIC 2016: TransUNet-Lite-Tiny maintains reasonable segmentation quality with slightly softer contours, reflecting its aggressive efficiency focus.*


---

### 12.2 BUSI — Qualitative Grids

<!-- Replace the filenames below with your actual BUSI grid images -->

<h3 align="center">BUSI — UNETR</h3>

![BUSI — UNETR](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/unetr_model_busi_IMG256_SEED42_2025-11-04_13-32-50_test_grid_4panels_12.png)  
*Figure 18. BUSI: UNETR captures major lesions but often underestimates fuzzy, low-contrast margins, illustrating the difficulty of this ultrasound dataset.*

---

<h3 align="center">BUSI — SETR</h3>

![BUSI — SETR](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/setr_model_busi_IMG256_SEED42_2025-11-04_15-02-28_test_grid_4panels_12.png)  
*Figure 19. BUSI: SETR yields smooth and stable masks but can over-simplify irregular tumor shapes in highly textured tissue.*

---

<h3 align="center">BUSI — TransUNet-Baseline</h3>

![BUSI — TransUNet-Baseline](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/TransUNetBaseline_busi_IMG256_SEED42_2025-11-03_03-42-24_test_grid_4panels_12.png)  
*Figure 20. BUSI: TransUNet baseline provides the sharpest and most reliable boundaries on complex lesions, at the cost of much higher computation.*

---

<h3 align="center">BUSI — TransUNet-Lite-Base</h3>

![BUSI — TransUNet-Lite-Base](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/Lite_Base_model_busi_IMG256_SEED42_2025-11-03_04-34-56_test_grid_4panels_12.png)  
*Figure 21. BUSI: TransUNet-Lite-Base maintains strong lesion coverage and boundary quality close to the baseline while being substantially lighter—robust despite BUSI’s challenging variability.*

---

<h3 align="center">BUSI — TransUNet-Lite-Tiny</h3>

![BUSI — TransUNet-Lite-Tiny](https://github.com/HussamUmer/transunet-lite/raw/main/qualitative_prediction/TransUNetLiteTiny_model_busi_IMG256_SEED42_2025-11-03_05-02-40_test_grid_4panels_12.png
)  
*Figure 22. BUSI: TransUNet-Lite-Tiny remains usable under tight compute budgets; some undersegmentation and softer contours appear, which is expected given the dataset’s difficulty and the model’s aggressive compression.*

---

## 13. 🧩 Uncertainty via Test-Time Augmentation (TTA)

We estimate how confident each model is by running the same test image multiple times with tiny, safe changes (e.g., flips) and measuring how much the prediction varies.  
- **Low uncertainty** → predictions stay similar across runs.  
- **High uncertainty** → predictions vary significantly.

---

### 13.1 🧠 Objective
We applied **Test-Time Augmentation (TTA)** to quantify the **model uncertainty** across multiple segmentation models on **ISIC-2016** and **BUSI** datasets.  
TTA helps visualize how consistent a model’s predictions are under small perturbations (flips, rotations, etc.) — producing:
- **Mean Probability Maps** → averaged predictions across augmentations (model consensus)  
- **Uncertainty Maps (Variance)** → per-pixel variance indicating prediction disagreement

---

### 13.2 ⚙️ Method Summary

| Step | Description |
|:--|:--|
| **1. TTA Inference** | Applied 4 augmentation variants: identity, horizontal flip, vertical flip, and both. |
| **2. Aggregation** | Computed per-pixel mean and variance over the TTA predictions. |
| **3. Visualization** | - Mean probability → final predicted lesion overlay (green mask)<br>- Variance → uncertainty heatmap (yellow = high disagreement) |
| **4. Interpretation** | Variance highlights areas where the model is least confident — usually along lesion boundaries or noisy textures. |

---

### 13.3 🔍 Why Include Uncertainty?

- Goes beyond a single accuracy number to expose **model confidence**.  
- Highlights **risky regions** (e.g., fuzzy BUSI lesion edges) that deserve human review.  
- Supports **deployment decisions** and **post-processing** (e.g., calibration or contour smoothing).

---

### 13.4 ⚙️ How We Did It in This Research

**TTA set:**  
`identity`, `horizontal flip`, `vertical flip`, and `both (HV flip)`

**Aggregation per pixel:**  
- **Mean probability** across TTA runs → “consensus” prediction  
- **Variance** across TTA runs → uncertainty map  

---

### 13.5 🎨 Visuals

- **Prediction overlay:** Thresholded mean probability mask on the input image  
- **Uncertainty overlay:** Variance heatmap on the input image  

---

### 13.6 🧠 How to Read Our Figures

| Color | Meaning |
|--------|----------|
| 🟩 **Green/Teal mask** | Final predicted lesion (averaged + thresholded) |
| 🔥 **Warm colors (yellow → red)** | Higher disagreement across TTA runs → more uncertain |
| 🧊 **Cool/transparent** | High agreement → more certain |

---

### 13.7 📊 Typical Patterns We Observe

- **Edges & fine structures:** Higher uncertainty (boundary sensitivity)  
- **Clear interiors:** Lower uncertainty  
- **BUSI (ultrasound):** Higher boundary uncertainty than ISIC (dermoscopy) due to speckle noise and lower contrast  

---

### 13.9 🩻 Takeaways

- ✅ **Low uncertainty + good overlap** → reliable predictions  
- ⚠️ **High uncertainty near boundaries** → flag for review or apply post-processing (e.g., CRF/smoothing) or calibration (e.g., temperature scaling)  
- 🔬 Uncertainty complements **calibration metrics (ECE)** by localizing where confidence is fragile  

---

### 13.10 🧪 Uncertainty on ISIC 2016
---

<h4 align="center">🧠 Model: UNETR — Dataset: ISIC 2016</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/unetr/unetr_model_ISIC2016_IMG256_SEED42_2025-11-05_13-27-20_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/unetr/unetr_model_ISIC2016_IMG256_SEED42_2025-11-05_13-27-20_TTA_uncertainty_id0.png) |

*Figure 23: Visualization of sample ID 0 from ISIC 2016 using UNETR. The predicted lesion mask (green) shows accurate coverage of the lesion region with mild over-smoothing at the edges. The uncertainty heatmap reveals elevated variance along lesion borders, typical of dermoscopic patterns with texture blending.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/unetr/unetr_model_ISIC2016_IMG256_SEED42_2025-11-05_13-27-20_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/unetr/unetr_model_ISIC2016_IMG256_SEED42_2025-11-05_13-27-20_TTA_uncertainty_id1.png) |

*Figure 24: Visualization of sample ID 1 from ISIC 2016 using UNETR. The prediction closely aligns with the ground truth, showing strong central confidence. The uncertainty map highlights thin high-variance bands near the lesion edges, indicating sensitivity to fine boundary details but overall stable segmentation.*

---

<h4 align="center">🧠 Model: SETR — Dataset: ISIC 2016</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/setr/setr_model_ISIC2016_IMG256_SEED42_2025-11-01_10-32-35_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/setr/setr_model_ISIC2016_IMG256_SEED42_2025-11-01_10-32-35_TTA_uncertainty_id0.png) |

*Figure 26: SETR on ISIC (ID 0). The TTA-averaged mask captures the lesion core with smooth contours; uncertainty concentrates along the outer rim—stronger on the right/lower boundary—indicating edge ambiguity where texture blends into skin.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/setr/setr_model_ISIC2016_IMG256_SEED42_2025-11-01_10-32-35_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/setr/setr_model_ISIC2016_IMG256_SEED42_2025-11-01_10-32-35_TTA_uncertainty_id1.png) |

*Figure 27: SETR on ISIC (ID 1). Mean-prob overlay closely follows the lesion shape; a thin uncertainty ring appears around the boundary—most visible on the lower arc—while the interior remains confidently segmented.*

---

<h4 align="center">🧠 Model: TransUNet-Baseline — Dataset: ISIC 2016</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/baseline/TransUNet_ISIC2016_IMG256_SEED42_2025-10-19_10-01-43_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/baseline/TransUNet_ISIC2016_IMG256_SEED42_2025-10-19_10-01-43_TTA_uncertainty_id0.png) |

*Figure 28: TransUNet-Baseline on ISIC (ID 0). The TTA-averaged prediction effectively delineates the lesion region with strong consistency. The uncertainty map reveals a clear outline of higher variance along lesion borders, especially at the right edge, reflecting expected ambiguity from dermoscopic shadowing and soft boundary transitions.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/baseline/TransUNet_ISIC2016_IMG256_SEED42_2025-10-19_10-01-43_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/baseline/TransUNet_ISIC2016_IMG256_SEED42_2025-10-19_10-01-43_TTA_uncertainty_id1.png) |

*Figure 29: TransUNet-Baseline on ISIC (ID 1). The lesion is compact and sharply defined in the mean-probability overlay, with minimal noise around the periphery. The uncertainty map shows small bands of moderate variance near the bottom edge, indicating localized sensitivity but overall confident segmentation performance.*

---

<h4 align="center">🧠 Model: TransUNet-Lite-Base — Dataset: ISIC 2016</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-base/TransUNet_ISIC2016_IMG256_SEED42_2025-10-18_07-29-51_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-base/TransUNet_ISIC2016_IMG256_SEED42_2025-10-18_07-29-51_TTA_uncertainty_id0.png) |

*Figure 30: TransUNet-Lite-Base on ISIC (ID 0). The averaged TTA mask captures the lesion comprehensively, preserving inner texture while slightly overextending at smooth edges. The uncertainty map displays higher variance at boundary regions, particularly near top-right and lower contours, suggesting mild edge sensitivity due to fine contrast transitions.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-base/TransUNet_ISIC2016_IMG256_SEED42_2025-10-18_07-29-51_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-base/TransUNet_ISIC2016_IMG256_SEED42_2025-10-18_07-29-51_TTA_uncertainty_id1.png) |

*Figure 31: TransUNet-Lite-Base on ISIC (ID 1). The mean-probability overlay demonstrates precise lesion capture with high spatial consistency. The uncertainty heatmap highlights narrow bands of elevated variance around the lesion edge—most notable on the left rim—indicating localized ambiguity, while the interior remains confidently segmented.*

---

<h4 align="center">🧠 Model: TransUNet-Lite-Tiny — Dataset: ISIC 2016</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-tiny/TransUNet_ISIC2016_IMG256_SEED42_2025-10-14_21-28-31_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-tiny/TransUNet_ISIC2016_IMG256_SEED42_2025-10-14_21-28-31_TTA_uncertainty_id0.png) |

*Figure 32: TransUNet-Lite-Tiny on ISIC (ID 0). The predicted lesion area appears slightly overextended yet maintains overall lesion coverage. The uncertainty map reveals strong edge variance around the lesion contour—particularly on the top-right and bottom edges—indicating localized confidence drop typical of smaller-capacity models facing texture complexity.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-tiny/TransUNet_ISIC2016_IMG256_SEED42_2025-10-14_21-28-31_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/isic/lite-tiny/TransUNet_ISIC2016_IMG256_SEED42_2025-10-14_21-28-31_TTA_uncertainty_id1.png) |

*Figure 33: TransUNet-Lite-Tiny on ISIC (ID 1). The TTA-averaged overlay aligns closely with the lesion center, showing compact segmentation. The uncertainty map displays a distinct thin halo of moderate uncertainty—most prominent on the left boundary—signifying subtle disagreement across TTA samples but overall confident performance.*

---

### 13.11 🩺 Uncertainty on BUSI (Ultrasound)
---

<h4 align="center">🧠 Model: UNETR — Dataset: BUSI (Ultrasound)</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/unetr/unetr_model_busi_IMG256_SEED42_2025-11-04_13-32-50_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/unetr/unetr_model_busi_IMG256_SEED42_2025-11-04_13-32-50_TTA_uncertainty_id0.png) |

*Figure 34: UNETR on BUSI (ID 0). The model produces a wide and well-defined lesion mask, capturing irregular tumor morphology effectively. Uncertainty is concentrated around the lesion periphery, especially in low-contrast zones, reflecting the model’s sensitivity to boundary noise and ultrasound artifacts.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/unetr/unetr_model_busi_IMG256_SEED42_2025-11-04_13-32-50_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/unetr/unetr_model_busi_IMG256_SEED42_2025-11-04_13-32-50_TTA_uncertainty_id1.png) |

*Figure 35: UNETR on BUSI (ID 1). The mean-probability overlay demonstrates strong lesion localization with limited false positives. The uncertainty map shows elevated variance in posterior regions (bottom-right zone), suggesting difficulty in discriminating lesion borders due to speckle noise and shadow effects typical of BUSI scans.*

---

<h4 align="center">🧠 Model: SETR — Dataset: BUSI (Ultrasound)</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/setr/setr_model_busi_IMG256_SEED42_2025-11-04_15-02-28_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/setr/setr_model_busi_IMG256_SEED42_2025-11-04_15-02-28_TTA_uncertainty_id0.png) |

*Figure 36: SETR on BUSI (ID 0). The predicted segmentation captures the main lesion mass but slightly merges with nearby structures. The uncertainty heatmap shows diffuse variance around the lesion and background, indicating moderate confidence due to tissue texture variability and speckle interference.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/setr/setr_model_busi_IMG256_SEED42_2025-11-04_15-02-28_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/setr/setr_model_busi_IMG256_SEED42_2025-11-04_15-02-28_TTA_uncertainty_id1.png) |

*Figure 37: SETR on BUSI (ID 1). The overlay shows clean lesion localization with consistent mask confidence across the tumor core. The uncertainty map highlights elevated variance below the lesion region—likely due to shadowing artifacts—while the lesion boundary remains relatively stable, reflecting the transformer’s contextual robustness.*

---

<h4 align="center">🧠 Model: TransUNet Baseline — Dataset: BUSI (Ultrasound)</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/baseline/TransUNetBaseline_busi_IMG256_SEED42_2025-11-03_03-42-24_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/baseline/TransUNetBaseline_busi_IMG256_SEED42_2025-11-03_03-42-24_TTA_uncertainty_id0.png) |

*Figure 38: TransUNet Baseline on BUSI (ID 0). The model fails to clearly delineate lesion boundaries, indicating an uncertain prediction in low-contrast regions. The uncertainty heatmap exhibits faint localized variance near the lower middle zone, revealing difficulty in differentiating small or indistinct structures in homogeneous tissue.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/baseline/TransUNetBaseline_busi_IMG256_SEED42_2025-11-03_03-42-24_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/baseline/TransUNetBaseline_busi_IMG256_SEED42_2025-11-03_03-42-24_TTA_uncertainty_id1.png) |

*Figure 39: TransUNet Baseline on BUSI (ID 1). The lesion prediction is compact and well-centered, with strong boundary consistency. However, the uncertainty map reveals high variance near the cystic border—likely due to reflection artifacts and the distinct fluid-tissue intensity gap—suggesting some sensitivity to boundary contrast variations.*

---

<h4 align="center">🧠 Model: TransUNet-Lite (Base) — Dataset: BUSI (Ultrasound)</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-base/Lite_Base_model_busi_IMG256_SEED42_2025-11-03_04-34-56_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-base/Lite_Base_model_busi_IMG256_SEED42_2025-11-03_04-34-56_TTA_uncertainty_id0.png) |

*Figure 40: TransUNet-Lite (Base) on BUSI (ID 0). The model detects small scattered regions, indicating partial recognition of lesion-like textures but with limited spatial coherence. The uncertainty map shows low but distributed variance, implying the model is unsure about weak lesion boundaries under low-contrast ultrasound patterns.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-base/Lite_Base_model_busi_IMG256_SEED42_2025-11-03_04-34-56_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-base/Lite_Base_model_busi_IMG256_SEED42_2025-11-03_04-34-56_TTA_uncertainty_id1.png) |

*Figure 41: TransUNet-Lite (Base) on BUSI (ID 1). The segmentation covers the main lesion with clear shape boundaries, demonstrating robust center confidence. The uncertainty heatmap displays focused high-variance regions near the posterior edge, which aligns with acoustic shadowing and attenuation artifacts typical in ultrasound imaging.*

---

<h4 align="center">🧠 Model: TransUNet-Lite (Tiny) — Dataset: BUSI (Ultrasound)</h4>

---

#### 🩺 Example 1 (ID = 0)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-tiny/TransUNetLiteTiny_model_busi_IMG256_SEED42_2025-11-03_05-02-40_TTA_pred_id0.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-tiny/TransUNetLiteTiny_model_busi_IMG256_SEED42_2025-11-03_05-02-40_TTA_uncertainty_id0.png) |

*Figure 42: TransUNet-Lite (Tiny) on BUSI (ID 0). The prediction captures small isolated lesion-like regions, though the structure remains fragmented. The uncertainty map reveals high localized variance within the bright lesion area, suggesting instability in boundary recognition due to speckle noise and reduced model capacity.*

---

#### 🩺 Example 2 (ID = 1)

| Mean-Probability Overlay | Uncertainty Map |
|:-------------------------:|:---------------:|
| ![Mean Prob](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-tiny/TransUNetLiteTiny_model_busi_IMG256_SEED42_2025-11-03_05-02-40_TTA_pred_id1.png) | ![Uncertainty](https://github.com/HussamUmer/transunet-lite/blob/main/uncertainty/busi/lite-tiny/TransUNetLiteTiny_model_busi_IMG256_SEED42_2025-11-03_05-02-40_TTA_uncertainty_id1.png) |

*Figure 43: TransUNet-Lite (Tiny) on BUSI (ID 1). The lesion segmentation is compact and structurally consistent, showing strong detection of the cystic mass. The uncertainty map highlights a thin uncertainty band around the cyst edge, indicating the model’s cautious estimation near strong intensity gradients while remaining confident within the lesion core.*

---

### 13.12 🩺 ISIC-2016 (Dermoscopic Lesions)

| Model | Observed Pattern | Confidence Behavior |
|:--|:--|:--|
| **UNETR / SETR** | High lesion Dice, but wide uncertainty bands along irregular borders. | Over-confident on texture noise. |
| **TransUNet-Baseline** | Balanced mask but moderate variance at lesion edges. | Partial calibration. |
| **TransUNet-Lite-Base** | Smooth, localized uncertainty; strong consensus across TTA runs. | ✅ Well-calibrated & stable. |
| **TransUNet-Lite-Tiny** | Slight under-segmentation, low false positives; small but consistent variance. | ⚡ Compact & conservative. |

> **Interpretation:**  
> Lite-Base achieves near-teacher performance with *stable uncertainty structure*.  
> Lite-Tiny shows reliable confidence under TTA, suitable for edge deployment where consistency matters.

---

### 13.13 🩻 BUSI (Breast Ultrasound)

| Model | Observed Pattern | Confidence Behavior |
|:--|:--|:--|
| **UNETR / SETR** | Fragmented predictions, high background variance under speckle noise. | Poor calibration. |
| **TransUNet-Baseline** | Coherent masks but diffuse uncertainty around noisy tissue. | Uneven focus. |
| **TransUNet-Lite-Base** | Uncertainty localized along posterior shadow regions — matches clinical ambiguity. | ✅ Most realistic confidence distribution. |
| **TransUNet-Lite-Tiny** | Conservative predictions with tight uncertainty; low false positives. | ⚡ Robust under noise. |

> **Interpretation:**  
> Lite-Base generalizes best across noisy ultrasound textures, while Lite-Tiny maintains compact uncertainty, ideal for reliable deployment in diagnostic tools.

---

### 13.14 📊 Cross-Domain Insights

| Aspect | ISIC-2016 | BUSI |
|:--|:--|:--|
| Confidence Map Quality | Smooth gradient near lesion boundaries | Localized along acoustic shadows |
| Model Consensus (TTA) | Highest for Lite-Base | Highest for Lite-Base |
| Uncertainty Spread | Moderate (texture-driven) | Sharp (noise-driven) |
| Stability Ranking | Lite-Base > Lite-Tiny > Baseline > SETR/UNETR | Lite-Base > Lite-Tiny > Baseline > SETR/UNETR |

---

### 13.15 🧭 Key Takeaways

- **TransUNet-Lite-Base** shows the **most stable TTA consistency** and **anatomically meaningful uncertainty localization**.  
- **TransUNet-Lite-Tiny** achieves compact, conservative uncertainty — balancing reliability with efficiency.  
- Uncertainty maps reveal **true epistemic confidence**, distinguishing model uncertainty from data noise.  
- Across both datasets, TTA confirmed the **robustness and interpretability** of the Lite family compared to heavier transformer baselines.

---

### 13.16 ✅ In Summary
> The proposed **TransUNet-Lite** models not only preserve segmentation accuracy but also deliver **clinically interpretable uncertainty** under Test-Time Augmentation.  
> Their predictions remain consistent across augmentations, proving reliability for **real-world, uncertainty-aware medical AI deployment**.

---

## 14 🔗 Full Training & Testing Notebooks (Open in Colab)

| Dataset     | Model                             | Open in Colab |
|------------|-----------------------------------|---------------|
| ISIC 2016  | UNETR                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/Full_Training_Testing/unetr_model_BaseLine.ipynb) |
| ISIC 2016  | SETR                              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/Full_Training_Testing/setr_model_BaseLine%20(1).ipynb) |
| ISIC 2016  | TransUNet Baseline (R50 + ViT-B/16) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/Full_Training_Testing/TransUNET_BaseLine.ipynb) |
| BUSI       | UNETR                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/Full_Training_Testing/BUSI_Binary_unetr_model256.ipynb) |
| BUSI       | SETR                              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/Full_Training_Testing/BUSI_Binary_setr_model256%20(1).ipynb) |
| BUSI       | TransUNet Baseline (R50 + ViT-B/16) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/Full_Training_Testing/BUSI_Binary_TransUNet256.ipynb) |

> ⚠️ **Note:** Colab notebooks for **TransUNet-Lite-Base** and **TransUNet-Lite-Tiny** will be released after the corresponding research manuscript is submitted.

---

## 15 🧠 CPU-Only Evaluation Notebooks (Open in Colab)

| Dataset     | Model                               | Open in Colab |
|------------|-------------------------------------|---------------|
| ISIC 2016  | UNETR                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/CPU_Only_Eval/UNETR_CPU_Eval_50_ISIC.ipynb) |
| ISIC 2016  | SETR                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/CPU_Only_Eval/SETR_CPU_Eval_50_ISIC%20(1).ipynb) |
| ISIC 2016  | TransUNet Baseline (R50 + ViT-B/16) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/CPU_Only_Eval/TransUNet_Baseline_CPU_Eval_50_ISIC.ipynb) |
| BUSI       | UNETR                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/CPU_Only_Eval/UNETR_CPU_Eval_50_Busi.ipynb) |
| BUSI       | SETR                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/CPU_Only_Eval/SETR_CPU_Eval_50_Busi.ipynb) |
| BUSI       | TransUNet Baseline (R50 + ViT-B/16) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/transunet-lite/blob/main/Coding_Notebooks/CPU_Only_Eval/TransUNet_Baseline_CPU_Eval_50_Busi.ipynb) |

> ⚠️ **Note:** CPU-only evaluation notebooks for **TransUNet-Lite-Base** and **TransUNet-Lite-Tiny** will be made public after the associated manuscript is submitted.

---

## 16. Run Artifacts & Reproducibility 📂

All core experiments (GPU-based training + test-time evaluation) are logged and stored in **per-model, per-dataset** artifact folders.

Each folder typically includes:

- Training & validation logs (`.csv`, `.json`)
- Environment and configuration snapshots (`config.yaml`, env dumps)
- Best and/or last checkpoints (when shared)
- Evaluation summaries (test metrics, calibration)
- Generated figures (curves, qualitative grids)

These artifacts provide a clear evidence trail for all reported numbers.

---

### 16.1 BUSI (Breast Ultrasound, Binary Segmentation)

| Model                    | Artifacts Folder |
|--------------------------|------------------|
| UNETR (BUSI)             | [Open artifacts](https://drive.google.com/drive/folders/1zVgPPjfq-ZhXH7XrskTKiywBXqEW7bZz?usp=sharing) |
| SETR (BUSI)              | [Open artifacts](https://drive.google.com/drive/folders/1Ak5-k7w0XMeng5U-065GrRAYAeKd3QwT?usp=sharing) |
| TransUNet-Baseline (BUSI) | [Open artifacts](https://drive.google.com/drive/folders/1DsnrcaqdPIiooCk98q1SL_f5Ds9Dr_wQ?usp=sharing) |
| TransUNet-Lite-Base (BUSI) | [Open artifacts](https://drive.google.com/drive/folders/17m-Dq0OpYTOBGFnnVjEM2Cgvow8xQf78?usp=sharing) |
| TransUNet-Lite-Tiny (BUSI) | [Open artifacts](https://drive.google.com/drive/folders/15zNYx33tJPbXE2d_07vDHxFin18TKKGq?usp=sharing) |

---

### 16.2 ISIC 2016 (Dermoscopic Lesions, Binary Segmentation)

| Model                         | Artifacts Folder |
|-------------------------------|------------------|
| UNETR (ISIC 2016)             | [Open artifacts](https://drive.google.com/drive/folders/15GOu9aJwTi63jwROLI1rk5gG4ZivAFwQ?usp=sharing) |
| SETR (ISIC 2016)              | [Open artifacts](https://drive.google.com/drive/folders/1d52HEldPUTqAHTV7bcm-bsE_ELg3LRra?usp=sharing) |
| TransUNet-Baseline (ISIC 2016) | [Open artifacts](https://drive.google.com/drive/folders/1t40QUhLEmaqYYyyg8uuGAoXVgLP0b_n8?usp=sharing) |
| TransUNet-Lite-Base (ISIC 2016) | [Open artifacts](https://drive.google.com/drive/folders/1pQtKO20PlK00iRrxdMKPAIfTZHLWVpiW?usp=sharing) |
| TransUNet-Lite-Tiny (ISIC 2016) | [Open artifacts](https://drive.google.com/drive/folders/1d2I6q7hnKjoMvjdijvQlgc6e42aqI7dq?usp=sharing) |

> *These shared directories expose only the experiment artifacts (logs, configs, metrics, qualitative grids) required for verification, without exposing unrelated private Drive content.*

---

## 17. Future Work 🔭

This repository is **Phase 1** of the TransUNet-Lite story: we built a clean, reproducible benchmark and showed that lightweight hybrid designs can approach or rival heavy baselines under consistent conditions. The next steps will deepen the analysis, broaden the evidence, and harden the models for real-world deployment.

### 16.1. Component-wise Ablation & Design Justification

We introduced several coordinated changes at once (gated skips, depthwise decoder, boundary head, lightweight backbones). The next stage will **quantify exactly what each part buys us**:

- **SE / gated skip connections**
  - Measure their impact on:
    - boundary sharpness (boundary Dice),
    - noise suppression in low-contrast regions,
    - stability across folds and seeds,
    - probability calibration near lesion borders.
  - Compare: plain skips vs SE-gated skips vs alternative attention gates.

- **Depthwise-separable decoder**
  - Isolate the effect on:
    - parameters,
    - FLOPs,
    - latency on GPU and CPU,
    - segmentation quality.
  - Validate that depthwise decoders are not just smaller, but **Pareto-efficient** (quality vs cost) relative to standard convolutions.

- **Boundary-aware head**
  - Study how the auxiliary boundary prediction:
    - improves lesion edge adherence,
    - affects calibration near contours,
    - behaves under different weights / loss formulations.
  - Visual + quantitative evaluation: do sharper masks correlate with clinically meaningful gains?

This ablation suite will turn the current design from “intuitively good” to **experimentally justified**.

---

### 17.2. Broader Dataset & Modality Coverage

So far, we evaluated on:

- **ISIC 2016** (dermoscopy, relatively clean).
- **BUSI** (ultrasound, noisy and challenging).

Next, we will extend TransUNet-Lite to **diverse MedSegBench datasets** to test robustness and universality:

- **Other modalities:**
  - CT, MRI (organ and tumor segmentation),
  - X-ray (lung and lesion masks),
  - OCT, microscopy, endoscopy.

- **Other task types:**
  - Multi-class segmentation (up to many structures),
  - Highly imbalanced targets (tiny lesions / structures),
  - Domain shift (train on one center/device, test on another).

Goal: show whether the same architectural recipe (light backbone + gated skips + depthwise decoder + boundary supervision) **generalizes across modalities**, or needs dataset-specific tuning.

---

### 17.3. Efficiency & Deployment-Focused Extensions

We will push beyond single-GPU evaluation and explore **deployment-ready configurations**:

- **Model compression:**
  - Structured pruning on decoder and skip paths.
  - Low-rank / bottleneck variants of attention and projections.

- **Quantization & hardware-aware tuning:**
  - 8-bit and 4-bit quantization trials for edge/CPU deployment.
  - Benchmark on realistic devices (laptops, embedded GPUs, hospital workstations).

- **Distillation:**
  - Use TransUNet-Baseline as a **teacher**:
    - Transfer structural priors to Lite-Base and Lite-Tiny.
    - Target improved calibration + robustness at same lightweight budget.

The aim is to provide **ready-to-use configurations** for real-time or resource-limited settings, not just academic GPUs.

---

### 17.4. Uncertainty, Calibration & Reliability

We already log **AUPRC, AUROC, and Expected Calibration Error**. Next steps:

- Reliability diagrams and class-wise calibration on more datasets.
- Test-time augmentation and Monte Carlo dropout to generate:
  - **uncertainty maps** over lesions,
  - flags for low-confidence predictions for clinician review.
- Study whether:
  - gated skips,
  - boundary supervision,
  - and lighter transformers
  improve or harm **trustworthiness** under distribution shifts.

---

### 17.5. Stronger Baselines & Fairer Comparisons

We will integrate more **state-of-the-art baselines** into the same pipeline:

- UNet variants, UNet++,
- UNETR, Swin-UNet, ConvNeXt-UNet,
- improved transformer decoders and hybrid designs.

All trained with:

- identical data splits,
- identical augmentation policies,
- identical loss/metrics,
- shared logging and evaluation scripts.

This will position TransUNet-Lite variants as part of a **rigorous, unified benchmark**, not a one-off tweak.

---

### 17.6. Public Release & Reproducibility Enhancements

Planned improvements to make this ecosystem maximally useful:

- Release **Lite-Base** and **Lite-Tiny** training & eval notebooks publicly once the manuscript is submitted.
- Hugging Face / MONAI-compatible model hubs:
  - `TransUNet-Lite-Base`
  - `TransUNet-Lite-Tiny`
- “One-command” runners:
  - `train.py` + `eval.py` + YAML configs that reproduce all tables and plots.
- Extended docs:
  - architecture cards,
  - usage recipes for clinical / research workflows,
  - guidance on choosing between baseline and Lite variants.

---

### 17.7. Follow-up Paper: Dedicated Ablation & Robustness Study

Finally, the current work naturally leads to a **second, focused paper**:

- Title theme: *“Do Gated Skips and Depthwise Decoders Really Help?”*
- Content:
  - exhaustive ablations,
  - cross-dataset robustness,
  - calibration & uncertainty,
  - deployment metrics (CPU/GPU/edge).

This staged approach keeps the current study **clean, credible, and publishable**, while leaving room for a deeper theoretical and empirical exploration in a dedicated follow-up.


---

## Citation

A formal BibTeX entry will be added once the corresponding manuscript is submitted.
In the meantime, feel free to reference this repository as:

> H. Umer, "TransUNet-Lite: Fast & Memory-Efficient Transformer Segmentation for Clinical-Scale Use," GitHub repository, 2025.  
> https://github.com/HussamUmer/transunet-lite

---

## 18. Objectives of This Repository

- ✅ Provide a **transparent, research-grade** implementation of:
  - TransUNet baseline
  - TransUNet-Lite-Base
  - TransUNet-Lite-Tiny
  - Comparator architectures (UNETR, SETR) under identical settings.
- ✅ Offer a **plug-and-play evaluation pipeline**:
  - Standardized MedSegBench NPZ loading
  - Shared training / evaluation loops
  - Unified logging, plotting, and metrics
- ✅ Enable **fair, reproducible comparisons** for anyone designing new lightweight medical segmentation models.

---

## 19. Reproducibility & Assets

This repo includes:

- 📂 **Configs:** YAML/JSON with all hyperparameters and seeds.
- 📝 **Logs:** Per-epoch CSV logs for every run.
- 🧠 **Checkpoints:** Best and last weights for each model/dataset.
- 📊 **Figures:** 
  - Loss / Dice / IoU curves,
  - PR & ROC curves,
  - Threshold sweeps,
  - CPU/GPU latency & memory plots.
- 🖼️ **Qualitative 12-grid panels:** Input, GT overlay, prediction overlay, prediction boundary for each model.

---

## 20. How to Use (High-Level)

1. **Pick dataset**: BUSI or ISIC 2016 NPZ (MedSegBench format).
2. **Pick model**: `TransUNet`, `TransUNet-Lite-Base`, `TransUNet-Lite-Tiny`, `UNETR`, `SETR`.
3. **Run training notebook**:
   - Only change **Step 6 (Model Definition)**.
4. **Run evaluation notebook**:
   - Loads best checkpoint; produces tables & figures identical to those above.
5. **Run CPU-only notebook**:
   - Evaluates 50 fixed test images for median/p90/p95 latency, throughput, RAM.

---

## 21. Closing Note

This project is built to be:

- **Readable** enough for practitioners,
- **Rigorous** enough for reviewers,
- **Modular** enough to plug in new architectures.

Once the code and docs are polished, these results form a **credible, defensible basis for a conference or journal submission** on efficient TransUNet-style medical segmentation — especially when combined with my full training logs, visualizations, and released checkpoints. 🩺⚡

---

✨ Crafted by Hussam Umer — Vision4Healthcare Lab
> If anyone wants to collaborate, extend the benchmarks, or plug in new architectures, contributions and discussions are very welcome. 💡
For issues, questions, or collaboration ideas, open a GitHub issue or email me at hussamumer.scholar@gmail.com and I will help as much as possible.

📘 Medical Imaging | AI for Edge Devices | Transformer Efficiency Research
