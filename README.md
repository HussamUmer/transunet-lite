# 🚀 Lite Yet Sharp: Gated Skips and Depthwise Decoders for Fast TransUNet Segmentation
## Results Overview
> **Status:** 🚧 *This repository is under construction.* Interfaces, plots, and docs may change as we add more results.

### What we evaluated
- **Dataset:** ISIC 2016 (skin-lesion segmentation), images resized to 256×256.  
- **Fair setup:** identical data splits, preprocessing, augmentations, losses, metrics, logging, and evaluation for all models.  
- **Models compared:**
  1. **TransUNet (baseline):** ResNet-50 skips + ViT-B/16 transformer + standard decoder.  
  2. **TransUNet-Lite-Base:** ViT-S/16 + lightweight CNN skip path + depthwise-separable decoder + squeeze–excitation gates + boundary head.  
  3. **TransUNet-Lite-Tiny:** DeiT-Tiny/16 + the same lightweight decoder and gates as above.

### Headline takeaways (plain language)
- **Lite-Base** delivers **almost the same segmentation quality as the baseline** while being **much smaller and more memory-efficient**.  
  - About **4.7× fewer parameters**, **~5.3× less compute**, **~3.2× lower peak VRAM**, and **~7% faster** per image.  
- **Lite-Tiny** is **extremely efficient** with **~16× fewer parameters** and **~5× lower peak VRAM**, trading **a small amount of accuracy** for speed and compactness.

### Final test quality (ISIC 2016, 256×256)
| Model | Dice score (higher is better) |
|---|---:|
| TransUNet (baseline) | **0.917** |
| TransUNet-Lite-Base | **0.916** |
| TransUNet-Lite-Tiny | **0.900** |

> **Interpretation:** Lite-Base essentially matches the baseline on mask quality; Lite-Tiny gives a stronger efficiency–speed profile with a modest quality trade-off.

### Efficiency and resources
| Model | Parameters (millions) | Compute (multiply–accumulate, billions) | Peak VRAM (MB) | Inference time (ms / image) |
|---|---:|---:|---:|---:|
| TransUNet (baseline) | **108.77** | **39.93** | **4554.0** | **1094.44** |
| TransUNet-Lite-Base | **23.30** | **7.59** | **1430.6** | **1016.31** |
| TransUNet-Lite-Tiny | **6.81** | **2.68** | **863.7** | **1003.70** |

**Relative savings vs baseline**
- **Lite-Base:** ~4.7× fewer parameters, ~5.3× less compute, ~3.2× lower memory, ~7% faster per image.  
- **Lite-Tiny:** ~16× fewer parameters, ~15× less compute, ~5.3× lower memory, ~8% faster per image.

### Additional evidence (test set)
- **Precision–recall and ROC areas:** all three models are **very strong and very close** (around 0.98 for PR area and 0.991–0.992 for ROC area).  
- **Calibration:** smaller models are **slightly less calibrated** (Lite-Tiny the most); simple temperature scaling can improve this.  
- **Threshold sweep:** best Dice appears between **0.55 and 0.75** across models; reporting uses a fixed **0.50** threshold for comparability.

### What to look at in this repo
- **Side-by-side training logs** (same format across models).  
- **Interactive plots** for loss, Dice score, and intersection-over-union across training.  
- **Post-training test report** with latency, memory, and parameter counts.  
- **Qualitative panels:** input image, ground-truth overlay, predicted overlay, and predicted boundary outline (generated from the thresholded probability map).

---
**Bottom line:** If baseline-level quality with far fewer parameters and much lower memory is needed, choose **TransUNet-Lite-Base**. For an ultra-light, fast model with a small quality trade-off, choose **TransUNet-Lite-Tiny**.

