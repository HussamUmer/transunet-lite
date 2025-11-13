# TransUNet-Lite-Base — Model Card (Stub)

**Summary.** Lightweight TransUNet variant targeting near-baseline accuracy with a 3–4× memory reduction.

## Architecture (high level)
- **Backbone:** ViT-S/16 (ImageNet pretrained)
- **Skip path:** Lite CNN at {H/4, H/8, H/16}
- **Decoder:** Depthwise-separable U-Net decoder
- **Gates:** SE-style gated skip connections
- **Heads:** Segmentation head (+ optional boundary head during training)

## Intended use
- Medical binary segmentation at **256×256** (dermoscopy, ultrasound)
- GPU with limited VRAM (≤6–8 GB) or **CPU-only** evaluation

## Expected footprint (reference)
- **Params:** ~23.3M
- **Peak VRAM (GPU):** ~0.50 GB @ 256×256, BS=1
- **Latency (GPU):** ~57 ms/img (T4)
- **Latency (CPU):** ~0.63 s median/img (50-img set)

## Training/eval protocol
- MedSegBench splits, Dice+BCE loss (0.7/0.3), identical augs across models

## Strengths
- Near-baseline Dice/IoU on ISIC 2016 with large VRAM savings
- Stable TTA uncertainty structure

## Limitations
- Slight contour softness on BUSI versus heavy baseline
- Results reported at 256×256; larger inputs increase cost

## Release status
- **Code/weights** to be released **upon manuscript submission**.

