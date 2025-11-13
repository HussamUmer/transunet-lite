# TransUNet-Lite-Tiny — Model Card (Stub)

**Summary.** Ultra-compact variant optimized for speed and memory on CPU/edge while keeping competitive accuracy.

## Architecture (high level)
- **Backbone:** DeiT-Tiny/16 (ImageNet pretrained)
- **Skip path:** Lite CNN at {H/4, H/8, H/16}
- **Decoder:** Depthwise-separable U-Net decoder
- **Gates:** SE-style gated skip connections
- **Heads:** Segmentation head (+ optional boundary head during training)

## Intended use
- CPU-only or embedded inference where **latency/RAM** dominate

## Expected footprint (reference)
- **Params:** ~6.8M
- **Peak VRAM (GPU):** ~0.22 GB @ 256×256, BS=1
- **Latency (GPU):** ~52 ms/img (T4)
- **Latency (CPU):** ~0.27 s median/img (50-img set)
- **Throughput (CPU):** ~3.7 FPS (256×256)

## Strengths
- Fastest & lowest memory in the family
- Conservative predictions with low false positives

## Limitations
- Slight under-segmentation on noisy BUSI edges
- Accuracy < Lite-Base / baseline on hardest cases

## Release status
- **Code/weights** to be released **upon manuscript submission**.

