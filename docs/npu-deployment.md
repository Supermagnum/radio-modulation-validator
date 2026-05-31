# NPU deployment (SpacemiT K3)

This guide covers INT8 quantisation of the rmv ONNX classifiers and conversion to
SpacemiT NPU `.nb` binaries for the K3 A100 cores. CPU-only deployments can use
INT8 ONNX directly via onnxruntime without the SpacemiT SDK.

## Overview

| Step | Command | SDK required |
|------|---------|--------------|
| Train and export FP32 | `rmv train`, `rmv export` | PyTorch (train extra) |
| Quantise to INT8 | `rmv export-quantised` | onnxruntime only |
| Convert to NPU | `rmv export-npu` or `spacemit-npu-convert` | SpacemiT NPU SDK |
| Verify checksums | `rmv checksum update`, `rmv checksum verify` | — |

Calibration uses `datasets/synthetic/synthetic.npz` (high-SNR chunks by default).
Generate it with `rmv dataset generate-synthetic` if missing.

## Workflow (development machine)

```bash
uv sync --extra train
rmv train --device cuda
rmv export --checkpoint checkpoints/best_family_classifier.pt
rmv export --checkpoint checkpoints/best_order_classifier.pt

rmv dataset generate-synthetic --output datasets/synthetic/

rmv export-quantised \
  --synthetic datasets/synthetic/synthetic.npz \
  --calibration-chunks 512 \
  --tolerance 3.0

rmv checksum update
rmv checksum verify
```

`ModulationClassifier` and `rmv scan` prefer `*_int8.onnx` when those files exist.

## Workflow (K3 device, no cross-compile SDK)

1. On the dev machine: `rmv export-quantised` (produces `*_int8.onnx` only).
2. Copy `models/family_classifier_int8.onnx` and `models/order_classifier_int8.onnx`
   to the K3.
3. On the K3, with the SpacemiT SDK installed:

```bash
spacemit-npu-convert models/family_classifier_int8.onnx \
  --precision int8 --output models/family_classifier.nb
spacemit-npu-convert models/order_classifier_int8.onnx \
  --precision int8 --output models/order_classifier.nb
```

Or use `rmv export-npu --calibration datasets/synthetic/synthetic.npz` on the K3.

`.nb` files are gitignored; INT8 ONNX files may be committed (~650 KB each).

## Output layout

```
models/
  family_classifier.onnx          # FP32 reference
  family_classifier_int8.onnx     # INT8 (CPU or NPU input)
  family_classifier.nb              # NPU binary (local build only)
  order_classifier.onnx
  order_classifier_int8.onnx
  order_classifier.nb
  *.meta.json
```

## Accuracy

The **family** classifier uses static QDQ INT8 (calibrated on synthetic IQ). The
**order** classifier (43 classes) may fall back to **dynamic** weight-only INT8 if
static QDQ exceeds `--tolerance` on the calibration set (typical static agreement ~86%,
dynamic ~99% on the same data).

`rmv export-quantised` compares FP32 and INT8 top-1 labels on the calibration set and
fails if disagreement exceeds `--tolerance` (default 3.0%). Disable order fallback with
`--no-order-dynamic-fallback`.

If static order quantisation fails and you need static QDQ for NPU, increase
`--calibration-chunks` or relax tolerance; NPU toolchains may require re-running
`spacemit-npu-convert` on the K3 with device-specific calibration.

## Performance (K3 estimates)

| Model | Format | Approx. size | Latency |
|-------|--------|--------------|---------|
| family_classifier | FP32 ONNX | ~2.6 MB | ~15 ms (CPU) |
| family_classifier | INT8 ONNX | ~650 KB | ~5 ms (CPU) |
| family_classifier | INT8 .nb | ~650 KB | ~15 us (NPU) |
| order_classifier | INT8 .nb | ~650 KB | ~15 us (NPU) |

Both classifiers on NPU in sequence: ~30 us total for one 1024-sample chunk at 48 kHz.

## References

- [ONNX Runtime quantisation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [SpacemiT developer portal](https://developer.spacemit.com/)
