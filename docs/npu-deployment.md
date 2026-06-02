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

`ModulationClassifier` and `rmv scan` use INT8 for **family** when
`family_classifier_int8.onnx` exists; **order** always uses FP32 `order_classifier.onnx`.

Use `--family-only` if you only need family INT8 (recommended for 50-class order models).

## Workflow (K3 device, no cross-compile SDK)

1. On the dev machine: `rmv export-quantised --family-only` (produces verified family INT8).
2. Copy `models/family_classifier_int8.onnx` and `models/order_classifier.onnx` (FP32) to the K3.
3. On the K3, with the SpacemiT SDK installed:

```bash
spacemit-npu-convert models/family_classifier_int8.onnx \
  --precision int8 --output models/family_classifier.nb
# Order: run FP32 on CPU or re-quantise on-device with larger calibration if needed
```

Or use `rmv export-npu --calibration datasets/synthetic/synthetic.npz` on the K3.

`.nb` files are gitignored; INT8 ONNX files may be committed (~650 KB each).

## Output layout

```
models/
  family_classifier.onnx          # FP32 reference
  family_classifier_int8.onnx     # INT8 (CPU or NPU input)
  family_classifier.nb            # NPU binary (local build only)
  order_classifier.onnx           # FP32 (always used for inference)
  *.meta.json
```

`order_classifier_int8.onnx` is not deployed: INT8 verification typically fails for the
50-class order head (>8% accuracy drop vs FP32 on the calibration set).

## Accuracy

The **family** classifier uses static QDQ INT8 (calibrated on synthetic IQ) and must pass
verification (default: FP32 vs INT8 top-1 disagreement ≤ `--tolerance`, 3.0%).

The **order** classifier tries static QDQ, then dynamic weight-only INT8 if
`--no-order-dynamic-fallback` is not set. If both exceed tolerance, export keeps FP32 only,
removes any failed `order_classifier_int8.onnx`, and logs a warning (exit code 0 if family
INT8 succeeded).

Use `--family-only` to quantise only the family model. Use `--order-only` to attempt order
INT8 in isolation (for experiments).

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
