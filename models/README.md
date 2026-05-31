# Pre-trained ONNX models

Pre-trained classifiers are **committed in this directory** on the default branch:

- [`family_classifier.onnx`](https://github.com/Supermagnum/radio-modulation-validator/blob/main/models/family_classifier.onnx) — modulation family (AM, FM, FSK, PSK, QAM, PAM)
- [`order_classifier.onnx`](https://github.com/Supermagnum/radio-modulation-validator/blob/main/models/order_classifier.onnx) — specific order within family
- `family_classifier.meta.json` / `order_classifier.meta.json` — class vocabularies used by `rmv validate` and `rmv scan`

Browse all files: [models on GitHub](https://github.com/Supermagnum/radio-modulation-validator/tree/main/models).

## Use the shipped models

After cloning or pulling the repository, verify integrity (optional but recommended):

```bash
uv sync --extra dev
uv run rmv checksum verify
```

No separate download step is required when you use a full clone that includes `models/*.onnx`.

If you use a sparse checkout or a source-only tarball without `models/`, restore the files from
[the models directory on `main`](https://github.com/Supermagnum/radio-modulation-validator/tree/main/models)
or download release assets when published:

https://github.com/Supermagnum/radio-modulation-validator/releases/latest

Place `family_classifier.onnx`, `order_classifier.onnx`, and both `.meta.json` sidecars in this
directory, then run `uv run rmv checksum verify`.

## INT8 quantisation (NPU / faster CPU)

Quantise committed FP32 models with synthetic calibration data:

```bash
uv run rmv dataset generate-synthetic --output datasets/synthetic/
uv run rmv export-quantised --synthetic datasets/synthetic/synthetic.npz
uv run rmv checksum update
```

This writes `family_classifier_int8.onnx` and `order_classifier_int8.onnx`. Inference
prefers INT8 when present. SpacemiT `.nb` conversion is optional; see
[docs/npu-deployment.md](../docs/npu-deployment.md).

## Train your own models

See the root [README.md](../README.md) retraining section. After training and export:

```bash
uv run rmv export --checkpoint checkpoints/best_family_classifier.pt
uv run rmv export --checkpoint checkpoints/best_order_classifier.pt
uv run rmv checksum update
```

Checksums are recorded in `checksums.sha256` at the repository root.
