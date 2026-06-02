# Pre-trained ONNX models

Pre-trained classifiers are **committed in this directory** on the default branch:

- [`family_classifier.onnx`](https://github.com/Supermagnum/radio-modulation-validator/blob/main/models/family_classifier.onnx) — modulation family (AM, FM, FSK, PSK, QAM, PAM)
- [`order_classifier.onnx`](https://github.com/Supermagnum/radio-modulation-validator/blob/main/models/order_classifier.onnx) — specific order within family (FP32)
- `family_classifier.meta.json` / `order_classifier.meta.json` — class vocabularies used by `rmv validate` and `rmv scan`

Shipped order labels follow the last training export. After adding synthetic **GMSK_BT05** /
**GMSK_BT03**, retrain and run `uv run rmv checksum update` so `order_classifier.meta.json`
lists the new classes. Validation aliases for the GMSK family are already in the tool; the
ONNX head must be retrained to stop **GMSK→NXDN** mislabels.

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

Quantise a single model:

```bash
uv run rmv export-quantised --synthetic datasets/synthetic/synthetic.npz --family-only
uv run rmv export-quantised --synthetic datasets/synthetic/synthetic.npz --order-only
```

### Deployed formats (shipped / recommended)

| File | Format | Inference |
|------|--------|-----------|
| `family_classifier_int8.onnx` | INT8 static (QDQ) | Used when present; 100% FP32 top-1 agreement on calibration |
| `family_classifier.onnx` | FP32 | Fallback if INT8 not generated |
| `order_classifier.onnx` | FP32 | **Always** used for order classification |
| `order_classifier_int8.onnx` | — | **Not produced** — static and dynamic INT8 exceed accuracy tolerance (~8%+ drop) for the 50-class order head with current calibration data |

`rmv export-quantised` verifies FP32 vs INT8 top-1 agreement on the calibration set. If both
static and dynamic INT8 fail for a model, it logs a warning, deletes the failed `*_int8.onnx`,
and leaves FP32 deployed. Family quantisation typically succeeds; order quantisation is expected
to fail — use `--family-only` to skip order export.

`ModulationClassifier` and `rmv scan` prefer INT8 for **family** only; **order** always loads
`order_classifier.onnx` even if a stale `order_classifier_int8.onnx` exists.

SpacemiT `.nb` conversion applies only to models with verified INT8 ONNX; see
[docs/npu-deployment.md](../docs/npu-deployment.md).

## Train your own models

See the root [README.md](../README.md) retraining section. After training and export:

```bash
uv run rmv export --checkpoint checkpoints/best_family_classifier.pt
uv run rmv export --checkpoint checkpoints/best_order_classifier.pt
uv run rmv checksum update
```

Checksums are recorded in `checksums.sha256` at the repository root.
