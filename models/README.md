# Pre-trained ONNX models

Committed models (when available):

- `family_classifier.onnx` - predicts modulation family (AM, FM, FSK, PSK, QAM, PAM)
- `order_classifier.onnx` - predicts specific modulation order within family

## Download pre-trained models

Download release artifacts from the project GitHub Releases page (placeholder URL):

```
https://github.com/example/radio-modulation-validator/releases/latest
```

Place both `.onnx` files and their `.meta.json` sidecars in this directory, then verify:

```bash
uv run rmv checksum verify
```

## Train your own models

See the root [README.md](../README.md) retraining section. After training and export:

```bash
uv run rmv export --checkpoint checkpoints/best_family_classifier.pt
uv run rmv export --checkpoint checkpoints/best_order_classifier.pt
uv run rmv checksum update
```

Checksums are recorded in `checksums.sha256` at the repository root.
