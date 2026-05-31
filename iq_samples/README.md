# Contributed IQ samples

This directory holds IQ captures from GNU Radio out-of-tree modulator blocks used
for validation. See [docs/contributing_iq_samples.md](../docs/contributing_iq_samples.md)
for the full submission guide.

Contributed IQ is for `rmv validate` / `rmv classify` only — **not** for `rmv train`.
Custom or multi-carrier modes (for example several QPSK carriers in one wideband `.iq`
without per-carrier files) are not detectable as named modes; only standard family/order
labels apply. See the contributing guide and [README.md](../README.md).

## File format

Each contribution consists of:

1. **IQ data** - either:
   - SigMF: `name.sigmf-data` + `name.sigmf-meta`, or
   - Simple binary: `name.iq`
2. **Metadata sidecar** - `name.json` (required for `.iq` files)

### `.iq` binary format

- Raw interleaved **float32** I/Q pairs, **little-endian**
- **1024 complex samples** per chunk = **2048** float32 values
- File length must be a multiple of 2048 float32 values
- Multiple chunks per file are allowed
- **Maximum file size: 50 MB**

### Sidecar JSON example

```json
{
  "source": "gr-qradiolink",
  "block_name": "mod_nbfm",
  "expected_family": "FM",
  "expected_order": "NBFM",
  "sample_rate_hz": 48000,
  "center_freq_hz": 0,
  "snr_db": null,
  "notes": "Captured from GNU Radio flowgraph"
}
```

Valid `source` values: `gr-qradiolink`, `gr-packet-protocols`, `gr-sleipnir`, `other`.

Valid `expected_family` values: `FM`, `FSK`, `PSK`, `QAM`, `AM`, `PAM`.

Run validation:

```bash
uv run rmv validate iq_samples/gr-qradiolink/mod_nbfm.iq
```
