# Contributing IQ samples

This guide explains how to capture IQ from GNU Radio modulator blocks and submit
validation fixtures via pull request.

## What contributed IQ is for

Contributed `.iq` files are **inference fixtures** only. They are checked with
`rmv validate` or `rmv classify` against **existing** ONNX models. They are **not**
used by `rmv train`.

| Path | Input | Purpose |
|------|--------|---------|
| **Training** | RadioML pickle, HISARMOD HDF5, CSPB directory (`.tim` + truth file) | Fit or refresh classifiers — see [dataset_preparation.md](dataset_preparation.md) |
| **Contributed IQ** | `.iq` + `.json` sidecar | Validate OOT modulator output |

Training does not accept folders of `.iq` files and cannot add new modulation labels.
Retraining only covers standard families and orders from the public datasets (BPSK,
QPSK, 2FSK, NBFM, and so on). Do not commit training datasets or checkpoints in IQ PRs.

## Custom and multi-carrier modes

Sidecar labels must use the **standard** `expected_family` and `expected_order` values
the classifiers know (`PSK`, `QPSK`, `2FSK`, etc.). The tool does **not** detect custom
air interfaces or link-layer modes as named classes.

**Not supported:** certifying a bespoke composite mode from **one** wideband `.iq` file
with **no per-carrier splits**. For example, eight parallel QPSK carriers in a single
capture (per-carrier baud, pulse shaping, fixed carrier spacing) cannot be submitted as
a new mode the validator will recognize. There is no `8xQPSK` (or similar) label.

- `rmv classify` on combined IQ may sometimes report generic **PSK** / **QPSK** because
  each carrier uses a QPSK constellation; treat that as an unreliable sanity check, not
  mode detection.
- `rmv validate` only checks predictions against **your** sidecar (`PSK` / `QPSK`, etc.).
  It does not verify carrier count, spacing, symbol rate, or pulse shape.
- Putting those parameters in `notes` documents the capture; it does not enable detection.

Models were trained on **one modulation type per chunk**, not multiple stacked carriers in
one spectrum. For multi-carrier GNU Radio outputs, use **separate IQ files per carrier**
when you need trustworthy family/order validation.

See also [README.md](../README.md) — **Training vs IQ inference** and **Custom and
multi-carrier modes (IQ)**.

## Step 1: Capture IQ from a GNU Radio block

1. Build a flowgraph with the OOT modulator under test (e.g. `mod_nbfm` from
   gr-qradiolink).
2. Insert a sink that writes complex float32 samples (File Sink or custom block).
3. Capture at least **1024 samples** (one chunk); prefer several seconds for
   multiple chunks.
4. Export as interleaved float32 `.iq` (little-endian) or SigMF.

Example GNU Radio File Sink settings for `.iq`:

- Type: complex
- Convert to interleaved float32 I/Q in a short Python post-process if needed

## Step 2: Create the JSON sidecar

For `myblock.iq`, create `myblock.json`:

| Field | Description |
|-------|-------------|
| `source` | Repository: `gr-qradiolink`, `gr-packet-protocols`, `gr-sleipnir`, or `other` |
| `block_name` | GNU Radio block name (e.g. `mod_nbfm`) |
| `expected_family` | `FM`, `FSK`, `PSK`, `QAM`, `AM`, or `PAM` |
| `expected_order` | Specific mode (e.g. `NBFM`, `QPSK`, `2FSK`) |
| `sample_rate_hz` | Sample rate used in capture |
| `center_freq_hz` | Center frequency (0 for baseband) |
| `snr_db` | Optional measured SNR |
| `notes` | Free text (capture conditions only; does not define a new detectable mode) |

## Step 3: Validate locally

```bash
uv sync
# Download models per models/README.md
uv run rmv validate iq_samples/<repo>/<block>.iq --verbose
```

Exit codes: `0` pass, `1` fail, `2` hard fail (wrong family or very low confidence).

## Expected family / order by repository

### gr-qradiolink (representative modes)

| Block | expected_family | expected_order |
|-------|-----------------|----------------|
| mod_nbfm | FM | NBFM |
| mod_wbfm | FM | WBFM |
| mod_bpsk | PSK | BPSK |
| mod_qpsk | PSK | QPSK |
| mod_2fsk | FSK | 2FSK |
| mod_4fsk | FSK | 4FSK |
| mod_am_dsb | AM | AM-DSB |

### gr-packet-protocols

| Block | expected_family | expected_order |
|-------|-----------------|----------------|
| mod_dmr | FSK | 4FSK |
| mod_dstar | PSK | QPSK |
| mod_ysf | FSK | 4FSK |
| mod_p25 | FSK | 4FSK |

### gr-sleipnir

| Block | expected_family | expected_order |
|-------|-----------------|----------------|
| mod_carrier_0 | (per carrier) | (per mode) |
| SleipnirTxHier (8-carrier composite) | `custom` | `sleipnir_8qpsk` |

**Composite 8-carrier QPSK** in one wideband `.iq` file: use the
`sleipnir_8qpsk` plugin (`expected_family: "custom"`). The plugin analyses the full
composite spectrum and does not require per-carrier file splits. See
[contributing_plugins.md](contributing_plugins.md).

**Per-carrier validation** (standard CNN) still uses separate IQ files and
`expected_family` / `expected_order` for each carrier (PSK / QPSK, etc.).

## Step 4: Submit via pull request

1. Place files under `iq_samples/<source>/` (max **50 MB** per file).
2. Do not commit training datasets or model checkpoints.
3. Run `uv run pytest` and `uv run rmv validate` on your samples.
4. Open a PR describing the block and capture conditions.

Maintainers will review classifier results under `validation_results/` after CI runs.
