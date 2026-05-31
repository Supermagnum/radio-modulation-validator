# radio-modulation-validator

Validate AI-generated GNU Radio out-of-tree (OOT) modulator blocks by classifying
IQ samples against known modulation types. The tool checks that modulator blocks
produce waveforms matching expected modulation **families** (FM, FSK, PSK, QAM, AM,
PAM) and **orders** (NBFM, QPSK, 2FSK, etc.) using ONNX classifiers trained on
public RF ML datasets.

## Table of contents

- [Modulation coverage, training data, and synthesis](#modulation-coverage-training-data-and-synthesis)
  - [Complete order and bandwidth reference](#complete-order-and-bandwidth-reference)
  - [Families (6)](#families-6)
  - [RadioML 2016.10A (11 orders)](#radioml-201610a-11-orders)
  - [HISARMOD 2019.1 (26 orders)](#hisarmod-20191-26-orders)
  - [CSPB.ML.2018R2 (8 orders)](#cspbml2018r2-8-orders)
  - [Synthetic orders (12) — bandwidths verified at generation](#synthetic-orders-12--bandwidths-verified-at-generation)
  - [Custom-mode plugins (not CNN orders)](#custom-mode-plugins-not-cnn-orders)
  - [Where training data comes from](#where-training-data-comes-from)
  - [How synthetic data is created](#how-synthetic-data-is-created)
- [Quick start](#quick-start)
- [Python API](#python-api)
- [CLI](#cli)
- [Usage examples](#usage-examples)
  - [Validate one GNU Radio modulator (standard CNN)](#validate-one-gnu-radio-modulator-standard-cnn)
  - [Classify without a sidecar](#classify-without-a-sidecar)
  - [Validate composite 8-carrier QPSK (custom plugin)](#validate-composite-8-carrier-qpsk-custom-plugin)
  - [Prepare datasets and retrain](#prepare-datasets-and-retrain)
  - [Report and checksums](#report-and-checksums)
  - [Python API (usage)](#python-api-1)
- [Training vs IQ inference](#training-vs-iq-inference)
- [Custom and multi-carrier modes (IQ)](#custom-and-multi-carrier-modes-iq)
- [Contributing IQ samples](#contributing-iq-samples)
- [Training datasets](#training-datasets)
- [Retraining models](#retraining-models)
- [Dataset version pinning (released models)](#dataset-version-pinning-released-models)
- [Validation methodology](#validation-methodology)
- [Known limitations](#known-limitations)
- [Development](#development)
- [License](#license)

## Modulation coverage, training data, and synthesis

Inference uses two ONNX classifiers on 1024-sample IQ chunks (interleaved float32,
typically 48 kHz):

1. **Family** — coarse type: `AM`, `FM`, `FSK`, `PSK`, `QAM`, `PAM`
2. **Order** — specific modulation label within that family (for example `QPSK`, `NBFM_25`)

`rmv validate` compares predictions to `expected_family` / `expected_order` in the
sidecar JSON. Composite air interfaces use **custom-mode plugins** (`expected_family:
"custom"`) instead of the CNN; see [Custom and multi-carrier modes](#custom-and-multi-carrier-modes-iq).

### Complete order and bandwidth reference

The order classifier is trained on labels from public datasets plus optional synthetic
data. Shipped ONNX models expose the class list in `models/*_classifier.meta.json`;
the tables below are the full union defined in the codebase (`src/rmv/constants.py`).

**CNN training / inference chunk format (all standard orders):** each example is
**1024** IQ samples as interleaved float32, shape `(2, 1024)`, nominal sample rate
**48 kHz** (RadioML windows are upsampled from 128-sample source chunks). Per-class RF
bandwidth for public datasets is whatever each dataset’s synthesizer produces; rmv does
not pin a Hz limit per RadioML/HISARMOD/CSPB label.

#### Families (6)

| Family | Role |
|--------|------|
| `AM` | Amplitude modulation (DSB, SSB, ASK, aviation voice AM) |
| `FM` | Frequency modulation (wide broadcast FM, narrowband FM) |
| `FSK` | Frequency-shift keying (CPFSK, GFSK, GMSK, MSK, M-FSK) |
| `PSK` | Phase-shift keying (BPSK through 32PSK, OQPSK, DQPSK) |
| `QAM` | Quadrature amplitude modulation |
| `PAM` | Pulse amplitude modulation (`PAM4`) |

#### RadioML 2016.10A (11 orders)

| Order | Family | RF bandwidth / notes |
|-------|--------|----------------------|
| `AM-DSB` | AM | Broadcast-style DSB; wide audio (~20 kHz class in literature); not aviation AM |
| `AM-SSB` | AM | Single-sideband; dataset-defined synthesis |
| `WBFM` | FM | Wideband FM; dataset-defined synthesis |
| `BPSK` | PSK | Digital; bandwidth set by symbol rate in RadioML GNU Radio flowgraphs |
| `QPSK` | PSK | Digital; dataset-defined |
| `8PSK` | PSK | Digital; dataset-defined |
| `16QAM` | QAM | Digital; dataset-defined |
| `64QAM` | QAM | Digital; dataset-defined |
| `CPFSK` | FSK | Continuous-phase FSK; dataset-defined |
| `GFSK` | FSK | Gaussian FSK; dataset-defined |
| `PAM4` | PAM | Four-level PAM; dataset-defined |

#### HISARMOD 2019.1 (26 orders)

| Order | Family | RF bandwidth / notes |
|-------|--------|----------------------|
| `2FSK` | FSK | M-FSK; order sets tone spacing / symbol rate in HDF5 synthesis |
| `4FSK` | FSK | M-FSK; dataset-defined |
| `8FSK` | FSK | M-FSK; dataset-defined |
| `16FSK` | FSK | M-FSK; dataset-defined |
| `32FSK` | FSK | M-FSK; dataset-defined |
| `64FSK` | FSK | M-FSK; dataset-defined |
| `128FSK` | FSK | M-FSK; dataset-defined |
| `256FSK` | FSK | M-FSK; dataset-defined |
| `BPSK` | PSK | Digital; includes multiple fading profiles in HDF5 |
| `QPSK` | PSK | Digital; dataset-defined |
| `8PSK` | PSK | Digital; dataset-defined |
| `16PSK` | PSK | Digital; dataset-defined |
| `32PSK` | PSK | Digital; dataset-defined |
| `16QAM` | QAM | Digital; dataset-defined |
| `32QAM` | QAM | Digital; dataset-defined |
| `64QAM` | QAM | Digital; dataset-defined |
| `128QAM` | QAM | Digital; dataset-defined |
| `256QAM` | QAM | Digital; dataset-defined |
| `AM-DSB` | AM | Analog; dataset-defined |
| `AM-SSB` | AM | Analog; dataset-defined |
| `FM` | FM | Generic FM label in HISARMOD (not `NBFM_25` / `NBFM_50`) |
| `GMSK` | FSK | MSK family mapping in rmv |
| `OQPSK` | PSK | Offset QPSK; dataset-defined |
| `MSK` | FSK | Minimum-shift keying |
| `OOK` | AM | On-off keying |
| `4ASK` | AM | Amplitude-shift keying |

#### CSPB.ML.2018R2 (8 orders)

| Order | Family | RF bandwidth / notes |
|-------|--------|----------------------|
| `BPSK` | PSK | Independent cyclostationary.blog generator; `.tim` + truth file |
| `QPSK` | PSK | Dataset-defined |
| `8PSK` | PSK | Dataset-defined |
| `DQPSK` | PSK | Differential QPSK |
| `16QAM` | QAM | Dataset-defined |
| `64QAM` | QAM | Dataset-defined |
| `256QAM` | QAM | Dataset-defined |
| `MSK` | FSK | Dataset-defined |

Use **R2 only** (not original CSPB 2018; see [Known limitations](#known-limitations)).

#### Synthetic orders (12) — bandwidths verified at generation

Generated by `rmv dataset generate-synthetic` (`src/rmv/dataset/synthetic.py`).
Occupied bandwidth is checked at **−26 dB** from the PSD peak on **clean** reference chunks
(no AWGN) before noisy data is saved. Protocol 4FSK modes also check instantaneous frequency
structure (four tones, symbol rate).

**FM, AM, and PSK (7 orders)**

| Order | Family | Peak deviation / modulation index | Audio passband (Hz) | Channel spacing | Carson / occupied BW (typical) | Max occupied BW (verify) |
|-------|--------|-----------------------------------|---------------------|-------------------|-------------------------------|---------------------------|
| `NBFM_25` | FM | ±2.5 kHz (`max_dev=2500`) | 300–800 (voice-like mix) | 12.5 kHz (IARU Region 1 amateur) | ~6 kHz (Carson: 2×(Δf+f_m)) | ≤7 kHz |
| `NBFM_50` | FM | ±5.0 kHz (`max_dev=5000`) | 300–1500 | 25 kHz (older amateur repeaters) | ~11 kHz | ≤13 kHz |
| `AM_AIR_25K` | AM | m=0.85 (DSB, full carrier) | 300–3400 (ICAO Annex 10 voice) | 25 kHz (worldwide airband) | ~7 kHz occupied | ≤8 kHz |
| `AM_AIR_833` | AM | m=0.85 | 300–2500 (EU 8.33 kHz channel) | 8.33 kHz (EU Reg. 1079/2012, FL195+) | ~5–6 kHz | ≤6.5 kHz |
| `WBFM` | FM | ±75 kHz (`max_dev=75000`) | ~50 Hz–15 kHz (broadcast-like) | — | ~200 kHz occupied | ≤200 kHz |
| `BPSK` | PSK | 8 samples/symbol @ 48 kHz | — | — | Digital; GNU Radio `psk_mod` | dataset verify |
| `QPSK` | PSK | 8 samples/symbol @ 48 kHz | — | — | Digital; GNU Radio `psk_mod` | dataset verify |

NBFM uses **no preemphasis** (`tau=0`); broadcast FM uses `tau=75 µs` for `WBFM` only.
Aviation AM is **not** RadioML `AM-DSB` (no broadcast preemphasis or ~20 kHz audio).
`WBFM`, `BPSK`, and `QPSK` replace weak or upsample-broken RadioML windows used only after
128→1024 upsampling.

**Protocol 4FSK (5 orders)** — modulation layer only (no framing, sync, or vocoders):

| Order | Family | Symbol rate | Tone deviations | Pulse shaping |
|-------|--------|-------------|-----------------|---------------|
| `DMR` | FSK | 4800 baud | ±648 / ±1944 Hz (4-ary) | Raised cosine, α=0.2 |
| `M17` | FSK | 4800 baud | ±800 / ±2400 Hz | Root raised cosine, β=0.5 |
| `YSF` | FSK | 4800 baud | ±800 / ±2400 Hz (C4FM map) | Gaussian, BT=0.5 |
| `NXDN` | FSK | 2400 baud | ±350 / ±1050 Hz | Raised cosine, α=0.2 |
| `dPMR` | FSK | 2400 baud | ±350 / ±1050 Hz | Raised cosine, α=0.2 |

D-Star and generic **GMSK** scan modes use the scan IQ generator’s MSK/GMSK path (4800 baud,
BT=0.5), not separate synthetic order labels. **NXDN** and **dPMR** share identical synthetic
parameters by design; the classifier may confuse them.

#### Custom-mode plugins (not CNN orders)

Composite waveforms use `expected_family: "custom"` and a plugin `mode_id` (see
[docs/contributing_plugins.md](docs/contributing_plugins.md)).

| mode_id | Description | Bandwidth / structure checked |
|---------|-------------|-------------------------------|
| `sleipnir_8qpsk` | 8 parallel QPSK carriers in one `.iq` | Carrier spacing **1300±150 Hz** (std ≤160 Hz); per-carrier **3 dB BW 800–1400 Hz**; symbol rate **900±50 Hz** on ≥6 carriers; total occupied **9–12 kHz** (−20 dB); nominal positions ±650…±4550 Hz |

#### Labels in `ORDER_CLASSES` without a dedicated training source row

These names appear in the unified order list for validation sidecars and retrained
heads; map to families via dataset rules above: `NBFM`, `WBFM`, `2FSK`, `4FSK`, `DQPSK`,
`MSK` (overlaps with dataset-specific rows).

RadioML and CSPB do **not** include amateur narrowband FM, aviation airband AM, or
protocol-specific 4FSK orders as distinct labels; synthetic data fills those gaps.
Broadcast `AM-DSB` must not be used as a stand-in for `AM_AIR_25K` / `AM_AIR_833`.

### Where training data comes from

| Source | Format | Role |
|--------|--------|------|
| [RadioML 2016.10A](https://www.deepsig.ai/datasets) | `RML2016.10a_dict.pkl` | Primary families (FM/FSK/PSK/QAM/AM); GNU Radio–style synthetic IQ |
| [HISARMOD 2019.1](https://github.com/kit-cel/HisarMod2019.1) | HDF5 (`.h5`) | Extra FSK orders and fading channels |
| [CSPB.ML.2018R2](https://cspb.ml/) | `.tim` batches + truth file | Independent generator; cross-check against RadioML bias |
| **Synthetic** (this repo) | `datasets/synthetic/synthetic.npz` | NBFM, aviation AM, WBFM, BPSK, QPSK, protocol 4FSK; generated locally, not downloaded |

Download public sets with `rmv dataset download` (see
[docs/dataset_preparation.md](docs/dataset_preparation.md)). Training data stays
under `datasets/` and is gitignored; only loaders and the generation script are
committed.

### How synthetic data is created

Synthetic IQ is produced by `src/rmv/dataset/synthetic.py` and written with
`rmv dataset generate-synthetic`. It is **ground truth for retraining**, not for
validating OOT blocks under test.

**Important:** Generation uses **GNU Radio built-in blocks** (`gnuradio.analog`,
`gnuradio.filter`) and **numpy/scipy** only. It does **not** import or call
**gr-qradiolink** or any other OOT module — using those blocks would make validation
circular.

See the [Synthetic orders](#synthetic-orders-12--bandwidths-verified-at-generation) tables
above for deviation, audio passband, channel spacing, and occupied-bandwidth limits.
Implementation summary:

| Class | Implementation |
|-------|----------------|
| `NBFM_25` / `NBFM_50` | Audio 8 kHz → interp to 48 kHz → GNU Radio `frequency_modulator_fc`; **no** preemphasis (`tau=0`) |
| `AM_AIR_25K` / `AM_AIR_833` | NumPy/scipy DSB-AM full carrier: `s(t)=1+m·a(t)` at baseband |
| `WBFM` | Band-limited audio → interp → `frequency_modulator_fc` with broadcast deviation (±75 kHz) |
| `BPSK` / `QPSK` | GNU Radio `digital.psk_mod`, 8 samples/symbol @ 48 kHz |
| `DMR` / `M17` / `YSF` / `NXDN` / `dPMR` | NumPy/scipy 4FSK: symbol map, deviation, RC / RRC / Gaussian shaping (`verify_4fsk_signal`) |

Per chunk: random excitation (voice-band audio, PSK symbols, or 4FSK symbols), optional AWGN
(−20 to +30 dB in 2 dB steps), random ±200 Hz frequency offset, unit-power normalisation,
shape `(2, 1024)` float32. Default run: 1000 chunks × 26 SNR levels × **12** classes ≈
**312k** samples. Bandwidth and 4FSK structure checks use clean reference chunks before
noisy data is kept.

Upstream `analog.nbfm_tx(tau=0)` fails in GNU Radio (preemphasis divide-by-zero); the
synthetic path uses the equivalent **interp + `frequency_modulator_fc`** without
preemphasis, matching amateur NFM intent.

```bash
# All 12 modes (default)
uv run rmv dataset generate-synthetic --output datasets/synthetic/

# Subset, e.g. protocol 4FSK only
uv run rmv dataset generate-synthetic --modes dmr,m17,ysf,nxdn,dpmr --output datasets/synthetic/

uv run rmv train --synthetic datasets/synthetic/ ...   # include in retraining
```

## Quick start

```bash
# Install with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev

# Download pre-trained ONNX models (see models/README.md)
# Place family_classifier.onnx and order_classifier.onnx in models/

uv run rmv checksum verify
uv run rmv validate iq_samples/gr-qradiolink/mod_example.iq
```

Inference requires only `onnxruntime` and committed `models/*.onnx` — no PyTorch.

Do **not** use bare `python3` from the OS unless you installed `rmv` into that
interpreter. Use the project environment:

```bash
cd radio-modulation-validator
uv sync --extra dev          # inference + tests
uv sync --extra train        # add PyTorch for rmv train (optional)

# Run checks (use venv Python, not system python3)
uv run python scripts/check_env.py
# or:
.venv/bin/python scripts/check_env.py
```

| Check | Required for | Install |
|-------|----------------|---------|
| `rmv`, `onnxruntime` | `validate`, `classify` | `uv sync --extra dev` |
| `torch` | `rmv train` only | `uv sync --extra train` |
| `gnuradio` | Capturing IQ in GNU Radio only | OS packages; **not** a pip dependency of rmv |

If `python3 -c "from rmv import ..."` fails but `uv run rmv --help` works, you are
using the wrong Python. The project venv uses `numpy<2` so system GNU Radio binaries
(from `--system-site-packages`) load correctly; run `uv sync` after pulling changes.
If GNU Radio fails with `No module named gnuradio.analog.analog_python`,
reinstall GNU Radio for your distro or ignore it when you only run `rmv validate`.
If import fails with `numpy.core.multiarray failed to import`, reinstall deps with
`uv sync` (NumPy 2.x in the venv is incompatible with typical distro GNU Radio builds).

```bash
uv run rmv --help
uv run python -c "from rmv import RadioModulationValidator; print('RMV OK')"
```

## Python API

```python
from pathlib import Path
from rmv import RadioModulationValidator

validator = RadioModulationValidator(models_dir=Path("models"))
result = validator.validate_file(Path("iq_samples/gr-qradiolink/mod_nbfm.iq"))
print(result.family_pass, result.order_pass)
```

## CLI

| Command | Description |
|---------|-------------|
| `rmv validate <path>` | Validate `.iq` + sidecar JSON |
| `rmv classify <file>` | Classify without expected labels |
| `rmv plugins list` | List custom-mode validators (composite modulations) |
| `rmv plugins describe <mode_id>` | Show plugin measurements and pass criteria |
| `rmv train` | Train family/order ResidualCNN models |
| `rmv export` | Export checkpoint to ONNX |
| `rmv report <dir>` | Summary from validation JSON |
| `rmv checksum verify` | Verify ONNX SHA-256 |
| `rmv checksum update` | Refresh checksums.sha256 |
| `rmv dataset download` | Download training datasets |
| `rmv dataset status` | Show dataset presence and checksums |
| `rmv dataset verify` | Verify dataset file checksums |
| `rmv dataset generate-synthetic` | Generate synthetic FM/AM/PSK/4FSK training IQ (12 modes by default) |
| `rmv scan run <dir>` | Discover OOT projects, validate modulators (prompts before writes) |
| `rmv scan run <dir> --yes-report` | Write `VALIDATION_REPORT.md` without interactive prompt |
| `rmv scan run <dir>` | Default: only `gr-qradiolink`, `gr-packet-protocols`, `gr-sleipnir` |
| `rmv scan run <dir> --filter a,b` | Override default include list |
| `rmv scan run <dir> --all` | Scan every discovered OOT (minus built-in excludes) |
| `rmv scan issues` | List findings from local `.rmv_findings.db` |
| `rmv scan issues --since 2026-05-31T15:00:00` | Only issues from the current (or recent) scan run |
| `rmv scan purge --keep-latest` | Delete stale validations/issues; keep latest per block (prompts) |
| `rmv scan status` | Database summary |

After training the family classifier, verify before ONNX export:

```bash
uv run rmv verify-family --checkpoint-dir checkpoints
```

This tests RadioML only for AM/QAM/FSK/PAM (valid 128-sample upsamples). **WBFM, BPSK, and QPSK are verified via synthetic IQ** — the same source used in training. Do not use a manual script that classifies RadioML `WBFM`/`BPSK` pickle rows; those windows lack FM/PSK structure after upsampling and will predict AM even when the model is correct.

Scan IQ sidecars use exact classifier class names (for example `NBFM_25`, not `NBFM`),
loaded from `models/*.meta.json` or `checkpoints/best_*_meta.json` at startup.
Reference IQ uses the same `dataset.synthetic` and scan generator paths as training (numpy,
GNU Radio built-ins only — never OOT blocks under test).

Generic scan FSK modes (`2FSK`, `4FSK`, `8FSK`, `P25`) expect order **CPFSK** (RadioML
label). Protocol modes **DMR**, **M17**, **YSF**, **NXDN**, and **dPMR** expect their exact
trained order strings. **GMSK** accepts **MSK** (and vice versa) as an alias at validation time.

Many OOT repos ship two branches: **`main`** (GNU Radio 3.10, `find_package(gnuradio)`) and
**`gnuradio4`** (GNU Radio 4.x). `rmv scan run` validates the **checked-out tree** only; commit
`VALIDATION_REPORT.md` on each branch you care about (reference IQ is numpy/built-in, not built
from the OOT blocks, so results are usually the same per project).

`rmv scan` skips by default (no prompts): GNU Radio framework trees (`gnuradio`,
`gnuradio4`, `gnuradio3`); cryptography OOT (`gr-nacl`, `gr-openssl`,
`gr-linux-crypto`); audio codecs (`gr-opus`); spread-spectrum reference
(`GR-K-GDSS`); and README modes DSSS/GDSS (noise-like — classifier not meaningful).
Add more names under `[scan] exclude_projects` in `.rmv_config.toml`.

Logs go to **stderr**; JSON results to **stdout** for CI pipelines.

## Usage examples

### Validate one GNU Radio modulator (standard CNN)

Place `mod_nbfm.iq` and `mod_nbfm.json` side by side (see [iq_samples/README.md](iq_samples/README.md)):

```json
{
  "source": "gr-qradiolink",
  "block_name": "mod_nbfm",
  "expected_family": "FM",
  "expected_order": "NBFM_25",
  "sample_rate_hz": 48000,
  "center_freq_hz": 0,
  "snr_db": null,
  "notes": "Captured from GNU Radio flowgraph"
}
```

```bash
rmv validate iq_samples/gr-qradiolink/mod_nbfm.iq
# Exit codes: 0 pass, 1 soft fail, 2 hard fail
# JSON on stdout; progress on stderr

rmv validate iq_samples/gr-qradiolink/ --verbose
rmv validate iq_samples/ --repo gr-qradiolink --output results.json
```

### Classify without a sidecar

Use when you only want predictions, not pass/fail against expected labels:

```bash
rmv classify capture.iq
rmv classify capture.iq --format json
```

### Validate composite 8-carrier QPSK (custom plugin)

For one wideband `.iq` with eight parallel QPSK carriers (no per-carrier splits), use
`expected_family: "custom"` and `expected_order: "sleipnir_8qpsk"`:

```bash
rmv plugins list
rmv plugins describe sleipnir_8qpsk
rmv validate iq_samples/gr-sleipnir/tx_output.iq
```

Stdout JSON includes a `custom_mode` block with carrier count, spacing, symbol-rate
estimates, and `pass_overall`.

### Prepare datasets and retrain

```bash
uv sync --extra train

rmv dataset download
rmv dataset verify
rmv dataset status

# RadioML + CSPB only (HISARMOD optional)
rmv train \
  --radioml datasets/radioml/RML2016.10a_dict.pkl \
  --cspb datasets/cspb/ \
  --cache .cache \
  --output checkpoints/ \
  --epochs 50

rmv export --checkpoint checkpoints/best_family_classifier.pt
rmv export --checkpoint checkpoints/best_order_classifier.pt
rmv checksum update
rmv checksum verify
```

### Report and checksums

```bash
rmv report validation_results/ --format markdown
rmv checksum verify --models-dir models
```

### Python API

```python
from pathlib import Path
from rmv import RadioModulationValidator

validator = RadioModulationValidator(models_dir=Path("models"))

# Validate with sidecar
result = validator.validate_file(Path("iq_samples/gr-qradiolink/mod_qpsk.iq"))
print(result.family_pass, result.order_pass)
if result.custom_mode:
    print(result.custom_mode["pass_overall"], result.custom_mode["metrics"])

# Classify only
agg = validator.classify_file(Path("capture.iq"))
print(agg.family, agg.order, agg.family_confidence)

# Batch directory
results = validator.validate_directory(Path("iq_samples/gr-qradiolink"))
```

## Training vs IQ inference

`rmv train` and contributed `.iq` files serve different purposes:

| Path | Input | Purpose |
|------|--------|---------|
| **Training** | RadioML pickle, HISARMOD HDF5, CSPB directory (`.tim` + truth file) | Fit or refresh family/order classifiers |
| **Inference** | `.iq` (+ `.json` sidecar for `validate`) | Check captures against **existing** ONNX models |

Training does **not** accept folders of `.iq` files and does **not** add new modulation
labels. Retraining only improves detection of the standard families and orders listed
in the public datasets (BPSK, QPSK, 2FSK, NBFM, and so on).

Contributed IQ under `iq_samples/` is for **validation** and **classification** after
models exist, not for fitting weights. See [iq_samples/README.md](iq_samples/README.md)
and [docs/contributing_iq_samples.md](docs/contributing_iq_samples.md).

## Custom and multi-carrier modes (IQ)

The classifiers output **standard** `expected_family` / `expected_order` strings (for
example `PSK` and `QPSK`). They do **not** recognize custom air interfaces or
link-layer modes as named classes.

**Custom composite waveforms** (for example eight parallel QPSK carriers in one
wideband `.iq` file) are **not** handled by the CNN. Use a **custom-mode plugin**
(see [Usage examples](#validate-composite-8-carrier-qpsk-custom-plugin)) or per-carrier
IQ files with standard family/order labels.

- The CNN has no label such as `8xQPSK`; `rmv classify` on combined IQ may only guess
  generic **PSK** / **QPSK**, which is unreliable.
- Standard `rmv validate` with `expected_family: "PSK"` does not check carrier count,
  spacing, or baud rate.

For composite modes with a registered plugin, set `expected_family` to `"custom"` and
`expected_order` to the plugin `mode_id` (for example `sleipnir_8qpsk`). See
[docs/contributing_plugins.md](docs/contributing_plugins.md) to add plugins.

For other multi-carrier layouts without a plugin, use **per-carrier** `.iq` files and
sidecars. See **Known limitations** below.

## Contributing IQ samples

Contributed captures live in `iq_samples/` as `.iq` (interleaved float32) plus a
`.json` sidecar. See [iq_samples/README.md](iq_samples/README.md) and
[docs/contributing_iq_samples.md](docs/contributing_iq_samples.md).

## Training datasets

Training combines up to **three public datasets** plus **optional synthetic** IQ.
See [Modulation coverage, training data, and synthesis](#modulation-coverage-training-data-and-synthesis)
for class lists and how synthetic data is built.

- **RadioML 2016.10A** — Primary source for FM, FSK, PSK, QAM, and AM families.
- **HISARMOD 2019.1** — Extra FSK orders and fading channels.
- **CSPB.ML.2018R2** — Independent generator; reduces overfitting to RadioML-style synthesis.
- **Synthetic** — twelve orders (`NBFM_25`, `NBFM_50`, `AM_AIR_25K`, `AM_AIR_833`, `WBFM`,
  `BPSK`, `QPSK`, `DMR`, `M17`, `YSF`, `NXDN`, `dPMR`) via `rmv dataset generate-synthetic`.

You do not need all sources to run inference (`rmv validate`); they are for retraining only.
See [docs/dataset_preparation.md](docs/dataset_preparation.md) for download steps.

## Retraining models

1. Download datasets automatically:

```bash
rmv dataset download
rmv dataset verify
```

See [docs/dataset_preparation.md](docs/dataset_preparation.md) for manual fallback steps.

2. Optional: generate synthetic data (see
   [How synthetic data is created](#how-synthetic-data-is-created)):

```bash
uv run rmv dataset generate-synthetic --output datasets/synthetic/
```

3. Train:

```bash
uv sync --extra train
uv run rmv train \
  --radioml data/radioml/RML2016.10a_dict.pkl \
  --hisarmod data/hisarmod/HisarMod2019.1.h5 \
  --cspb data/cspb/ \
  --synthetic datasets/synthetic/ \
  --cache .cache/ \
  --output checkpoints/
```

Order-only retrain (e.g. after adding synthetic aviation AM, family model unchanged):

```bash
uv run rmv train --order-only --synthetic datasets/synthetic/ ...
```

4. Export and update checksums:

```bash
uv run rmv export --checkpoint checkpoints/best_family_classifier.pt
uv run rmv export --checkpoint checkpoints/best_order_classifier.pt
uv run rmv checksum update
```

## Dataset version pinning (released models)

| Dataset | Pinned version |
|---------|----------------|
| RadioML | 2016.10A |
| HISARMOD | 2019.1 |
| CSPB | **ML.2018R2 only** (not original 2018) |

Placeholder: first release models trained on the versions above with 50 epochs,
batch size 512, AdamW lr=1e-3.

## Validation methodology

Full rules, order aliases, and accepted scan soft fails are documented in
[docs/validation_methodology.md](docs/validation_methodology.md). Summary:

1. **Family and order pass** — predicted labels must match the sidecar
   `expected_family` / `expected_order` (after aliases such as GMSK↔MSK and
   CPFSK↔GFSK). Confidence above the scan threshold is **not** required for pass.
2. **Warnings** — correct label with low confidence may still pass; scan may record a
   warning issue.
3. **Aggregation** — per-chunk predictions are combined by **majority vote** per file.
4. **Hard fail (exit 2)** — wrong family, or family confidence below **0.40**.
5. **Soft fail (exit 1)** — family or order label mismatch (after aliases).

`rmv validate` exit codes: 0 pass, 1 soft fail, 2 hard fail. JSON outputs use
`"schema_version": "1.0"`.

## Known limitations

- **Custom modes from one wideband `.iq` file:** Cannot detect a bespoke multi-carrier
  or other custom mode as its own class; only generic family/order labels apply, with
  low trust on non-isolated carriers. See **Custom and multi-carrier modes (IQ)** above.
- **gr-sleipnir multi-carrier:** Validate each carrier with a separate IQ file and
  sidecar; a single wideband capture may not match one order label.
- **Similar orders within family** (e.g. WBFM vs NBFM) may confuse the order
  classifier when SNR is low or capture is short.
- **Documented scan soft fails** (not pipeline bugs): NXDN↔dPMR (identical synthetic
  params), SSB↔WBFM on baseband reference, 8PSK↔QPSK when order resolution is limited.
  See [docs/validation_methodology.md](docs/validation_methodology.md).
- **CSPB original dataset** must not be used for training (RNG flaw); use R2 only.
- **No IQ-based training:** Cannot train on contributed `.iq` captures or define new
  classes without extending the codebase and supplying a labelled dataset.

## Development

```bash
uv sync --extra dev --extra train
uv run python scripts/check_env.py
uv run pytest
uv run ruff check src tests
uv run mypy src/rmv
```

## License

See [LICENSE](LICENSE).
