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
  - [Synthetic orders (4) — bandwidths verified at generation](#synthetic-orders-4--bandwidths-verified-at-generation)
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

#### Synthetic orders (4) — bandwidths verified at generation

Generated by `rmv dataset generate-synthetic` (`src/rmv/dataset/synthetic.py`).
Occupied bandwidth is checked at **−26 dB** from the PSD peak on clean (no AWGN) reference
chunks; limits below are enforced before data is saved.

| Order | Family | Peak deviation / modulation index | Audio passband (Hz) | Channel spacing | Carson / occupied BW (typical) | Max occupied BW (verify) |
|-------|--------|-----------------------------------|---------------------|-------------------|-------------------------------|---------------------------|
| `NBFM_25` | FM | ±2.5 kHz (`max_dev=2500`) | 300–800 (voice-like mix) | 12.5 kHz (IARU Region 1 amateur) | ~6 kHz (Carson: 2×(Δf+f_m)) | ≤7 kHz |
| `NBFM_50` | FM | ±5.0 kHz (`max_dev=5000`) | 300–1500 | 25 kHz (older amateur repeaters) | ~11 kHz | ≤13 kHz |
| `AM_AIR_25K` | AM | m=0.85 (DSB, full carrier) | 300–3400 (ICAO Annex 10 voice) | 25 kHz (worldwide airband) | ~7 kHz occupied | ≤8 kHz |
| `AM_AIR_833` | AM | m=0.85 | 300–2500 (EU 8.33 kHz channel) | 8.33 kHz (EU Reg. 1079/2012, FL195+) | ~5–6 kHz | ≤6.5 kHz |

NBFM uses **no preemphasis** (`tau=0`); broadcast FM `tau=75 µs` is rejected. Aviation AM is
**not** RadioML `AM-DSB` (no broadcast preemphasis or ~20 kHz audio).

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

RadioML and CSPB do **not** include amateur narrowband FM or aviation airband AM as
distinct orders; synthetic data fills that gap. Broadcast `AM-DSB` must not be used as
a stand-in for `AM_AIR_25K` / `AM_AIR_833`.

### Where training data comes from

| Source | Format | Role |
|--------|--------|------|
| [RadioML 2016.10A](https://www.deepsig.ai/datasets) | `RML2016.10a_dict.pkl` | Primary families (FM/FSK/PSK/QAM/AM); GNU Radio–style synthetic IQ |
| [HISARMOD 2019.1](https://github.com/kit-cel/HisarMod2019.1) | HDF5 (`.h5`) | Extra FSK orders and fading channels |
| [CSPB.ML.2018R2](https://cspb.ml/) | `.tim` batches + truth file | Independent generator; cross-check against RadioML bias |
| **Synthetic** (this repo) | `datasets/synthetic/synthetic.npz` | NBFM + aviation AM; generated locally, not downloaded |

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

See the [Synthetic orders](#synthetic-orders-4--bandwidths-verified-at-generation) table
above for deviation, audio passband, channel spacing, and occupied-bandwidth limits.
Implementation summary:

| Class | Implementation |
|-------|----------------|
| `NBFM_25` / `NBFM_50` | Audio 8 kHz → interp to 48 kHz → GNU Radio `frequency_modulator_fc`; **no** preemphasis (`tau=0`) |
| `AM_AIR_25K` / `AM_AIR_833` | NumPy/scipy DSB-AM full carrier: `s(t)=1+m·a(t)` at baseband |

Per chunk: random voice-band audio (tones + band-limited noise), optional AWGN
(−20 to +30 dB in 2 dB steps), random ±200 Hz frequency offset, unit-power
normalisation, shape `(2, 1024)` float32. Default run: 1000 chunks × 26 SNR levels ×
4 classes ≈ 104k samples. Bandwidth is checked on clean reference chunks before
noisy data is kept.

Upstream `analog.nbfm_tx(tau=0)` fails in GNU Radio (preemphasis divide-by-zero); the
synthetic path uses the equivalent **interp + `frequency_modulator_fc`** without
preemphasis, matching amateur NFM intent.

```bash
uv run rmv dataset generate-synthetic --output datasets/synthetic/
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
| `rmv dataset generate-synthetic` | Generate NBFM / aviation AM training IQ (not in RadioML/CSPB) |

Logs go to **stderr**; JSON results to **stdout** for CI pipelines.

## Usage examples

### Validate one GNU Radio modulator (standard CNN)

Place `mod_nbfm.iq` and `mod_nbfm.json` side by side (see [iq_samples/README.md](iq_samples/README.md)):

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
- **Synthetic** — `NBFM_25`, `NBFM_50`, `AM_AIR_25K`, `AM_AIR_833` via `rmv dataset generate-synthetic`.

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

1. **Family check** — coarse modulation family must match sidecar `expected_family`
   with confidence >= threshold (default **0.70**).
2. **Order check** — specific mode must match `expected_order` with confidence >=
   threshold.
3. **Aggregation** — per-chunk predictions are combined by **majority vote** per file.
4. **Hard fail (exit 2)** — wrong family entirely, or majority confidence **< 0.40**
   (ambiguous or broken signal).
5. **Soft fail (exit 1)** — family or order mismatch below hard-fail severity.

All JSON outputs include `"schema_version": "1.0"`.

## Known limitations

- **Custom modes from one wideband `.iq` file:** Cannot detect a bespoke multi-carrier
  or other custom mode as its own class; only generic family/order labels apply, with
  low trust on non-isolated carriers. See **Custom and multi-carrier modes (IQ)** above.
- **gr-sleipnir multi-carrier:** Validate each carrier with a separate IQ file and
  sidecar; a single wideband capture may not match one order label.
- **Similar orders within family** (e.g. WBFM vs NBFM) may confuse the order
  classifier when SNR is low or capture is short.
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
