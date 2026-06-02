# radio-modulation-validator

Validate AI-generated GNU Radio out-of-tree (OOT) modulator blocks by classifying
IQ samples against known modulation types. The tool checks that modulator blocks
produce waveforms matching expected modulation **families** (FM, FSK, PSK, QAM, AM,
PAM) and **orders** (NBFM, QPSK, 2FSK, etc.) using ONNX classifiers trained on
public RF ML datasets.

## What is an IQ file?

An **IQ file** stores a chunk of **baseband** radio signal as a sequence of complex
samples. Each sample has an **I** (in-phase) and **Q** (quadrature) component; together
they describe amplitude and phase of the signal at a given **sample rate** (for example
48 kHz). IQ captures are what SDR receivers and GNU Radio flowgraphs work with before
demodulation or decoding.

**GNU Radio** can write IQ to disk and read it back: use blocks such as **File Sink**
and **File Source** (or SigMF equivalents) in a flowgraph to record modulator output or
feed recorded data into a chain. This project expects simple binary `.iq` files:
interleaved **float32** I/Q pairs (little-endian), often 1024 complex samples per chunk.
See [Contributing IQ samples](#contributing-iq-samples) and [iq_samples/README.md](iq_samples/README.md)
for the exact layout and sidecar JSON.

## Table of contents

- [What is an IQ file?](#what-is-an-iq-file)
- [Modulation coverage, training data, and synthesis](#modulation-coverage-training-data-and-synthesis)
  - [Complete order and bandwidth reference](#complete-order-and-bandwidth-reference)
  - [Families (6)](#families-6)
  - [RadioML 2016.10A (11 orders)](#radioml-201610a-11-orders)
  - [HISARMOD 2019.1 (26 orders)](#hisarmod-20191-26-orders)
  - [CSPB.ML.2018R2 (8 orders)](#cspbml2018r2-8-orders)
  - [Synthetic orders (19) — bandwidths verified at generation](#synthetic-orders-19--bandwidths-verified-at-generation)
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
  - [Why synthetic training data is used](#why-synthetic-training-data-is-used)
- [Retraining models](#retraining-models)
- [Extending rmv](#extending-rmv)
  - [Paths and configuration](#paths-and-configuration-no-hardcoded-paths)
  - [Larger training sets](#larger-training-sets)
  - [Add a new order mode (CNN)](#add-a-new-order-mode-cnn)
  - [Write a custom-mode plugin](#write-a-custom-mode-plugin)
  - [Add an OOT project to scan](#add-an-oot-project-to-scan)
- [Dataset version pinning (released models)](#dataset-version-pinning-released-models)
- [Validation methodology](#validation-methodology)
- [Understanding validation results](#understanding-validation-results)
- [Why the classifier needs training data](#why-the-classifier-needs-training-data)
- [Soft fails vs hard fails](#soft-fails-vs-hard-fails)
- [The validation boundary](#the-validation-boundary)
- [NPU deployment (INT8)](#npu-deployment-int8)
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

#### Synthetic orders (19) — bandwidths verified at generation

Generated by `rmv dataset generate-synthetic` (`src/rmv/dataset/synthetic.py`).
Occupied bandwidth is checked at **−26 dB** from the PSD peak on **clean** reference chunks
(no AWGN) before noisy data is saved. Protocol 4FSK modes also check instantaneous frequency
structure (four tones, symbol rate).

**FM, AM, and PSK (9 orders)**

| Order | Family | Peak deviation / modulation index | Audio passband (Hz) | Channel spacing | Carson / occupied BW (typical) | Max occupied BW (verify) |
|-------|--------|-----------------------------------|---------------------|-------------------|-------------------------------|---------------------------|
| `NBFM_25` | FM | ±2.5 kHz (`max_dev=2500`) | 300–800 (voice-like mix) | 12.5 kHz (IARU Region 1 amateur) | ~6 kHz (Carson: 2×(Δf+f_m)) | ≤7 kHz |
| `NBFM_50` | FM | ±5.0 kHz (`max_dev=5000`) | 300–1500 | 25 kHz (older amateur repeaters) | ~11 kHz | ≤13 kHz |
| `NFM_CTCSS` | FM | ±2.5 kHz (voice ±2 kHz + CTCSS ±0.5 kHz) | 300–3000 + subaudible 67–254 Hz | 12.5 kHz | ~6 kHz | ≤7 kHz |
| `NFM_DCS` | FM | ±2.5 kHz (voice + DCS ±134 Hz FSK) | 300–3000 + DCS 134.4 bit/s | 12.5 kHz | ~6 kHz | ≤7 kHz |
| `AM_AIR_25K` | AM | m=0.85 (DSB, full carrier) | 300–3400 (ICAO Annex 10 voice) | 25 kHz (worldwide airband) | ~7 kHz occupied | ≤8 kHz |
| `AM_AIR_833` | AM | m=0.85 | 300–2500 (EU 8.33 kHz channel) | 8.33 kHz (EU Reg. 1079/2012, FL195+) | ~5–6 kHz | ≤6.5 kHz |
| `WBFM` | FM | ±75 kHz (`max_dev=75000`) | ~50 Hz–15 kHz (broadcast-like) | — | ~200 kHz occupied | ≤200 kHz |
| `BPSK` | PSK | 8 samples/symbol @ 48 kHz | — | — | Digital; GNU Radio `psk_mod` | dataset verify |
| `QPSK` | PSK | 8 samples/symbol @ 48 kHz | — | — | Digital; GNU Radio `psk_mod` | dataset verify |

NBFM uses **no preemphasis** (`tau=0`); broadcast FM uses `tau=75 µs` for `WBFM` only.
Aviation AM is **not** RadioML `AM-DSB` (no broadcast preemphasis or ~20 kHz audio).
`WBFM`, `BPSK`, and `QPSK` replace weak or upsample-broken RadioML windows used only after
128→1024 upsampling.

**Protocol 4FSK (6 orders)** — modulation layer only (no framing, sync, or vocoders):

| Order | Family | Symbol rate | Tone deviations | Pulse shaping |
|-------|--------|-------------|-----------------|---------------|
| `DMR` | FSK | 4800 baud | ±648 / ±1944 Hz (4-ary) | Raised cosine, α=0.2 |
| `M17` | FSK | 4800 baud | ±800 / ±2400 Hz | Root raised cosine, β=0.5 |
| `YSF` | FSK | 4800 baud | ±800 / ±2400 Hz (C4FM map) | Gaussian, BT=0.5 |
| `P25` | FSK | 4800 baud | ±600 / ±1800 Hz (C4FM) | Raised cosine, α=0.2 |
| `NXDN` | FSK | 2400 baud | ±350 / ±1050 Hz | Raised cosine, α=0.2 |
| `dPMR` | FSK | 2400 baud | ±350 / ±1050 Hz | Raised cosine, α=0.2 |

**Packet radio physical layers (2 orders)** — AX.25, APRS, FX.25, and IL2P map to `BELL202`:

| Order | Family | Description |
|-------|--------|-------------|
| `BELL202` | FSK | Bell 202 AFSK (1200/2200 Hz, 1200 baud NRZI) through FM |
| `G3RUH` | FSK | 9600 baud direct 2FSK, ±3500 Hz, Gaussian BT=0.5 |

**GMSK (2 orders)** — GNU Radio `digital.gmsk_mod`, 4800 baud, 10 sps @ 48 kHz, h=0.5
(±1200 Hz peak deviation):

| Order | Family | BT | Role |
|-------|--------|-----|------|
| `GMSK_BT05` | FSK | 0.5 | D-Star profile; scan reference for `D-Star` / `DSTAR` |
| `GMSK_BT03` | FSK | 0.3 | Standard GMSK; scan reference for generic `GMSK` mode |

CSPB provides **MSK** only (no Gaussian BT product). These synthetic classes separate
continuous-phase GMSK from **NXDN** / **dPMR** 4FSK at training time.

At validation, **GMSK**, **GMSK_BT05**, **GMSK_BT03**, and **MSK** are one **order alias**
group (sidecars may still say `GMSK` for `mod_gmsk` / `mod_d_star`). **NXDN** is not aliased
to **GMSK**. **NXDN** and **dPMR** share identical synthetic
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
| **Synthetic** (this repo) | `datasets/synthetic/synthetic.npz` | NBFM, squelch, aviation AM, WBFM, PSK, GMSK (BT=0.5/0.3), protocol 4FSK/P25, packet radio |

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

See the [Synthetic orders](#synthetic-orders-19--bandwidths-verified-at-generation) tables
above for deviation, audio passband, channel spacing, and occupied-bandwidth limits.
Implementation summary:

| Class | Implementation |
|-------|----------------|
| `NBFM_25` / `NBFM_50` | Audio 8 kHz → interp to 48 kHz → GNU Radio `frequency_modulator_fc`; **no** preemphasis (`tau=0`) |
| `AM_AIR_25K` / `AM_AIR_833` | NumPy/scipy DSB-AM full carrier: `s(t)=1+m·a(t)` at baseband |
| `WBFM` | Band-limited audio → interp → `frequency_modulator_fc` with broadcast deviation (±75 kHz) |
| `BPSK` / `QPSK` | GNU Radio `digital.psk_mod`, 8 samples/symbol @ 48 kHz |
| `DMR` / `M17` / `YSF` / `P25` / `NXDN` / `dPMR` | NumPy/scipy 4FSK: symbol map, deviation, RC / RRC / Gaussian shaping (`verify_4fsk_signal`) |
| `NFM_CTCSS` / `NFM_DCS` | NFM with subaudible tone or DCS FSK mixed pre-modulator |
| `BELL202` | Bell 202 AFSK (continuous-phase) → FM; `AX.25` / `IL2P` / `FX.25` scan as `BELL202` |
| `G3RUH` | 9600 baud 2FSK with Gaussian BT=0.5 |
| `GMSK_BT05` / `GMSK_BT03` | GNU Radio `digital.gmsk_mod`, BT=0.5 / 0.3 |

Per chunk: random excitation (voice-band audio, PSK symbols, or 4FSK symbols), optional AWGN
(−20 to +30 dB in 2 dB steps), random ±200 Hz frequency offset, unit-power normalisation,
shape `(2, 1024)` float32. Default run: 1000 chunks × 26 SNR levels × **19** classes ≈
**494k** samples. Bandwidth and 4FSK structure checks use clean reference chunks before
noisy data is kept.

Upstream `analog.nbfm_tx(tau=0)` fails in GNU Radio (preemphasis divide-by-zero); the
synthetic path uses the equivalent **interp + `frequency_modulator_fc`** without
preemphasis, matching amateur NFM intent.

```bash
# All 19 modes (default)
uv run rmv dataset generate-synthetic --output datasets/synthetic/

# Subset examples
uv run rmv dataset generate-synthetic --modes gmsk_bt05,gmsk_bt03 --output datasets/synthetic/
uv run rmv dataset generate-synthetic --modes dmr,m17,ysf,nxdn,dpmr,p25 --output datasets/synthetic/
```

## Quick start

```bash
# Install with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev

# Pre-trained ONNX models are in models/ (see models/README.md)
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
| `rmv export-quantised` | Quantise FP32 ONNX to INT8 (`--family-only` / `--order-only`) |
| `rmv export-npu` | Convert INT8 ONNX to SpacemiT `.nb` (SDK optional) |
| `rmv report <dir>` | Summary from validation JSON |
| `rmv checksum verify` | Verify ONNX SHA-256 |
| `rmv checksum update` | Refresh checksums.sha256 |
| `rmv dataset download` | Download training datasets |
| `rmv dataset status` | Show dataset presence and checksums |
| `rmv dataset verify` | Verify dataset file checksums |
| `rmv dataset generate-synthetic` | Generate synthetic FM/AM/PSK/4FSK/GMSK/squelch/packet IQ (19 modes) |
| `rmv scan run [dir]` | Discover OOT projects (dir defaults to parent of rmv repo, e.g. `..`) |
| `rmv scan run .. --yes --yes-report` | Typical layout: rmv inside `github-projects` |
| `rmv scan run <dir> --yes-report` | Write `VALIDATION_REPORT.md` without interactive prompt |
| `rmv scan run` | Default include: `gr-qradiolink`, `gr-packet-protocols`, `gr-sleipnir` |
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

Generic scan FSK modes (`2FSK`, `4FSK`, `8FSK`) expect order **CPFSK** (RadioML label).
Protocol modes **DMR**, **M17**, **YSF**, **NXDN**, **dPMR**, and **P25** expect their exact
trained order strings (`P25` uses synthetic C4FM, not CPFSK).

**GMSK family (scan and validate):** reference IQ for mode `GMSK` is synthetic **GMSK_BT03**;
**D-Star** / **DSTAR** use **GMSK_BT05**. OOT sidecars often list `expected_order: GMSK`; after
retraining, predictions of **GMSK_BT05**, **GMSK_BT03**, or **MSK** also pass via aliases.
Packet link modes (**AX.25**, **APRS**, **FX.25**, **IL2P**) validate at **BELL202** physical layer only.

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
- **Synthetic** — nineteen orders (NBFM, aviation AM, WBFM, PSK, GMSK BT=0.5/0.3, protocol
  4FSK, squelch tones, P25, packet radio) via `rmv dataset generate-synthetic`; see
  [Synthetic orders (19)](#synthetic-orders-19--bandwidths-verified-at-generation).

### Why synthetic training data is used

The three public datasets (RadioML 2016.10A, HISARMOD 2019.1, CSPB.ML.2018R2) cover standard
modulation families well but have two specific gaps that matter for amateur radio and repeater
validation:

**Gap 1 — Short sample windows in RadioML**

RadioML 2016.10A stores each signal as 128 complex samples. At 48 kHz this is only 2.7 ms of
signal — too short to show meaningful modulation structure for FM and PSK modes. After
upsampling to the required 1024 samples the interpolated signal loses its characteristic
features. Specifically:

- **WBFM** at 128 samples looks like band-limited noise after upsampling — the classifier
  cannot distinguish it from AM
- **BPSK** and **QPSK** at 128 samples contain only a handful of symbols — not enough to
  show clear constellation structure

Rather than excluding FM and PSK entirely, synthetic data generated from GNU Radio built-in
blocks (`frequency_modulator_fc` for NBFM/WBFM, `digital.psk_mod` for PSK) replaces those
RadioML entries with correctly structured 1024-sample signals. The synthetic signals are
verified against bandwidth and constellation specifications before use.

**Gap 2 — Modes absent from all public datasets**

Several modulation modes common in amateur radio repeater operation do not appear in
RadioML, CSPB, or HISARMOD:

| Mode | Why absent | How covered |
|------|------------|-------------|
| `NBFM_25` / `NBFM_50` | RadioML has only WBFM (broadcast FM) | GNU Radio NFM, no preemphasis, max_dev 2500/5000 Hz |
| `AM_AIR_25K` / `AM_AIR_833` | No aviation AM in any public dataset | NumPy DSB-AM, ICAO Annex 10 parameters |
| `DMR`, `M17`, `YSF`, `NXDN`, `dPMR` | Amateur digital voice not in public datasets | NumPy 4FSK per published standards |
| `NFM_CTCSS` / `NFM_DCS` | Squelch tone modes not in any dataset | NFM + scipy subaudible tone / DCS FSK |
| `P25` | Public safety digital not in public datasets | NumPy C4FM per TIA-102.BAAA |
| `BELL202` / `G3RUH` | Packet radio physical layers not in datasets | NumPy AFSK/FSK per AX.25 / G3RUH specs |
| `GMSK_BT05` / `GMSK_BT03` | CSPB has MSK only; no BT=0.5/0.3 GMSK | GNU Radio `digital.gmsk_mod` |

**Gap 3 — GMSK confused with NXDN (training)**

HISARMOD/CSPB include **GMSK** or **MSK**, but not Gaussian-filtered GMSK at 4800 baud with
the BT products D-Star and general narrowband digital use. Without **GMSK_BT05** / **GMSK_BT03**
synthetic data, the order classifier may label GMSK-layer captures as **NXDN** (another 4FSK family
order). Retrain with the new synthetic classes; validation aliases are already in place.

**The validation boundary**

All synthetic data is generated using GNU Radio built-in blocks and numpy/scipy only. The OOT
modules under validation (`gr-qradiolink`, `gr-packet-protocols`, `gr-sleipnir`) are never
used to generate training data — doing so would make the validation circular. This boundary is
enforced by an AST-based test (`test_no_oot_imports`) that runs in CI.

The public datasets provide the broad modulation taxonomy and cross-validation from
independent toolchains. The synthetic data fills the specific gaps needed for repeater and
amateur radio validation. Both are needed; neither alone is sufficient.

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
uv run rmv dataset generate-synthetic \
  --modes nbfm25,nbfm50,am_air_25k,am_air_833,wbfm,bpsk,qpsk,\
dmr,m17,ysf,nxdn,dpmr,nfm_ctcss,nfm_dcs,p25,bell202,g3ruh,gmsk_bt05,gmsk_bt03 \
  --output datasets/synthetic/
```

3. Train (clear preprocess cache after new synthetic classes):

```bash
uv sync --extra train
rm -rf .cache/
uv run rmv train \
  --radioml datasets/radioml/RML2016.10a_dict.pkl \
  --hisarmod datasets/hisarmod/HisarMod2019.1.h5 \
  --cspb datasets/cspb/ \
  --synthetic datasets/synthetic/synthetic.npz \
  --cache .cache \
  --output checkpoints \
  --device cuda \
  -y
```

Order-only retrain (e.g. new GMSK or aviation AM classes, family model unchanged):

```bash
uv run rmv train \
  --order-only \
  --synthetic datasets/synthetic/synthetic.npz \
  --datasets-dir datasets \
  --cache .cache \
  --output checkpoints \
  --device cuda \
  -y
```

After retraining, export ONNX, run `uv run rmv checksum update`, and re-run `rmv scan` on OOT
trees. Expected improvements for GMSK-layer blocks (aliases apply immediately; the model must
learn **GMSK_BT05** / **GMSK_BT03** labels):

| Block (typical) | Before retrain | After retrain (target) |
|-----------------|----------------|-------------------------|
| `mod_gmsk` | NXDN soft fail | GMSK / GMSK_BT03 / MSK |
| `mod_d_star` (gr-qradiolink, gr-sleipnir) | NXDN soft fail | GMSK_BT05 |
| `mod_freedv` | may vary | GMSK family |

4. Export and update checksums:

```bash
uv run rmv export --checkpoint checkpoints/best_family_classifier.pt
uv run rmv export --checkpoint checkpoints/best_order_classifier.pt
uv run rmv checksum update
```

## Extending rmv

This section covers growing training data, adding classifier labels, custom plugins, and
new GNU Radio OOT trees to `rmv scan`. All workflows use **CLI flags, environment
variables, or config files** — the Python codebase must not embed machine-specific
absolute paths (enforced in CI; see [Paths and configuration](#paths-and-configuration-no-hardcoded-paths)).

### Paths and configuration (no hardcoded paths)

| What | How to set it |
|------|----------------|
| Training datasets | `rmv dataset download --dest <dir>`; `rmv train --datasets-dir <dir>` |
| Synthetic IQ output | `rmv dataset generate-synthetic --output <dir>` |
| Preprocess cache | `rmv train --cache <dir>` |
| Checkpoints / export | `rmv train --output <dir>`; `rmv export --checkpoint <path>` |
| ONNX models | `rmv validate --models-dir <dir>` (default: `models/` under repo root) |
| Scan root (OOT parent) | `rmv scan run <dir>` or omit to auto-detect parent of this repo |
| Scan IQ output | `rmv scan run <dir> --iq-output <dir>` (default: `.scan_iq/`) |
| Findings database | `.rmv_findings.db` next to `pyproject.toml` (see `rmv scan status`) |

Optional project config at the repository root (`.rmv_config.toml`):

```toml
[scan]
root = "../github-projects"          # optional default scan parent
include_projects = ["gr-my-oot", "gr-qradiolink"]
exclude_projects = ["gnuradio", "gnuradio4"]
iq_output = ".scan_iq"
```

Use **relative paths** from the directory where you run `rmv`, or absolute paths on your
machine in config and shell commands — never commit personal paths into `src/`.
`find_rmv_project_root()` locates `pyproject.toml` so defaults resolve from the clone
location, not from a fixed install path.

### Larger training sets

Public datasets (RadioML, HISARMOD, CSPB R2) provide the bulk of training diversity.
Download and verify them under a directory you choose:

```bash
rmv dataset download --dest datasets/
rmv dataset verify
```

Synthetic data scales with `--chunks-per-snr` (default **1000**) and the built-in
**26 SNR levels** (-20 to +30 dB in 2 dB steps). Approximate synthetic sample count:

`chunks_per_snr × 26 × number_of_modes`

Example — roughly **2.5M** synthetic chunks for all 19 modes at default density:

```bash
uv run rmv dataset generate-synthetic \
  --chunks-per-snr 5000 \
  --output datasets/synthetic/ \
  --seed 42
```

Combine with public data in training (paths are arguments, not hardcoded):

```bash
rm -rf .cache/
uv run rmv train \
  --datasets-dir datasets \
  --synthetic datasets/synthetic/synthetic.npz \
  --cache .cache \
  --output checkpoints \
  --device cuda \
  -y
```

Increase `chunks_per_snr` until validation accuracy on held-out synthetic chunks stabilises.
After changing dataset layout or class list, delete `.cache/` so preprocessing shards rebuild.
See [docs/dataset_preparation.md](docs/dataset_preparation.md).

### Add a new order mode (CNN)

Use this when the modulation should be classified by the **family/order ONNX models**
(not a custom plugin). Work in the repository clone; do not import OOT modules under test
into `src/rmv/dataset/synthetic.py` or `src/rmv/scan/iq_generator.py`.

1. **Generator** — in `src/rmv/dataset/synthetic.py`:
   - Add a `VariantSpec` entry (bandwidth, `kind`, protocol parameters).
   - Implement `generate_*` / wire into `generate_variant_chunks()`.
   - Add CLI key to `MODE_TO_CLASS`, `ALL_MODES`, and bandwidth verify helpers.
   - Add tests in `tests/test_synthetic.py` (structure + occupied BW).

2. **Label maps** — in `src/rmv/constants.py`:
   - Append class name to `SYNTHETIC_CLASSES` and `SYNTHETIC_TO_FAMILY`.
   - `ORDER_CLASSES` updates automatically from dataset unions.

3. **Validation aliases** (if needed) — in `src/rmv/validate.py`:
   - Extend `ORDER_ALIASES` or `SSB_AMBIGUOUS_ORDERS` for known ambiguities.

4. **Scan reference** (if the mode appears in OOT READMEs) — in `src/rmv/scan/mode_table.py`:
   - Add `ModeSpec(mode_name, expected_family, expected_order, generation_method)`.
   - Use `generation_method="synthetic"` and reference IQ via `_gen_synthetic_scan_chunks()`,
     or `numpy` / `gr3_builtin` with built-in GNU Radio only.
   - Add the mode name to `KNOWN_MODE_NAMES` in `src/rmv/scan/readme_parser.py` if README
     parsing should detect it.

5. **Retrain and ship** — generate synthetic NPZ, `rmv train --order-only` (or full train),
   `rmv export`, `rmv checksum update`, copy ONNX + meta into `models/`, re-run scan:

   ```bash
   rm -rf .scan_iq/
   uv run rmv scan run <oot-parent-dir> --yes --yes-report
   ```

CI runs `tests/test_synthetic.py::test_no_oot_imports` to block OOT imports in synthetic
and scan generator code.

### Write a custom-mode plugin

Use a plugin when the waveform is **not** a single standard family/order label (for example
eight parallel QPSK carriers in one wideband capture). The CNN is skipped; a Python plugin
checks structure, spacing, and constellation metrics.

Full guide: [docs/contributing_plugins.md](docs/contributing_plugins.md). Summary:

1. Subclass `CustomModeValidator` in `src/rmv/plugins/base.py` (copy
   `src/rmv/plugins/sleipnir_8qpsk.py` as a template).
2. Register in `src/rmv/plugins/registry.py` (`_register_builtin_plugins()`).
3. Add tests in `tests/test_plugins.py` using small synthetic IQ (no large binaries in git).
4. Add `ModeSpec(..., expected_family="custom", expected_order="<mode_id>", generation_method="plugin")`
   in `src/rmv/scan/mode_table.py` for scan reference IQ generation.
5. Sidecar for validate: `"expected_family": "custom"`, `"expected_order": "<mode_id>"`.

```bash
rmv plugins list
rmv plugins describe sleipnir_8qpsk
rmv validate iq_samples/<repo>/<block>.iq
```

### Add an OOT project to scan

`rmv scan` discovers GNU Radio OOT projects under the directory you pass (sibling folders
named `gr-*` with `CMakeLists.txt` and/or `grc/*.block.yml`). It does not modify OOT
source trees.

1. **Layout** — place the OOT repo next to others (typical: parent folder containing
   `radio-modulation-validator` and `gr-my-oot/`).

2. **README** — list modulation mode names in the project `README.md`. The scanner parses
   known names (see `KNOWN_MODE_NAMES` in `src/rmv/scan/readme_parser.py`). Add new
   names there and a matching `ModeSpec` in `mode_table.py`.

3. **Include in scan** — default scan only runs
   `gr-qradiolink`, `gr-packet-protocols`, `gr-sleipnir`. To scan your project:

   ```bash
   rmv scan run <oot-parent-dir> --filter gr-my-oot --yes --yes-report
   ```

   Or add to `.rmv_config.toml` `include_projects`, or use `--all` to scan every discovered
   OOT (minus framework/crypto exclusions).

4. **Outputs** — reference `.iq` + `.json` under `--iq-output` (default `.scan_iq/<project>/`),
   `VALIDATION_REPORT.md` in the OOT repo (with `--yes-report`), rows in `.rmv_findings.db`.

5. **GNU Radio 3** — optional `--gr3-prefix` for built-in reference generators that use
   GNU Radio; GR4-only trees may skip numpy/builtin modes until a GR4 generator exists.

Exclude framework trees (`gnuradio`, `gnuradio4`) and non-RF modules via
`[scan] exclude_projects` in `.rmv_config.toml` (see [CLI](#cli) scan section).

## Dataset version pinning (released models)

| Dataset | Pinned version |
|---------|----------------|
| RadioML | 2016.10A |
| HISARMOD | 2019.1 |
| CSPB | **ML.2018R2 only** (not original 2018) |

Placeholder: first release models trained on the versions above with 50 epochs,
batch size 512, AdamW lr=1e-3.

## NPU deployment (INT8)

For SpacemiT K3 NPU deployment, quantise shipped FP32 ONNX models to INT8 and optionally
convert to `.nb` binaries. See [docs/npu-deployment.md](docs/npu-deployment.md) and
[models/README.md](models/README.md). Family INT8 is deployed when verification passes;
order stays FP32 (50-class INT8 typically fails tolerance).

```bash
uv run rmv dataset generate-synthetic --output datasets/synthetic/
uv run rmv export-quantised --synthetic datasets/synthetic/synthetic.npz --family-only
uv run rmv export-npu --calibration datasets/synthetic/synthetic.npz   # on K3 or with SDK
```

## Validation methodology

Full rules, order aliases, and scan reference mapping are documented in
[docs/validation_methodology.md](docs/validation_methodology.md). Per-chunk predictions
are combined by **majority vote** per file. `rmv validate` exit codes: **0** pass,
**1** soft fail, **2** hard fail. JSON outputs use `"schema_version": "1.0"`.

The sections below explain what those results mean, what the tool cannot validate,
and how this relates to unit tests and retraining.

## Understanding validation results

The IQ classifier validates at the **physical layer only** — it checks that a
modulator block produces a signal whose IQ characteristics match the expected
modulation family (FM, FSK, PSK, QAM, AM) and order (NBFM_25, DMR, BELL202, etc.).
It does not validate:

- Protocol framing, sync words, or FEC correctness
- Codec output (IMBE, Codec2, AMBE, OPUS)
- Link-layer state machines (AX.25, IL2P, FX.25 frame structure)
- Reed-Solomon or LDPC decoder correctness
- End-to-end encode/decode round-trips

For protocol-level correctness, the project's own unit tests (ctest, pytest) are
the authoritative source. The IQ classifier and unit tests are complementary, not
interchangeable.

**Example:** gr-packet-protocols passes 13/13 unit tests covering Reed-Solomon codec,
AX.25/FX.25/IL2P encoder/decoder round-trips, and HDLC framing correctness. The IQ
validator shows 11/12 with a mod_8psk soft fail — this reflects a classifier training
gap (8PSK is underrepresented in the training data), not a bug in the block. The unit
tests are the correct tool for validating protocol correctness.

## Why the classifier needs training data

The classifier cannot identify a modulation it has not been trained on. Adding a
new modulation order to the training set requires:

1. **Synthetic IQ data generation** — generate labelled IQ samples using verified
   GNU Radio built-in blocks or numpy/scipy. Never use the OOT module under validation
   to generate its own reference data — that makes the validation circular.

2. **Retraining the order classifier** — run `rmv train` with the new synthetic data
   included. The family classifier (6 classes) rarely needs retraining; the order
   classifier (currently 50 classes) is retrained when new modes are added.

3. **Re-export and re-quantise** — export the new ONNX models, regenerate the INT8
   family classifier (`rmv export-quantised --family-only`), update checksums.

4. **Re-run the scan** — delete `.scan_iq/` and re-run `rmv scan run` to pick up the
   new classifier.

Current training data sources and the modes they cover are documented in
[docs/dataset_preparation.md](docs/dataset_preparation.md).

**Current known training gaps** (modes not yet in the order classifier):

| Mode | Gap | Impact |
|------|-----|--------|
| 8PSK at 12,500 baud | Not in training data | gr-packet-protocols mod_8psk soft fail |
| SSTV (modes 210–223) | Not in training data | No SSTV validation possible |
| FreeDV 700D/1600/2020 | OFDM-based, not modelled | FreeDV classifies as generic FSK |
| WFM Stereo (mode 11) | Not in training data | Classifies as WBFM |
| CW / Morse | Short observation window | Not classifiable at 1024 samples |

Adding any of these requires generating synthetic IQ and retraining. See
[docs/contributing_iq_samples.md](docs/contributing_iq_samples.md) for contributed
capture guidelines.

## Soft fails vs hard fails

**Hard fail** (exit code 2): the predicted modulation family is completely wrong —
for example an FSK block predicting AM. This indicates either a genuinely broken
modulator or a systematic error in the scan IQ generator. Investigate before trusting
the block.

**Soft fail** (exit code 1): the family is correct but the order prediction does not
match — for example 8PSK predicting QPSK, or NXDN predicting dPMR. This is almost
always a classifier training gap or a known physical ambiguity (NXDN and dPMR use
identical modulation parameters and cannot be distinguished at the IQ level). Check
the known limitations table and the unit tests before concluding the block is broken.

**Warning**: the prediction is correct but confidence is below 0.70. Common for modes
in the NXDN/dPMR/GMSK cluster where waveforms overlap. The block is producing the
right modulation but the classifier is uncertain — not a block bug.

**Pass**: prediction matches expected family and order (after aliases). Confidence
above 0.70 is typical but **not** required for pass; low-confidence correct labels
still pass and may generate a warning issue in scan.

Label aliases (GMSK family, NXDN↔dPMR, AM-SSB family/order overlap) are listed in
[docs/validation_methodology.md](docs/validation_methodology.md).

## The validation boundary

Reference IQ for scan validation is generated using:

- GNU Radio built-in blocks (`gnuradio.analog`, `gnuradio.digital`)
- numpy/scipy direct DSP

**The OOT module under validation is never used to generate its own reference data.**
Using gr-qradiolink's NBFM block to validate gr-qradiolink's NBFM block would make
the result meaningless. This boundary is enforced by an AST-based test
(`test_no_oot_imports`) that runs in CI.

The public datasets (RadioML 2016.10A, CSPB.ML.2018R2) provide cross-validation from
independent toolchains. The synthetic data fills gaps where public datasets have no
coverage. See [Training datasets](#training-datasets) and
[Why synthetic training data is used](#why-synthetic-training-data-is-used).

## Known limitations

**Classifier limitations:**

- The order classifier cannot identify modulations not in its training set. Adding
  new modes requires retraining (see [Why the classifier needs training data](#why-the-classifier-needs-training-data)).
- At 1024 samples (21 ms at 48 kHz), some modes are physically indistinguishable:
  NXDN and dPMR (identical waveforms), GMSK variants at similar symbol rates.
- SSB (AM-SSB) is genuinely ambiguous at baseband — the classifier may predict AM, FM,
  or QAM family. All are accepted as valid predictions. This is a property of the
  signal, not a bug.
- High-order FSK (128FSK, 256FSK) may be confused with lower-order variants. Family-level
  FSK classification is reliable; order-level discrimination degrades above 8FSK.

**Scope limitations:**

- Protocol framing, FEC, and codec correctness require unit tests, not IQ
  classification. The tool validates physical layer only.
- Spread-spectrum modes (DSSS, GDSS) are excluded by design — they are designed to look
  like noise and classifier output is meaningless for them.
- LDPC, Reed-Solomon, and other FEC codecs are not modulations and cannot be validated
  by IQ classification.
- CW (Morse) requires longer observation windows than 1024 samples at typical beacon
  speeds and is not reliably classifiable.
- **Custom modes from one wideband `.iq` file:** Cannot detect a bespoke multi-carrier
  or other custom mode as its own class; only generic family/order labels apply. See
  [Custom and multi-carrier modes (IQ)](#custom-and-multi-carrier-modes-iq).
- **gr-sleipnir multi-carrier:** Validate each carrier with a separate IQ file and
  sidecar, or use the `sleipnir_8qpsk` plugin for the composite case.
- **No IQ-based training:** Cannot train on contributed `.iq` captures or define new
  classes without extending the codebase and supplying labelled dataset sources.
- **CSPB original dataset** must not be used for training (RNG flaw); use R2 only.

**Infrastructure limitations:**

- The INT8 order classifier cannot be produced with current calibration data (50
  classes exceed static QDQ accuracy threshold). The order classifier deploys as FP32
  ONNX (~2.7 MB, ~15 ms CPU). The family classifier deploys as INT8 (~703 KB, ~5 ms CPU).
  See [models/README.md](models/README.md) and [NPU deployment (INT8)](#npu-deployment-int8).
- SpacemiT NPU `.nb` model files require the SpacemiT NPU SDK for conversion, which is
  not available in the CI environment. Convert on the K3 target using `rmv export-npu`.
- The database (`.rmv_findings.db`) accumulates issues from all scan runs. Use
  `rmv scan purge --keep-latest` or `rmv scan issues --since <timestamp>` to view only
  current results.

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
