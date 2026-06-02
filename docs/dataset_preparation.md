# Dataset Preparation

Training uses three public datasets plus optional **synthetic** IQ generated locally.
Use **CSPB.ML.2018R2 only** (not the original 2018 release; see [Known issues](#known-issues)).

## Automatic download (recommended)

Download and prepare all datasets under `datasets/`:

```bash
uv sync
rmv dataset download
```

Download a single dataset:

```bash
rmv dataset download --radioml
rmv dataset download --hisarmod
rmv dataset download --cspb
```

Options:

| Flag | Description |
|------|-------------|
| `--dest PATH` | Root directory [default: `datasets/`] |
| `--force` | Re-download even if files exist |
| `--verify-only` | Skip download; verify checksums only |

### What to expect

| Dataset | Approx. size | Time (typical) | Notes |
|---------|--------------|----------------|-------|
| RadioML 2016.10A | ~1.1 GB compressed | 5-30 min | Direct opendata.deepsig.ai URL |
| HISARMOD 2019.1 | varies (HDF5) | 5-60 min | IQFormer GitHub mirror if available |
| CSPB.ML.2018R2 | many batch archives | 30+ min | Scraped from cyclostationary.blog |

**Disk space:** allow at least **15-20 GB** for all three extracted/prepared.

Progress bars show per-file speed, ETA, and bytes downloaded. Interrupted HTTP
downloads resume when the server supports Range requests (partial files are never
treated as complete without checksum or size verification).

**Timeout:** use `--timeout SECONDS` on `rmv dataset download` (default 300). Increase
for slow links or multi-gigabyte files at remote sites.

**Proxy:** httpx reads `HTTP_PROXY` and `HTTPS_PROXY` automatically. Example:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
rmv dataset download
```

Check status anytime:

```bash
rmv dataset status
```

## Manual download fallback

### RadioML 2016.10A

If the primary URL fails:

1. Visit https://www.deepsig.ai/datasets
2. Download `RML2016.10a.tar.bz2`
3. Place at: `datasets/radioml/RML2016.10a.tar.bz2`
4. Run: `rmv dataset verify` (extracts automatically on next `rmv train` or re-run download)

### HISARMOD 2019.1

If the IQFormer GitHub mirror is unavailable:

1. Go to https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset
2. Log in with an IEEE account
3. Download `HisarMod2019.1.h5` or `.mat` files
4. Place HDF5 at: `datasets/hisarmod/HisarMod2019.1.h5`

Convert `.mat` to HDF5:

```bash
rmv dataset convert-hisarmod --input path/to/HisarMod.mat --output datasets/hisarmod/HisarMod2019.1.h5
```

### CSPB.ML.2018R2

If blog scraping finds no R2 links:

1. Go to https://cyclostationary.blog/data-sets/
2. Open the **CSPB.ML.2018R2** correction post
3. Download all R2 batch archives
4. Extract into `datasets/cspb/`
5. Download the R2 metadata file from the post (link: "The new metadata ... can be found here").
   It is named `signal_record_C_2023.txt` (~9 MB), not inside the batch zips. Save it under
   `datasets/cspb/` (or run `rmv dataset download --cspb` to fetch it automatically).

**Do not use original CSPB.ML.2018 files** (RNG flaw).

## Verifying downloads

```bash
rmv dataset verify
```

Exit code `0` when every **present** dataset passes checksum checks; `1` on corruption or
missing required files. **HISARMOD 2019.1 is optional** (reported as missing, not a hard error).
**CSPB.ML.2018R2** is detected from `Batch_Dir_*` folders and `CSPB.ML_.2018R2_*.zip` names;
without `signal_record_C_2023.txt` (or `signal_record.txt`) verify reports signals present but
labels unavailable for training.

Inspect downloaded data (classes, sample counts, SNR range):

```bash
rmv dataset info
```

Maintain checksums in `datasets/.manifest.json` after verifying a new dataset version:

```bash
rmv dataset checksum-update --dataset radioml
rmv dataset checksum-update --dataset hisarmod
rmv dataset checksum-update --dataset cspb
```

`rmv dataset status` reads the manifest (no full re-hash). Use `rmv dataset verify` to
re-hash files explicitly.

## Training with auto-detection

`rmv train` searches `datasets/` when paths are omitted:

```bash
uv sync --extra train
rmv train   # prompts to download missing datasets
rmv train -y   # download missing datasets without prompting
```

Explicit paths still override defaults:

```bash
rmv train --radioml datasets/radioml/RML2016.10a_dict.pkl --hisarmod datasets/hisarmod/HisarMod2019.1.h5 --cspb datasets/cspb/
```

## Dataset versions used

| Dataset | Version | Download date | SHA-256 (first 16 chars) |
|---------|---------|---------------|--------------------------|
| RadioML | 2016.10A | TBD | UNVERIFIED |
| HISARMOD | 2019.1 | TBD | UNVERIFIED |
| CSPB | ML.2018R2 | TBD | manifest / UNVERIFIED |

Update this table after the first verified download (`rmv dataset checksum-update`).

## Directory layout

```
datasets/
  radioml/
    RML2016.10a.tar.bz2
    RML2016.10a_dict.pkl
  hisarmod/
    HisarMod2019.1.h5
  cspb/
    signal_record.txt
    signal_*.tim
    .rmv_cspb_checksums.json   # CSPB batch checksum manifest
```

## Synthetic training data (optional)

Modes missing from public datasets (NBFM, squelch tones, aviation AM, WBFM, PSK,
GMSK BT=0.5/0.3, protocol 4FSK, P25 C4FM, packet AFSK/G3RUH) are generated locally
with GNU Radio built-ins and numpy/scipy only — never OOT blocks under validation.

```bash
uv sync --extra train   # GNU Radio needed for NBFM/WBFM and GMSK paths
uv run rmv dataset generate-synthetic --output datasets/synthetic/
```

Default run: **19** modes × 1000 chunks × 26 SNR levels ≈ **494k** samples written to
`datasets/synthetic/synthetic.npz` (gitignored). Subset with `--modes`, for example
`gmsk_bt05,gmsk_bt03` or `dmr,m17,ysf,nxdn,dpmr,p25`.

Include in training:

```bash
uv run rmv train \
  --synthetic datasets/synthetic/synthetic.npz \
  --datasets-dir datasets \
  --cache .cache \
  --output checkpoints \
  -y
```

After adding or changing synthetic classes, remove `.cache/` before retraining so
preprocessed shards are rebuilt.

See [README.md](../README.md) — **Synthetic orders (19)** and
[validation_methodology.md](validation_methodology.md) for scan-side label rules.

## Cache directory

Preprocessed training shards are stored under `.cache/` (see training `--cache`).
Dataset downloads in `datasets/` are separate from the preprocessing cache.

## Known issues

- **CSPB original has RNG flaw** -- always use **R2** from the 2023 correction post.
- **HISARMOD requires IEEE account** unless the IQFormer GitHub release mirror is available.
- **RadioML hosting** -- `rmv` tries `opendata.deepsig.io` then a Zenodo mirror
  (https://zenodo.org/records/18397070). The `opendata.deepsig.ai` hostname often
  does not resolve. Manual download from https://www.deepsig.ai/datasets still works.
- **CSP blog links** may move; use manual fallback if scraping finds no R2 archives.
- Checksums are stored in `datasets/.manifest.json` (not modified in the installed package).
  Release fallbacks in source start as `UNVERIFIED` until `rmv dataset checksum-update`.
- CSPB scraper only accepts links with explicit R2 markers; original 2018 files trigger
  a hard warning and are rejected for training.
- HISARMOD `.mat` files from IEEE are often MATLAB v7.3 (HDF5); `convert-hisarmod` uses
  h5py, not `scipy.io.loadmat`.
