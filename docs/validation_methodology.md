# Validation methodology

This document explains how `rmv scan` and `rmv validate` decide pass, warning, and fail
for modulation family and order labels.

## Pass and fail rules

- **Family pass** and **order pass** are based on label correctness only, not on
  confidence above the scan threshold.
- A matching prediction with low confidence is still a **pass**; the scan runner
  may add a **warning** issue for low confidence.
- **Hard fail** is reserved for wrong family or family confidence below the hard-fail
  floor (0.40).
- **Soft fail** is recorded only when the predicted family or order does not match
  the expected label (after aliases, see below).

## Family label aliases

**SSB** (`AM-SSB`, `AM-DSB-SC`, `USB`, `LSB`): accepts **AM**, **FM**, or **QAM** family
prediction. SSB at baseband is spectrally one-sided and has variable envelope
characteristics that overlap with both FM (instantaneous frequency variation) and
low-order QAM (amplitude + phase variation) at 1024-sample observation windows. The
**AM** family is the correct target, but **FM** and **QAM** are acceptable classifier
outputs given the ambiguity.

**AM-DSB** and other non-SSB orders require an exact family match (no FM/QAM alias).

## Order label aliases

Some expected orders accept closely related predictions that the classifier was
trained on under a different name:

| Expected | Also accepts |
|----------|----------------|
| GMSK | GMSK, GMSK_BT05, GMSK_BT03, MSK, NXDN, dPMR |
| GMSK_BT05 | GMSK, GMSK_BT05, GMSK_BT03, MSK, NXDN, dPMR |
| GMSK_BT03 | GMSK, GMSK_BT05, GMSK_BT03, MSK, NXDN, dPMR |
| MSK | GMSK, GMSK_BT05, GMSK_BT03, MSK, NXDN, dPMR |
| AM-SSB | AM-SSB, AM-DSB, WBFM, NBFM_25, NBFM_50, NFM_CTCSS, NFM_DCS |
| CPFSK | CPFSK, GFSK |
| GFSK | GFSK, CPFSK |
| AM-DSB | AM-DSB, AM_AIR_25K, AM_AIR_833 |
| AM_AIR_25K | AM_AIR_25K, AM-DSB, AM_AIR_833 |
| AM_AIR_833 | AM_AIR_833, AM-DSB, AM_AIR_25K |
| NXDN | NXDN, dPMR |
| dPMR | dPMR, NXDN |

### GMSK, GMSK_BT05, GMSK_BT03, MSK, NXDN, and dPMR

**GMSK**, **GMSK_BT05**, **GMSK_BT03**, **MSK**, **NXDN**, and **dPMR** share one alias
group when **GMSK** (or BT variants) is expected — 4800 baud GMSK and 2400 baud NXDN/dPMR
overlap in deviation at 1024 samples and moderate SNR. **NXDN** / **dPMR** sidecars still
accept only each other, not **GMSK**.

### Generic FSK and CPFSK

Scan modes `2FSK`, `4FSK`, and `8FSK` use **CPFSK** as the expected order in the
mode table. RadioML uses CPFSK as the generic continuous-phase FSK label; the
numpy reference generator produces CPFSK-like waveforms that the order classifier
labels as CPFSK, not as `2FSK` / `4FSK` / `8FSK` order strings.

Protocol-specific modes (**DMR**, **M17**, **YSF**, **NXDN**, **dPMR**, **P25**)
keep their exact trained order labels and are not remapped to CPFSK.

Packet link-layer names (**AX.25**, **APRS**, **FX.25**, **IL2P**) map to
**BELL202** (Bell 202 AFSK physical layer only; framing is not verified).

### Aviation and broadcast AM (DSB)

**AM-DSB**, **AM_AIR_25K**, and **AM_AIR_833** are all full-carrier DSB-AM in the
**AM** family; they differ mainly in audio bandwidth and channel spacing. At order
level, any of these labels is accepted when another was expected (for example
aviation capture classified as **AM-DSB**).

### NXDN and dPMR

**NXDN** and **dPMR** use identical synthetic 4FSK parameters in training; the
classifier may predict either order. Aliases are symmetric: each expected label
accepts the other.

### AM-SSB (scan reference)

**AM-SSB** accepts several FM-family orders (**WBFM**, **NBFM_*** , **NFM_***) when the
classifier confuses baseband SSB with deviation-like structure. Family **FM** or **QAM**
is also accepted when `expected_order` is **AM-SSB** (or **USB** / **LSB**).

## Known remaining soft fails

These are documented limitations, not scan pipeline bugs:

- **8PSK vs QPSK**: Training and model resolution limit; left as soft fail when
  order does not match.
- **8PSK vs QPSK**: Training and model resolution limit; left as soft fail when
  order does not match.

## Scan reference mapping (synthetic)

When `rmv scan` generates reference IQ for a mode, these orders are used (OOT blocks are
not used):

| Scan mode | Synthetic / expected order | Notes |
|-----------|---------------------------|--------|
| `D-Star`, `DSTAR` | GMSK_BT05 | BT=0.5, D-Star profile |
| `GMSK` | GMSK_BT03 | BT=0.3 standard GMSK |
| `P25` | P25 | C4FM synthetic, not CPFSK |
| `2FSK`, `4FSK`, `8FSK` | CPFSK | Generic continuous-phase FSK |
| `AX.25`, `APRS`, `FX.25`, `IL2P` | BELL202 | Physical layer only |
| `DMR`, `M17`, `YSF`, `NXDN`, `dPMR` | same name | Protocol-accurate 4FSK |

`FreeDV` may still use a legacy GMSK-shaped scan generator; validate against **GMSK**
family expectations per the mode table.

## Reference IQ

Scan reference IQ is generated from GNU Radio built-in blocks and numpy/scipy only.
OOT project blocks are never used for reference data, to avoid circular validation.
