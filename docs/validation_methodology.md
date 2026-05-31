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

## Order label aliases

Some expected orders accept closely related predictions that the classifier was
trained on under a different name:

| Expected | Also accepts |
|----------|----------------|
| GMSK | MSK, GMSK |
| MSK | MSK, GMSK |
| CPFSK | CPFSK, GFSK |
| GFSK | GFSK, CPFSK |

### GMSK and MSK

GMSK and MSK are treated as equivalent for validation purposes. GMSK (Gaussian
Minimum Shift Keying) is a variant of MSK with Gaussian pulse shaping. The
classifier, trained on RadioML MSK labels, correctly identifies GMSK-layer
reference signals as MSK. This is a correct result at the signal level.

### Generic FSK and CPFSK

Scan modes `2FSK`, `4FSK`, `8FSK`, and `P25` use **CPFSK** as the expected order in
the mode table. RadioML uses CPFSK as the generic continuous-phase FSK label; the
numpy reference generator produces CPFSK-like waveforms that the order classifier
labels as CPFSK, not as `2FSK` / `4FSK` / `8FSK` order strings.

Protocol-specific modes (**DMR**, **M17**, **YSF**, **NXDN**, **dPMR**) keep their
exact trained order labels and are not remapped to CPFSK.

## Known remaining soft fails

These are documented limitations, not scan pipeline bugs:

- **NXDN vs dPMR**: Synthetic waveforms use identical modulation parameters; the
  classifier may predict either order.
- **SSB vs WBFM**: Baseband SSB reference can be ambiguous at order level (WBFM).
- **8PSK vs QPSK**: Training and model resolution limit; left as soft fail when
  order does not match.

## Reference IQ

Scan reference IQ is generated from GNU Radio built-in blocks and numpy/scipy only.
OOT project blocks are never used for reference data, to avoid circular validation.
