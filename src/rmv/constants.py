"""Class names, family mappings, and label indices."""

from __future__ import annotations

# RadioML 2016.10A modulation classes
RADIOML_CLASSES: list[str] = [
    "AM-DSB",
    "AM-SSB",
    "WBFM",
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16",
    "QAM64",
    "CPFSK",
    "GFSK",
    "PAM4",
]

RADIOML_TO_FAMILY: dict[str, str] = {
    "AM-DSB": "AM",
    "AM-SSB": "AM",
    "WBFM": "FM",
    "BPSK": "PSK",
    "QPSK": "PSK",
    "8PSK": "PSK",
    "16QAM": "QAM",
    "64QAM": "QAM",
    "QAM16": "QAM",
    "QAM64": "QAM",
    "CPFSK": "FSK",
    "GFSK": "FSK",
    "PAM4": "PAM",
}

# RadioML 128-sample windows upsampled to 1024 lack clear FM/PSK structure;
# family training uses synthetic replacements for these orders instead.
RADIOML_SKIP_FOR_FAMILY: frozenset[str] = frozenset({"WBFM", "BPSK", "QPSK"})

# HISARMOD 2019.1 - 26 classes
HISARMOD_CLASSES: list[str] = [
    "2FSK",
    "4FSK",
    "8FSK",
    "16FSK",
    "32FSK",
    "64FSK",
    "128FSK",
    "256FSK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-DSB",
    "AM-SSB",
    "FM",
    "GMSK",
    "OQPSK",
    "MSK",
    "OOK",
    "4ASK",
]

HISARMOD_TO_FAMILY: dict[str, str] = {
    "2FSK": "FSK",
    "4FSK": "FSK",
    "8FSK": "FSK",
    "16FSK": "FSK",
    "32FSK": "FSK",
    "64FSK": "FSK",
    "128FSK": "FSK",
    "256FSK": "FSK",
    "BPSK": "PSK",
    "QPSK": "PSK",
    "8PSK": "PSK",
    "16PSK": "PSK",
    "32PSK": "PSK",
    "16QAM": "QAM",
    "32QAM": "QAM",
    "64QAM": "QAM",
    "128QAM": "QAM",
    "256QAM": "QAM",
    "AM-DSB": "AM",
    "AM-SSB": "AM",
    "FM": "FM",
    "GMSK": "FSK",
    "OQPSK": "PSK",
    "MSK": "FSK",
    "OOK": "AM",
    "4ASK": "AM",
}

# CSPB.ML.2018R2 modulation strings (truth file lowercase)
CSPB_CLASS_ALIASES: dict[str, str] = {
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "8psk": "8PSK",
    "dqpsk": "DQPSK",
    "16qam": "16QAM",
    "64qam": "64QAM",
    "256qam": "256QAM",
    "msk": "MSK",
}

CSPB_CLASSES: list[str] = sorted(set(CSPB_CLASS_ALIASES.values()))

CSPB_TO_FAMILY: dict[str, str] = {
    "BPSK": "PSK",
    "QPSK": "PSK",
    "8PSK": "PSK",
    "DQPSK": "PSK",
    "16QAM": "QAM",
    "64QAM": "QAM",
    "256QAM": "QAM",
    "MSK": "FSK",
}

# Unified family and order label sets for trained models
FAMILY_CLASSES: list[str] = ["AM", "FM", "FSK", "PSK", "QAM", "PAM"]

# Order labels span all datasets; training uses dataset-specific order heads
SYNTHETIC_CLASSES: list[str] = [
    "NBFM_25",
    "NBFM_50",
    "AM_AIR_25K",
    "AM_AIR_833",
    "WBFM",
    "BPSK",
    "QPSK",
    "DMR",
    "M17",
    "YSF",
    "NXDN",
    "dPMR",
]

SYNTHETIC_TO_FAMILY: dict[str, str] = {
    "NBFM_25": "FM",
    "NBFM_50": "FM",
    "AM_AIR_25K": "AM",
    "AM_AIR_833": "AM",
    "WBFM": "FM",
    "BPSK": "PSK",
    "QPSK": "PSK",
    "DMR": "FSK",
    "M17": "FSK",
    "YSF": "FSK",
    "NXDN": "FSK",
    "dPMR": "FSK",
}

ORDER_CLASSES: list[str] = sorted(
    set(RADIOML_CLASSES)
    | set(HISARMOD_CLASSES)
    | set(CSPB_CLASSES)
    | set(SYNTHETIC_CLASSES)
    | {
        "16QAM",
        "64QAM",
        "NBFM",
        "WBFM",
        "2FSK",
        "4FSK",
        "DQPSK",
        "MSK",
    }
)

CHUNK_SAMPLES = 1024
CHUNK_FLOATS = CHUNK_SAMPLES * 2
MAX_IQ_FILE_BYTES = 50 * 1024 * 1024
HARD_FAIL_CONFIDENCE = 0.40
