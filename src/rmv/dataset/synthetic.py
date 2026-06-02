"""
Synthetic IQ data generation for radio modulation modes missing from
the public training datasets (RadioML 2016.10A, CSPB.ML.2018R2).

Modes generated:
  NBFM_25    — Amateur narrowband FM, ±2.5 kHz deviation, 12.5 kHz channel
  NBFM_50    — Amateur narrowband FM, ±5.0 kHz deviation, 25 kHz channel
  AM_AIR_25K — Aviation DSB-AM, 25 kHz channel spacing (ICAO Annex 10)
  AM_AIR_833 — Aviation DSB-AM, 8.33 kHz channel spacing (EU Reg. 1079/2012)
  WBFM       — Broadcast FM, 75 kHz deviation (replaces weak RadioML WBFM)
  BPSK       — GNU Radio digital.psk_mod, 8 sps @ 48 kHz
  QPSK       — GNU Radio digital.psk_mod, 8 sps @ 48 kHz

4FSK protocol-accurate modes (numpy/scipy only, no OOT modules):
  DMR    — ETSI TS 102 361-1, 4800 baud, +/-1944 Hz, RC alpha=0.2
  M17    — M17 Project spec v1.0, 4800 baud, +/-2400 Hz, RRC beta=0.5
  YSF    — TIA-102/Yaesu C4FM, 4800 baud, +/-2400 Hz, Gaussian BT=0.5
  NXDN   — ICOM/Kenwood, 2400 baud, +/-1050 Hz, RC alpha=0.2
  dPMR   — ETSI TS 102 490, 2400 baud, +/-1050 Hz, RC alpha=0.2
  P25    — TIA-102.BAAA C4FM, 4800 baud, +/-1800/600 Hz, RC alpha=0.2

Squelch tone modes (numpy/scipy + gnuradio.analog FM chain):
  NFM_CTCSS — NFM 12.5 kHz with CTCSS subaudible tone (67-254 Hz, EIA/TIA-603)
  NFM_DCS   — NFM 12.5 kHz with DCS digital squelch (134.4 bit/s, ETSI TS 103 236)

Packet radio physical layers (numpy/scipy):
  BELL202   — Bell 202 AFSK, mark=1200 Hz, space=2200 Hz, 1200 baud, NRZI
  G3RUH     — G3RUH 9600 baud direct FSK, +/-3500 Hz, Gaussian BT=0.5

GMSK (GNU Radio digital.gmsk_mod, 4800 baud, 10 sps @ 48 kHz):
  GMSK_BT05 — BT=0.5 (D-Star / MSK-equivalent profile)
  GMSK_BT03 — BT=0.3 (standard GMSK filtering)

Note: FX.25, IL2P, and AX.25 are protocol framings over Bell 202 AFSK.
They are not separate modulations and do not get separate classifier classes.

These are modulation-layer waveforms only. Protocol framing, sync
words, FEC, and vocoders are NOT reproduced. This is intentional —
the classifier validates modulation family and order, not protocol
correctness.

IMPORTANT: This module uses GNU Radio built-in blocks and numpy/scipy only.
No OOT (out-of-tree) modules are used. This is intentional —
OOT modules under validation cannot be used to generate reference
data without making the validation circular.

Specifically, the qradiolink OOT module is NOT used here even
though it is installed. Its blocks are the subject of validation,
not a source of ground truth.

Verified sources used:
  - gnuradio.analog.frequency_modulator_fc, gnuradio.filter (NBFM chain)
  - gnuradio.digital.gmsk_mod, digital.psk_mod (GMSK, PSK)
  - numpy / scipy for direct DSP (aviation AM, GMSK fallback)

Note: ``analog.nbfm_tx(tau=0)`` raises in upstream GNU Radio (division by zero
in preemphasis). Amateur NBFM here uses the same chain without preemphasis:
interp to quad_rate + ``frequency_modulator_fc`` (equivalent to nbfm_tx with
tau=0).

Parameters follow:
  NBFM: ITU-R SM.1138, IARU Region 1 band plan
        tau=0.0 is intentional — preemphasis is broadcast FM only
  Aviation AM: ICAO Annex 10 Volume III, EU Regulation 1079/2012
               modulation_index=0.85, audio 300-3400 Hz (25K)
                                      audio 300-2500 Hz (8.33K)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from rmv.constants import CHUNK_SAMPLES
from rmv.dataset.preprocess import normalise_unit_power
from rmv.types import IQDataset

logger = logging.getLogger(__name__)

BROADCAST_FM_TAU = 75e-6
DEFAULT_AUDIO_RATE_HZ = 8000
DEFAULT_QUAD_RATE_HZ = 48000
DEFAULT_MODULATION_INDEX = 0.85
DEFAULT_CHUNKS_PER_SNR = 1000
DEFAULT_FREQ_OFFSET_HZ = 200.0

# EIA/TIA-603 CTCSS tone table (subaudible, 67.0-254.1 Hz)
CTCSS_TONES_HZ: tuple[float, ...] = (
    67.0,
    69.3,
    71.9,
    74.4,
    77.0,
    79.7,
    82.5,
    85.4,
    88.5,
    91.5,
    94.8,
    97.4,
    100.0,
    103.5,
    107.2,
    110.9,
    114.8,
    118.8,
    123.0,
    127.3,
    131.8,
    136.5,
    141.3,
    146.2,
    151.4,
    156.7,
    162.2,
    167.9,
    173.8,
    179.9,
    186.2,
    192.8,
    199.5,
    203.5,
    206.5,
    210.7,
    218.1,
    225.7,
    229.1,
    233.6,
    241.8,
    250.3,
    254.1,
)

# 104 standard DCS codes (3-digit octal, ETSI TS 103 236)
DCS_CODES: tuple[int, ...] = (
    23,
    25,
    26,
    31,
    32,
    36,
    43,
    47,
    51,
    53,
    54,
    65,
    71,
    72,
    73,
    74,
    114,
    115,
    116,
    122,
    125,
    131,
    132,
    134,
    143,
    145,
    152,
    155,
    156,
    162,
    165,
    172,
    174,
    205,
    212,
    223,
    225,
    226,
    243,
    244,
    245,
    246,
    251,
    252,
    255,
    261,
    263,
    265,
    266,
    271,
    274,
    306,
    311,
    315,
    325,
    331,
    332,
    343,
    346,
    351,
    364,
    365,
    371,
    411,
    412,
    413,
    423,
    431,
    432,
    445,
    464,
    465,
    466,
    503,
    506,
    516,
    523,
    526,
    532,
    546,
    565,
    606,
    612,
    624,
    627,
    631,
    632,
    654,
    662,
    664,
    703,
    712,
    723,
    731,
    732,
    734,
    743,
    754,
)

NFM_CTCSS_VOICE_DEV_HZ = 2000.0
NFM_CTCSS_TONE_DEV_HZ = 500.0
NFM_DCS_VOICE_DEV_HZ = 2000.0
NFM_DCS_SHIFT_HZ = 134.0
NFM_DCS_BIT_RATE = 134.4
BELL202_MARK_HZ = 1200.0
BELL202_SPACE_HZ = 2200.0
BELL202_BAUD = 1200.0
BELL202_FM_DEV_HZ = 3500.0
G3RUH_BAUD = 9600.0
G3RUH_DEV_HZ = 3500.0
GMSK_SYMBOL_RATE_HZ = 4800.0
GMSK_SAMPLES_PER_SYMBOL = 10
GMSK_MODULATION_INDEX = 0.5

SNR_DB_LEVELS: list[float] = [float(s) for s in range(-20, 32, 2)]

NBFM_TONE_HZ = (300.0, 500.0, 1000.0, 2000.0, 3000.0)
AVIATION_TONE_HZ = (500.0, 1000.0, 2000.0)

SyntheticMode = Literal[
    "nbfm25",
    "nbfm50",
    "am_air_25k",
    "am_air_833",
    "wbfm",
    "bpsk",
    "qpsk",
    "dmr",
    "m17",
    "ysf",
    "nxdn",
    "dpmr",
    "nfm_ctcss",
    "nfm_dcs",
    "p25",
    "bell202",
    "g3ruh",
    "gmsk_bt05",
    "gmsk_bt03",
]

MODE_TO_CLASS: dict[SyntheticMode, str] = {
    "nbfm25": "NBFM_25",
    "nbfm50": "NBFM_50",
    "am_air_25k": "AM_AIR_25K",
    "am_air_833": "AM_AIR_833",
    "wbfm": "WBFM",
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "dmr": "DMR",
    "m17": "M17",
    "ysf": "YSF",
    "nxdn": "NXDN",
    "dpmr": "dPMR",
    "nfm_ctcss": "NFM_CTCSS",
    "nfm_dcs": "NFM_DCS",
    "p25": "P25",
    "bell202": "BELL202",
    "g3ruh": "G3RUH",
    "gmsk_bt05": "GMSK_BT05",
    "gmsk_bt03": "GMSK_BT03",
}

ALL_MODES: tuple[SyntheticMode, ...] = (
    "nbfm25",
    "nbfm50",
    "am_air_25k",
    "am_air_833",
    "wbfm",
    "bpsk",
    "qpsk",
    "dmr",
    "m17",
    "ysf",
    "nxdn",
    "dpmr",
    "nfm_ctcss",
    "nfm_dcs",
    "p25",
    "bell202",
    "g3ruh",
    "gmsk_bt05",
    "gmsk_bt03",
)

PROTOCOL_4FSK_ORDERS: frozenset[str] = frozenset(
    {"DMR", "M17", "YSF", "NXDN", "dPMR", "P25"}
)

SYNTHETIC_VARIANT_ORDERS: frozenset[str] = PROTOCOL_4FSK_ORDERS | frozenset(
    {"NFM_CTCSS", "NFM_DCS", "BELL202", "G3RUH", "GMSK_BT05", "GMSK_BT03"}
)

DEFAULT_PSK_SAMPLES_PER_SYMBOL = 8

@dataclass(frozen=True)
class VariantSpec:
    """Per-modulation generation and verification limits."""

    class_name: str
    max_bandwidth_hz: float
    kind: Literal[
        "nbfm",
        "am_air",
        "wbfm",
        "psk",
        "fsk4",
        "nbfm_ctcss",
        "nbfm_dcs",
        "afsk",
        "fsk2",
        "gmsk",
    ]
    max_dev: float | None = None
    am_variant: Literal["25k", "833"] | None = None
    psk_order: int | None = None
    gmsk_bt: float | None = None


VARIANT_SPECS: dict[str, VariantSpec] = {
    "NBFM_25": VariantSpec("NBFM_25", 7000.0, "nbfm", max_dev=2500.0),
    "NBFM_50": VariantSpec("NBFM_50", 13000.0, "nbfm", max_dev=5000.0),
    "AM_AIR_25K": VariantSpec("AM_AIR_25K", 8000.0, "am_air", am_variant="25k"),
    "AM_AIR_833": VariantSpec("AM_AIR_833", 6500.0, "am_air", am_variant="833"),
    "WBFM": VariantSpec("WBFM", 200000.0, "wbfm", max_dev=75000.0),
    "BPSK": VariantSpec("BPSK", 12000.0, "psk", psk_order=2),
    "QPSK": VariantSpec("QPSK", 12000.0, "psk", psk_order=4),
    "DMR": VariantSpec("DMR", 14000.0, "fsk4"),
    "M17": VariantSpec("M17", 14000.0, "fsk4"),
    "YSF": VariantSpec("YSF", 14000.0, "fsk4"),
    "NXDN": VariantSpec("NXDN", 8000.0, "fsk4"),
    "dPMR": VariantSpec("dPMR", 8000.0, "fsk4"),
    "P25": VariantSpec("P25", 14000.0, "fsk4"),
    "NFM_CTCSS": VariantSpec("NFM_CTCSS", 7200.0, "nbfm_ctcss", max_dev=2500.0),
    "NFM_DCS": VariantSpec("NFM_DCS", 7200.0, "nbfm_dcs", max_dev=2500.0),
    "BELL202": VariantSpec("BELL202", 14000.0, "afsk"),
    "G3RUH": VariantSpec("G3RUH", 14000.0, "fsk2"),
    "GMSK_BT05": VariantSpec("GMSK_BT05", 14000.0, "gmsk", gmsk_bt=0.5),
    "GMSK_BT03": VariantSpec("GMSK_BT03", 14000.0, "gmsk", gmsk_bt=0.3),
}


@dataclass(frozen=True)
class Fsk4ProtocolSpec:
    """Per-protocol 4FSK modulation parameters."""

    symbol_rate_hz: float
    symbol_map: dict[int, float]
    filter_type: Literal["rc", "rrc", "gauss"]
    filter_param: float
    filter_span: int
    inner_dev_hz: float
    outer_dev_hz: float


FSK4_PROTOCOL_SPECS: dict[str, Fsk4ProtocolSpec] = {
    "DMR": Fsk4ProtocolSpec(
        4800.0,
        {3: 1944.0, 2: 648.0, 0: -648.0, 1: -1944.0},
        "rc",
        0.2,
        8,
        648.0,
        1944.0,
    ),
    "M17": Fsk4ProtocolSpec(
        4800.0,
        {1: 2400.0, 0: 800.0, 2: -800.0, 3: -2400.0},
        "rrc",
        0.5,
        8,
        800.0,
        2400.0,
    ),
    "YSF": Fsk4ProtocolSpec(
        4800.0,
        {0: 2400.0, 1: 800.0, 3: -800.0, 2: -2400.0},
        "gauss",
        0.5,
        4,
        800.0,
        2400.0,
    ),
    "NXDN": Fsk4ProtocolSpec(
        2400.0,
        {3: 1050.0, 2: 350.0, 0: -350.0, 1: -1050.0},
        "rc",
        0.2,
        8,
        350.0,
        1050.0,
    ),
    "dPMR": Fsk4ProtocolSpec(
        2400.0,
        {3: 1050.0, 2: 350.0, 0: -350.0, 1: -1050.0},
        "rc",
        0.2,
        8,
        350.0,
        1050.0,
    ),
    "P25": Fsk4ProtocolSpec(
        4800.0,
        {1: 1800.0, 0: 600.0, 2: -600.0, 3: -1800.0},
        "rc",
        0.2,
        8,
        600.0,
        1800.0,
    ),
}


def _require_gnuradio() -> None:
    try:
        __import__("gnuradio.analog")
    except ImportError as exc:
        msg = (
            "GNU Radio is required for NBFM synthesis. Install OS packages "
            "and use a venv with --system-site-packages (see README)."
        )
        raise ImportError(msg) from exc


def validate_nbfm_params(
    tau: float,
    max_dev: float,
    *,
    class_name: str = "NBFM",
) -> None:
    """Reject broadcast-FM or out-of-spec amateur NBFM parameters."""
    if abs(tau - BROADCAST_FM_TAU) < 1e-12:
        msg = (
            f"{class_name}: tau=75e-6 is broadcast FM preemphasis, "
            "not valid for amateur NFM (use tau=0.0)"
        )
        raise ValueError(msg)
    if tau < 0.0:
        raise ValueError(f"{class_name}: tau must be non-negative")
    spec = VARIANT_SPECS.get(class_name)
    if spec is not None and spec.max_dev is not None:
        if max_dev > spec.max_dev * 3.0:
            msg = (
                f"{class_name}: max_dev={max_dev} Hz is far above the "
                f"amateur standard ({spec.max_dev} Hz)"
            )
            raise ValueError(msg)


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add complex AWGN at the given SNR (dB) relative to signal power."""
    signal_power = float(np.mean(np.abs(signal) ** 2))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    scale = np.sqrt(noise_power / 2)
    if rng is None:
        noise = scale * (
            np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
        )
    else:
        noise = scale * (
            rng.standard_normal(signal.shape)
            + 1j * rng.standard_normal(signal.shape)
        )
    return (signal + noise).astype(np.complex64)


def add_freq_offset(
    signal: np.ndarray,
    offset_hz: float,
    sample_rate_hz: float,
) -> np.ndarray:
    """Apply a complex exponential frequency offset."""
    t = np.arange(len(signal), dtype=np.float64) / sample_rate_hz
    return (signal * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def _occupied_bandwidth_hz(iq_row: np.ndarray, sample_rate_hz: float) -> float:
    """Occupied bandwidth (-26 dB from peak) for one complex chunk."""
    psd = np.abs(np.fft.fftshift(np.fft.fft(iq_row))) ** 2
    psd_db = 10 * np.log10(psd + 1e-12)
    peak_db = float(np.max(psd_db))
    occupied = np.where(psd_db >= peak_db - 26.0)[0]
    if occupied.size == 0:
        return 0.0
    bw_bins = int(occupied[-1] - occupied[0] + 1)
    return bw_bins * sample_rate_hz / CHUNK_SAMPLES


def verify_bandwidth(
    chunks: np.ndarray,
    sample_rate_hz: float,
    max_bandwidth_hz: float,
    variant_name: str,
) -> None:
    """Verify generated signal bandwidth; raise if over limit."""
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    per_chunk = [_occupied_bandwidth_hz(row, sample_rate_hz) for row in iq]
    bw_hz = float(np.percentile(per_chunk, 95))
    if bw_hz > max_bandwidth_hz:
        raise ValueError(
            f"{variant_name} bandwidth {bw_hz:.0f} Hz exceeds "
            f"limit {max_bandwidth_hz:.0f} Hz. "
            f"Check modulation parameters."
        )


def _dsb_am_full_carrier(
    audio: np.ndarray,
    modulation_index: float,
    sample_rate_hz: float,
) -> np.ndarray:
    """
    DSB-AM with full carrier per ICAO Annex 10.

    At baseband (f_c = 0): complex envelope = 1 + m * a(t).
    """
    del sample_rate_hz  # real envelope; rate applied when chunking
    carrier = 1.0 + modulation_index * audio
    return carrier.astype(np.complex64)


def _bandlimit_audio_aviation(
    audio: np.ndarray,
    sample_rate_hz: float,
    variant: Literal["25k", "833"],
) -> np.ndarray:
    """Apply ICAO-compliant audio bandlimiting."""
    if variant == "25k":
        low, high = 300.0, 3400.0
    else:
        low, high = 300.0, 2500.0
    sos = butter(6, [low, high], btype="bandpass", fs=sample_rate_hz, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _bandlimit_audio_nbfm(
    audio: np.ndarray,
    sample_rate_hz: float,
    high_hz: float = 3000.0,
) -> np.ndarray:
    """Band-limit NBFM audio (default 300-3000 Hz; tighter high_hz for narrow dev)."""
    sos = butter(6, [300.0, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _nbfm_audio_high_hz(max_dev: float) -> float:
    """Highest modulating frequency so occupied BW stays within variant limit."""
    if max_dev <= 3000.0:
        return 800.0
    return 1500.0


def _generate_nbfm_audio(
    n_samples: int,
    sample_rate_hz: float,
    rng: np.random.Generator,
    *,
    high_hz: float = 3000.0,
) -> np.ndarray:
    """50% tones (300-3000 Hz), 50% band-limited noise."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    tone_choices = tuple(f for f in NBFM_TONE_HZ if f <= high_hz)
    if not tone_choices:
        tone_choices = (min(NBFM_TONE_HZ),)
    if rng.random() < 0.5:
        freq = float(rng.choice(tone_choices))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    else:
        audio = rng.standard_normal(n_samples).astype(np.float32)
    audio = _bandlimit_audio_nbfm(audio, sample_rate_hz, high_hz=high_hz)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = (audio / peak * 0.9).astype(np.float32)
    return audio


def _generate_aviation_audio(
    n_samples: int,
    sample_rate_hz: float,
    variant: Literal["25k", "833"],
    rng: np.random.Generator,
) -> np.ndarray:
    """60% band-limited noise, 40% speech-band tones."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    if rng.random() < 0.4:
        freq = float(rng.choice(AVIATION_TONE_HZ))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    else:
        audio = rng.standard_normal(n_samples).astype(np.float32)
    audio = _bandlimit_audio_aviation(audio, sample_rate_hz, variant)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = (audio / peak * 0.9).astype(np.float32)
    return audio


def _nbfm_modulate_numpy(
    audio: np.ndarray,
    quad_rate: int,
    max_dev: float,
) -> np.ndarray:
    """NBFM without preemphasis (matches frequency_modulator_fc)."""
    k = 2.0 * math.pi * max_dev / quad_rate
    phase = np.cumsum(k * audio.astype(np.float64))
    return np.exp(1j * phase).astype(np.complex64)


def _nbfm_modulate_gnuradio(
    audio: np.ndarray,
    audio_rate: int,
    quad_rate: int,
    max_dev: float,
) -> np.ndarray:
    """NBFM via GNU Radio interp + frequency_modulator_fc (no preemphasis)."""
    from gnuradio import analog, blocks, filter, gr

    interp_factor = quad_rate // audio_rate
    if quad_rate % audio_rate != 0:
        msg = f"quad_rate {quad_rate} must be integer multiple of audio_rate {audio_rate}"
        raise ValueError(msg)

    tb = gr.top_block()
    src = blocks.vector_source_f(audio.astype(np.float32).tolist(), False)
    mod = analog.frequency_modulator_fc(2.0 * math.pi * max_dev / quad_rate)
    snk = blocks.vector_sink_c()

    if interp_factor > 1:
        interp_taps = filter.optfir.low_pass(
            interp_factor,
            quad_rate,
            4500,
            7000,
            0.1,
            40,
        )
        interp = filter.interp_fir_filter_fff(interp_factor, interp_taps)
        tb.connect(src, interp, mod, snk)
    else:
        tb.connect(src, mod, snk)

    tb.run()
    return np.array(snk.data(), dtype=np.complex64)


def _audio_to_quad(
    audio: np.ndarray,
    audio_rate: int,
    quad_rate: int,
) -> np.ndarray:
    """Resample audio to quad_rate (integer ratio)."""
    factor = quad_rate // audio_rate
    if factor == 1:
        return audio.astype(np.float32)
    return resample_poly(audio, factor, 1).astype(np.float32)


def _complex_to_iq_chunk(iq: np.ndarray) -> np.ndarray:
    """Convert complex baseband segment to (2, 1024) float32."""
    seg = iq[:CHUNK_SAMPLES]
    if len(seg) < CHUNK_SAMPLES:
        seg = np.pad(seg, (0, CHUNK_SAMPLES - len(seg)))
    return np.stack([seg.real, seg.imag], axis=0).astype(np.float32)


def _bandlimit_audio_wbfm(
    audio: np.ndarray,
    sample_rate_hz: float,
    high_hz: float = 15000.0,
) -> np.ndarray:
    """Band-limit WBFM audio (roughly 50 Hz - 15 kHz)."""
    sos = butter(6, [50.0, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _generate_wbfm_audio(
    n_samples: int,
    sample_rate_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tones and band-limited noise for broadcast FM audio."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    if rng.random() < 0.5:
        freq = float(rng.uniform(200.0, 8000.0))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    else:
        audio = rng.standard_normal(n_samples).astype(np.float32)
    audio = _bandlimit_audio_wbfm(audio, sample_rate_hz)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = (audio / peak * 0.9).astype(np.float32)
    return audio


def _psk_modulate_gnuradio(
    order: int,
    n_samples: int,
    sample_rate_hz: int,
    *,
    samples_per_symbol: int = DEFAULT_PSK_SAMPLES_PER_SYMBOL,
    rng: np.random.Generator,
) -> np.ndarray:
    """PSK via GNU Radio digital.psk_mod (random byte source)."""
    from gnuradio import blocks, digital, gr

    nbytes = max(n_samples // samples_per_symbol + 8, 64)
    data = rng.integers(0, 256, size=nbytes, dtype=np.uint8).tolist()
    tb = gr.top_block()
    src = blocks.vector_source_b(data, False)
    mod = digital.psk_mod(
        constellation_points=order,
        mod_code="gray",
        differential=False,
        samples_per_symbol=samples_per_symbol,
    )
    snk = blocks.vector_sink_c()
    tb.connect(src, mod, snk)
    tb.run()
    out = np.array(snk.data(), dtype=np.complex64)
    if len(out) < n_samples:
        reps = int(np.ceil(n_samples / max(len(out), 1)))
        out = np.tile(out, reps)[:n_samples]
    return out[:n_samples]


def _psk_modulate_numpy(
    order: int,
    n_samples: int,
    *,
    samples_per_symbol: int = DEFAULT_PSK_SAMPLES_PER_SYMBOL,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fallback PSK when GNU Radio is unavailable."""
    n_syms = n_samples // samples_per_symbol + 2
    if order == 2:
        bits = rng.integers(0, 2, size=n_syms)
        syms = (2 * bits - 1).astype(np.float32)
        pulse = np.repeat(syms, samples_per_symbol)[:n_samples]
        return pulse.astype(np.complex64)
    bits = rng.integers(0, order, size=n_syms)
    mapping = np.exp(2j * np.pi * bits / order)
    return np.repeat(mapping, samples_per_symbol)[:n_samples].astype(np.complex64)


def _generate_wbfm_chunk(
    *,
    max_dev: float,
    sample_rate_hz: float,
    snr_db: float,
    use_gnuradio: bool,
    rng: np.random.Generator,
    apply_channel: bool = True,
) -> np.ndarray:
    """Generate one WBFM chunk (2, 1024) with broadcast deviation."""
    quad_rate = int(sample_rate_hz)
    audio_len = CHUNK_SAMPLES + 64
    audio = _generate_wbfm_audio(audio_len, float(quad_rate), rng)
    if use_gnuradio:
        _require_gnuradio()
        iq = _nbfm_modulate_gnuradio(audio, quad_rate, quad_rate, max_dev)
    else:
        iq = _nbfm_modulate_numpy(audio.astype(np.float32), quad_rate, max_dev)
    if apply_channel:
        offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
        iq = add_freq_offset(iq, offset, sample_rate_hz)
        iq = add_awgn(iq, snr_db)
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def _generate_psk_chunk(
    *,
    order: int,
    sample_rate_hz: float,
    snr_db: float,
    use_gnuradio: bool,
    rng: np.random.Generator,
    apply_channel: bool = True,
) -> np.ndarray:
    """Generate one BPSK or QPSK chunk (2, 1024)."""
    quad_rate = int(sample_rate_hz)
    need = CHUNK_SAMPLES + DEFAULT_PSK_SAMPLES_PER_SYMBOL * 4
    if use_gnuradio:
        try:
            _require_gnuradio()
            __import__("gnuradio.digital")
            iq = _psk_modulate_gnuradio(
                order,
                need,
                quad_rate,
                rng=rng,
            )
        except ImportError:
            iq = _psk_modulate_numpy(order, need, rng=rng)
    else:
        iq = _psk_modulate_numpy(order, need, rng=rng)
    if apply_channel:
        offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
        iq = add_freq_offset(iq, offset, sample_rate_hz)
        iq = add_awgn(iq, snr_db)
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def _gmsk_modulate_gnuradio(
    n_samples: int,
    sample_rate_hz: float,
    *,
    bt: float,
    samples_per_symbol: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """GMSK via GNU Radio built-in digital.gmsk_mod (not OOT blocks)."""
    from gnuradio import blocks, digital, gr

    n_bits = (n_samples // samples_per_symbol) + 64
    data = rng.integers(0, 2, n_bits).astype(int).tolist()
    tb = gr.top_block()
    src = blocks.vector_source_b(data, False)
    mod = digital.gmsk_mod(
        samples_per_symbol=samples_per_symbol,
        bt=bt,
        verbose=False,
        do_unpack=True,
    )
    snk = blocks.vector_sink_c()
    tb.connect(src, mod, snk)
    tb.run()
    out = np.array(snk.data(), dtype=np.complex64)
    if len(out) < n_samples:
        reps = int(np.ceil(n_samples / max(len(out), 1)))
        out = np.tile(out, reps)
    return out[:n_samples].astype(np.complex64)


def _gmsk_modulate_numpy(
    n_samples: int,
    sample_rate_hz: float,
    *,
    bt: float,
    samples_per_symbol: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian-filtered CPFSK approximating GMSK when GNU Radio is unavailable."""
    filt = _gaussian_filter(bt, 4, samples_per_symbol)
    symbol_rate_hz = sample_rate_hz / samples_per_symbol
    deviation_hz = symbol_rate_hz / 4.0  # h=0.5 -> +/-symbol_rate/4
    n_symbols = n_samples // samples_per_symbol + len(filt)
    symbols = (rng.integers(0, 2, n_symbols) * 2 - 1).astype(np.float64)
    freq_sequence = symbols * deviation_hz
    freq_up = np.repeat(freq_sequence, samples_per_symbol)
    freq_shaped = np.convolve(freq_up, filt, mode="same")[:n_samples]
    phase = np.cumsum(2 * np.pi * freq_shaped / sample_rate_hz)
    return np.exp(1j * phase).astype(np.complex64)


def _generate_gmsk_chunk(
    *,
    bt: float,
    sample_rate_hz: float,
    snr_db: float,
    use_gnuradio: bool,
    rng: np.random.Generator,
    apply_channel: bool = True,
) -> np.ndarray:
    """Generate one GMSK chunk (2, 1024) at 4800 baud."""
    quad_rate = int(sample_rate_hz)
    sps = int(round(quad_rate / GMSK_SYMBOL_RATE_HZ))
    need = CHUNK_SAMPLES + sps * 8
    if use_gnuradio:
        try:
            _require_gnuradio()
            __import__("gnuradio.digital")
            iq = _gmsk_modulate_gnuradio(
                need,
                quad_rate,
                bt=bt,
                samples_per_symbol=sps,
                rng=rng,
            )
        except ImportError:
            iq = _gmsk_modulate_numpy(
                need, quad_rate, bt=bt, samples_per_symbol=sps, rng=rng
            )
    else:
        iq = _gmsk_modulate_numpy(
            need, quad_rate, bt=bt, samples_per_symbol=sps, rng=rng
        )
    if apply_channel:
        offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
        iq = add_freq_offset(iq, offset, sample_rate_hz)
        iq = add_awgn(iq, snr_db, rng=rng)
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def verify_gmsk_signal(
    chunks: np.ndarray,
    class_name: str,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
    *,
    symbol_rate_hz: float = GMSK_SYMBOL_RATE_HZ,
    bt: float = 0.5,
) -> None:
    """Verify GMSK constant envelope and h=0.5 instantaneous-frequency spread."""
    del bt  # reserved for logging; deviation uses symbol rate (h=0.5)
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    phase = np.unwrap(np.angle(iq), axis=1)
    inst_freq = np.diff(phase, axis=1) * sample_rate_hz / (2 * np.pi)
    freq_std = float(inst_freq.std())
    expected_dev = symbol_rate_hz / 4.0
    if freq_std < expected_dev * 0.3:
        raise ValueError(
            f"{class_name}: inst_freq std {freq_std:.0f} Hz too low for GMSK "
            f"(expected > {expected_dev * 0.3:.0f} Hz)"
        )
    if freq_std > expected_dev * 2.5:
        raise ValueError(
            f"{class_name}: inst_freq std {freq_std:.0f} Hz too high for GMSK"
        )
    envelope = np.abs(iq)
    env_variation = float(envelope.std() / max(envelope.mean(), 1e-12))
    if env_variation > 0.08:
        raise ValueError(
            f"{class_name}: envelope variation {env_variation:.3f} too high for GMSK"
        )


def generate_gmsk_bt05(
    n_chunks: int,
    *,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
    snr_db: float = 10.0,
    use_gnuradio: bool = True,
    rng: np.random.Generator | None = None,
    apply_channel: bool = True,
) -> np.ndarray:
    """GMSK BT=0.5 (D-Star profile), 4800 baud."""
    return generate_variant_chunks(
        "GMSK_BT05",
        n_chunks,
        snr_db=snr_db,
        sample_rate_hz=sample_rate_hz,
        use_gnuradio=use_gnuradio,
        rng=rng,
        apply_channel=apply_channel,
    )


def generate_gmsk_bt03(
    n_chunks: int,
    *,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
    snr_db: float = 10.0,
    use_gnuradio: bool = True,
    rng: np.random.Generator | None = None,
    apply_channel: bool = True,
) -> np.ndarray:
    """GMSK BT=0.3 (standard GMSK), 4800 baud."""
    return generate_variant_chunks(
        "GMSK_BT03",
        n_chunks,
        snr_db=snr_db,
        sample_rate_hz=sample_rate_hz,
        use_gnuradio=use_gnuradio,
        rng=rng,
        apply_channel=apply_channel,
    )


def _generate_nbfm_chunk(
    *,
    max_dev: float,
    tau: float,
    sample_rate_hz: float,
    audio_rate_hz: float,
    class_name: str,
    snr_db: float,
    use_gnuradio: bool,
    rng: np.random.Generator,
    apply_channel: bool = True,
    enforce_params: bool = True,
) -> np.ndarray:
    if enforce_params:
        validate_nbfm_params(tau, max_dev, class_name=class_name)
    quad_rate = int(sample_rate_hz)
    audio_rate = int(audio_rate_hz)
    interp_factor = quad_rate // audio_rate
    audio_len = int(math.ceil(CHUNK_SAMPLES / interp_factor)) + 32
    audio_high = _nbfm_audio_high_hz(max_dev)
    audio = _generate_nbfm_audio(audio_len, audio_rate, rng, high_hz=audio_high)
    if use_gnuradio and tau == 0.0:
        _require_gnuradio()
        iq = _nbfm_modulate_gnuradio(audio, audio_rate, quad_rate, max_dev)
    else:
        if tau != 0.0:
            raise ValueError(f"{class_name}: only tau=0.0 is supported for synthesis")
        audio_q = _audio_to_quad(audio, audio_rate, quad_rate)
        iq = _nbfm_modulate_numpy(audio_q, quad_rate, max_dev)
    if apply_channel:
        offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
        iq = add_freq_offset(iq, offset, sample_rate_hz)
        iq = add_awgn(iq, snr_db)
    chunk = _complex_to_iq_chunk(iq)
    return normalise_unit_power(chunk)


def _voice_audio_for_nfm_squelch(
    n_audio: int,
    audio_rate_hz: float,
    rng: np.random.Generator,
    *,
    max_dev: float = 2500.0,
) -> np.ndarray:
    """Band-limited voice-band audio for 12.5 kHz NFM squelch modes."""
    high_hz = _nbfm_audio_high_hz(max_dev)
    tone_choices = tuple(f for f in NBFM_TONE_HZ if f <= high_hz)
    if not tone_choices:
        tone_choices = (min(NBFM_TONE_HZ),)
    if rng.random() > 0.5:
        noise = rng.standard_normal(n_audio).astype(np.float32)
        sos = butter(4, [300, high_hz], btype="bandpass", fs=audio_rate_hz, output="sos")
        audio = sosfilt(sos, noise).astype(np.float32)
    else:
        t = np.arange(n_audio, dtype=np.float64) / audio_rate_hz
        freq = float(rng.choice(tone_choices))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = (audio / peak).astype(np.float32)
    return audio


def _limit_nfm_modulation_audio(
    combined: np.ndarray,
    audio_rate_hz: float,
) -> np.ndarray:
    """Low-pass baseband before FM so occupied RF BW stays within 12.5 kHz NFM."""
    cutoff = min(2500.0, audio_rate_hz * 0.45)
    sos = butter(4, cutoff, btype="low", fs=audio_rate_hz, output="sos")
    return sosfilt(sos, combined.astype(np.float32)).astype(np.float32)


def _dcs_subaudible_waveform(
    n_audio: int,
    audio_rate_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """DCS NRZ at 134.4 bit/s, low-passed so rectangular edges do not widen FM."""
    samples_per_bit = audio_rate_hz / NFM_DCS_BIT_RATE
    n_bits = int(n_audio / samples_per_bit) + 4
    dcs_bits = rng.integers(0, 2, n_bits)
    dcs_nrz = (dcs_bits * 2 - 1).astype(np.float32)
    spb = max(int(samples_per_bit), 1)
    dcs = np.repeat(dcs_nrz, spb)[:n_audio]
    cutoff = min(280.0, audio_rate_hz * 0.45)
    sos = butter(4, cutoff, btype="low", fs=audio_rate_hz, output="sos")
    return sosfilt(sos, dcs).astype(np.float32)


def _fm_modulate_combined_audio(
    combined: np.ndarray,
    *,
    max_dev: float,
    sample_rate_hz: float,
    audio_rate_hz: float,
    use_gnuradio: bool,
) -> np.ndarray:
    quad_rate = int(sample_rate_hz)
    audio_rate = int(audio_rate_hz)
    combined_q = _audio_to_quad(combined.astype(np.float32), audio_rate, quad_rate)
    if use_gnuradio:
        _require_gnuradio()
        return _nbfm_modulate_gnuradio(combined_q, audio_rate, quad_rate, max_dev)
    return _nbfm_modulate_numpy(combined_q, quad_rate, max_dev)


def _generate_nfm_ctcss_chunk(
    *,
    sample_rate_hz: float,
    audio_rate_hz: float,
    snr_db: float,
    use_gnuradio: bool,
    rng: np.random.Generator,
    apply_channel: bool = True,
) -> np.ndarray:
    max_dev = 2500.0
    quad_rate = int(sample_rate_hz)
    audio_rate = int(audio_rate_hz)
    interp_factor = quad_rate // audio_rate
    n_audio = int(math.ceil(CHUNK_SAMPLES / interp_factor)) + 32
    ctcss_freq = float(rng.choice(CTCSS_TONES_HZ))
    audio = _voice_audio_for_nfm_squelch(n_audio, float(audio_rate), rng)
    t_audio = np.arange(n_audio, dtype=np.float64) / audio_rate
    ctcss = np.sin(2 * np.pi * ctcss_freq * t_audio).astype(np.float32)
    voice_scale = NFM_CTCSS_VOICE_DEV_HZ / max_dev
    tone_scale = NFM_CTCSS_TONE_DEV_HZ / max_dev
    combined = voice_scale * audio + tone_scale * ctcss
    combined = _limit_nfm_modulation_audio(combined, float(audio_rate))
    iq = _fm_modulate_combined_audio(
        combined,
        max_dev=max_dev,
        sample_rate_hz=sample_rate_hz,
        audio_rate_hz=audio_rate_hz,
        use_gnuradio=use_gnuradio,
    )
    if apply_channel:
        offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
        iq = add_freq_offset(iq, offset, sample_rate_hz)
        iq = add_awgn(iq, snr_db, rng=rng)
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def _generate_nfm_dcs_chunk(
    *,
    sample_rate_hz: float,
    audio_rate_hz: float,
    snr_db: float,
    use_gnuradio: bool,
    rng: np.random.Generator,
    apply_channel: bool = True,
) -> np.ndarray:
    max_dev = 2500.0
    audio_rate = int(audio_rate_hz)
    interp_factor = int(sample_rate_hz) // audio_rate
    n_audio = int(math.ceil(CHUNK_SAMPLES / interp_factor)) + 32
    voice_scale = NFM_DCS_VOICE_DEV_HZ / max_dev
    dcs_scale = NFM_DCS_SHIFT_HZ / max_dev
    audio = _voice_audio_for_nfm_squelch(n_audio, float(audio_rate), rng, max_dev=max_dev)
    dcs = _dcs_subaudible_waveform(n_audio, float(audio_rate), rng)
    combined = (voice_scale * audio + dcs_scale * dcs).astype(np.float32)
    combined = _limit_nfm_modulation_audio(combined, float(audio_rate))
    iq = _fm_modulate_combined_audio(
        combined,
        max_dev=max_dev,
        sample_rate_hz=sample_rate_hz,
        audio_rate_hz=audio_rate_hz,
        use_gnuradio=use_gnuradio,
    )
    if apply_channel:
        offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
        iq = add_freq_offset(iq, offset, sample_rate_hz)
        iq = add_awgn(iq, snr_db, rng=rng)
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def _nrzi_encode(bits: np.ndarray) -> np.ndarray:
    """NRZI: transition on 0, hold on 1."""
    out = np.zeros(len(bits), dtype=np.int8)
    prev = 0
    for i, b in enumerate(bits):
        if b == 0:
            prev = 1 - prev
        out[i] = prev
    return out


def _generate_bell202_chunk(
    *,
    sample_rate_hz: float,
    snr_db: float,
    rng: np.random.Generator,
    apply_channel: bool = True,
) -> np.ndarray:
    samples_per_bit = int(sample_rate_hz / BELL202_BAUD)
    n_bits = CHUNK_SAMPLES // samples_per_bit + 4
    raw_bits = rng.integers(0, 2, n_bits)
    nrzi = _nrzi_encode(raw_bits)
    freqs = np.where(nrzi == 1, BELL202_MARK_HZ, BELL202_SPACE_HZ)
    phase = 0.0
    afsk_audio = np.zeros(CHUNK_SAMPLES, dtype=np.float64)
    sample_idx = 0
    for bit_freq in freqs:
        if sample_idx >= CHUNK_SAMPLES:
            break
        end_idx = min(sample_idx + samples_per_bit, CHUNK_SAMPLES)
        for i in range(sample_idx, end_idx):
            afsk_audio[i] = np.sin(phase)
            phase += 2 * np.pi * bit_freq / sample_rate_hz
        sample_idx = end_idx
    mod_scale = BELL202_FM_DEV_HZ / sample_rate_hz
    rf_phase = np.cumsum(2 * np.pi * afsk_audio * mod_scale)
    iq = np.exp(1j * rf_phase).astype(np.complex64)[:CHUNK_SAMPLES]
    if apply_channel:
        offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
        iq = add_freq_offset(iq, offset, sample_rate_hz)
        iq = add_awgn(iq, snr_db, rng=rng)
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def _generate_g3ruh_chunk(
    *,
    sample_rate_hz: float,
    snr_db: float,
    rng: np.random.Generator,
    apply_channel: bool = True,
) -> np.ndarray:
    samples_per_symbol = int(round(sample_rate_hz / G3RUH_BAUD))
    filt = _gaussian_filter(0.5, 4, samples_per_symbol)
    n_symbols = CHUNK_SAMPLES // samples_per_symbol + len(filt)
    symbols = rng.integers(0, 2, n_symbols)
    symbol_map = {0: G3RUH_DEV_HZ, 1: -G3RUH_DEV_HZ}
    freq_sequence = np.array([symbol_map[int(s)] for s in symbols], dtype=np.float64)
    freq_up = np.repeat(freq_sequence, samples_per_symbol)
    freq_shaped = np.convolve(freq_up, filt, mode="same")
    if apply_channel:
        freq_shaped += float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
    phase = np.cumsum(2 * np.pi * freq_shaped / sample_rate_hz)
    iq = np.exp(1j * phase).astype(np.complex64)[:CHUNK_SAMPLES]
    if apply_channel:
        iq = add_awgn(iq, snr_db, rng=rng)
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def _fm_demod_audio(
    chunks: np.ndarray,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
) -> np.ndarray:
    """FM demodulate chunks to instantaneous frequency (audio) per row."""
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    phase = np.unwrap(np.angle(iq), axis=1)
    return np.diff(phase, axis=1) * sample_rate_hz / (2 * np.pi)


def _envelope_variation(chunks: np.ndarray) -> float:
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    envelope = np.abs(iq)
    return float(envelope.std() / max(envelope.mean(), 1e-12))


def verify_nfm_ctcss(
    chunks: np.ndarray,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
) -> None:
    """Confirm subaudible CTCSS energy and constant FM envelope."""
    if _envelope_variation(chunks) > 0.05:
        msg = f"NFM_CTCSS envelope variation {_envelope_variation(chunks):.3f} too high"
        raise ValueError(msg)
    audio = _fm_demod_audio(chunks, sample_rate_hz)
    spec = np.mean(np.abs(np.fft.rfft(audio, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[1], 1.0 / sample_rate_hz)
    sub_mask = (freqs >= 60.0) & (freqs <= 260.0)
    if not np.any(sub_mask):
        raise ValueError("NFM_CTCSS: no subaudible FFT bins")
    sub_peak = float(np.max(spec[sub_mask]))
    wide_peak = float(np.max(spec))
    if sub_peak < wide_peak * 0.02:
        raise ValueError("NFM_CTCSS: no subaudible tone peak in 67-255 Hz band")


def verify_nfm_dcs(
    chunks: np.ndarray,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
) -> None:
    """Confirm low-rate DCS energy near DC and constant FM envelope."""
    if _envelope_variation(chunks) > 0.05:
        msg = f"NFM_DCS envelope variation {_envelope_variation(chunks):.3f} too high"
        raise ValueError(msg)
    audio = _fm_demod_audio(chunks, sample_rate_hz)
    spec = np.mean(np.abs(np.fft.rfft(audio, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[1], 1.0 / sample_rate_hz)
    low_mask = (freqs >= 20.0) & (freqs <= 250.0)
    if float(np.max(spec[low_mask])) < float(np.max(spec)) * 0.005:
        raise ValueError("NFM_DCS: no low-frequency DCS energy")


def verify_bell202(
    chunks: np.ndarray,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
) -> None:
    """Verify Bell 202 mark/space tones after FM demodulation."""
    if _envelope_variation(chunks) > 0.05:
        raise ValueError("BELL202: envelope variation too high for FM")
    inst_freq = _fm_demod_audio(chunks, sample_rate_hz)
    audio_fft = np.mean(np.abs(np.fft.rfft(inst_freq, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(inst_freq.shape[1], 1.0 / sample_rate_hz)
    bin_1200 = int(np.argmin(np.abs(freqs - BELL202_MARK_HZ)))
    bin_2200 = int(np.argmin(np.abs(freqs - BELL202_SPACE_HZ)))
    bin_dc = int(np.argmin(np.abs(freqs - 100.0)))
    if audio_fft[bin_1200] < audio_fft[bin_dc] * 5:
        raise ValueError("BELL202: 1200 Hz mark tone not dominant")
    if audio_fft[bin_2200] < audio_fft[bin_dc] * 5:
        raise ValueError("BELL202: 2200 Hz space tone not dominant")


def verify_g3ruh(
    chunks: np.ndarray,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
) -> None:
    """Verify G3RUH 9600 baud FSK modulation bandwidth."""
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    phase = np.unwrap(np.angle(iq), axis=1)
    inst_freq = np.diff(phase, axis=1) * sample_rate_hz / (2 * np.pi)
    freq_std = float(inst_freq.std())
    if freq_std < 800.0 or freq_std > 4500.0:
        raise ValueError(f"G3RUH inst_freq std {freq_std:.0f} Hz out of range")
    if _envelope_variation(chunks) > 0.15:
        raise ValueError("G3RUH: envelope variation too high")


def verify_p25(
    chunks: np.ndarray,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
) -> None:
    """Verify P25 C4FM deviation and envelope."""
    verify_4fsk_signal(chunks, (600.0, 1800.0), "P25", sample_rate_hz)
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    phase = np.unwrap(np.angle(iq), axis=1)
    inst_freq = np.diff(phase, axis=1) * sample_rate_hz / (2 * np.pi)
    freq_std = float(inst_freq.std())
    if freq_std < 500.0 or freq_std > 1500.0:
        raise ValueError(f"P25 inst_freq std {freq_std:.0f} Hz out of range")


def generate_aviation_am_25k(
    n_chunks: int,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
    audio_rate_hz: float = DEFAULT_AUDIO_RATE_HZ,
    modulation_index: float = DEFAULT_MODULATION_INDEX,
    *,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    apply_channel: bool = True,
) -> np.ndarray:
    """Generate AM_AIR_25K chunks shape (n_chunks, 2, 1024)."""
    return _generate_aviation_am_chunks(
        n_chunks,
        variant="25k",
        sample_rate_hz=sample_rate_hz,
        audio_rate_hz=audio_rate_hz,
        modulation_index=modulation_index,
        snr_db=snr_db,
        rng=rng,
        apply_channel=apply_channel,
    )


def generate_aviation_am_833(
    n_chunks: int,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
    audio_rate_hz: float = DEFAULT_AUDIO_RATE_HZ,
    modulation_index: float = DEFAULT_MODULATION_INDEX,
    *,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    apply_channel: bool = True,
) -> np.ndarray:
    """Generate AM_AIR_833 chunks shape (n_chunks, 2, 1024)."""
    return _generate_aviation_am_chunks(
        n_chunks,
        variant="833",
        sample_rate_hz=sample_rate_hz,
        audio_rate_hz=audio_rate_hz,
        modulation_index=modulation_index,
        snr_db=snr_db,
        rng=rng,
        apply_channel=apply_channel,
    )


def _generate_aviation_am_chunks(
    n_chunks: int,
    *,
    variant: Literal["25k", "833"],
    sample_rate_hz: float,
    audio_rate_hz: float,
    modulation_index: float,
    snr_db: float | None,
    rng: np.random.Generator | None,
    apply_channel: bool = True,
) -> np.ndarray:
    gen = rng or np.random.default_rng()
    quad_rate = int(sample_rate_hz)
    audio_rate = int(audio_rate_hz)
    factor = quad_rate // audio_rate
    audio_len = int(math.ceil(CHUNK_SAMPLES / factor)) + 32
    chunks = np.zeros((n_chunks, 2, CHUNK_SAMPLES), dtype=np.float32)
    for i in range(n_chunks):
        audio = _generate_aviation_audio(audio_len, audio_rate, variant, gen)
        audio_q = _audio_to_quad(audio, audio_rate, quad_rate)
        iq = _dsb_am_full_carrier(audio_q, modulation_index, sample_rate_hz)
        if apply_channel:
            level_snr = float(snr_db) if snr_db is not None else 10.0
            offset = float(gen.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
            iq = add_freq_offset(iq, offset, sample_rate_hz)
            iq = add_awgn(iq, level_snr)
        chunks[i] = normalise_unit_power(_complex_to_iq_chunk(iq))
    return chunks


def _raised_cosine_filter(
    alpha: float,
    span: int,
    samples_per_symbol: int,
) -> np.ndarray:
    """Raised cosine FIR filter coefficients."""
    n = span * samples_per_symbol
    t = np.arange(-n // 2, n // 2 + 1, dtype=np.float64) / samples_per_symbol
    eps = 1e-8
    h = np.zeros(len(t), dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < eps:
            h[i] = 1.0
        elif abs(abs(2 * alpha * ti) - 1.0) < eps:
            h[i] = (np.pi / 4) * np.sinc(1 / (2 * alpha))
        else:
            h[i] = (
                np.sinc(ti)
                * np.cos(np.pi * alpha * ti)
                / (1 - (2 * alpha * ti) ** 2)
            )
    return (h / h.sum()).astype(np.float64)


def _root_raised_cosine_filter(
    beta: float,
    span: int,
    samples_per_symbol: int,
) -> np.ndarray:
    """Root raised cosine FIR filter coefficients."""
    n = span * samples_per_symbol
    t = np.arange(-n // 2, n // 2 + 1, dtype=np.float64) / samples_per_symbol
    eps = 1e-8
    h = np.zeros(len(t), dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < eps:
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif abs(abs(4 * beta * ti) - 1.0) < eps:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            h[i] = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return (h / h.sum()).astype(np.float64)


def _gaussian_filter(
    bt: float,
    span: int,
    samples_per_symbol: int,
) -> np.ndarray:
    """Gaussian FIR filter coefficients for GMSK/C4FM."""
    n = span * samples_per_symbol
    t = np.arange(-n // 2, n // 2 + 1, dtype=np.float64) / samples_per_symbol
    c = np.sqrt(2 * np.pi**2 / np.log(2))
    h = np.exp(-0.5 * (c * bt * t) ** 2)
    return (h / h.sum()).astype(np.float64)


def _fsk4_pulse_filter(proto: Fsk4ProtocolSpec, samples_per_symbol: int) -> np.ndarray:
    if proto.filter_type == "rc":
        return _raised_cosine_filter(
            proto.filter_param, proto.filter_span, samples_per_symbol
        )
    if proto.filter_type == "rrc":
        return _root_raised_cosine_filter(
            proto.filter_param, proto.filter_span, samples_per_symbol
        )
    return _gaussian_filter(proto.filter_param, proto.filter_span, samples_per_symbol)


def _generate_4fsk_chunk(
    n_samples: int,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    symbol_map: dict[int, float],
    pulse_filter: np.ndarray,
    snr_db: float,
    *,
    freq_offset_hz: float = 0.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one chunk of 4FSK IQ at baseband (complex64, length n_samples)."""
    samples_per_symbol = int(round(sample_rate_hz / symbol_rate_hz))
    n_symbols = (n_samples // samples_per_symbol) + len(pulse_filter)

    symbols = rng.integers(0, 4, n_symbols)
    freq_sequence = np.array([symbol_map[int(s)] for s in symbols], dtype=np.float64)
    freq_upsampled = np.repeat(freq_sequence, samples_per_symbol)
    freq_shaped = np.convolve(freq_upsampled, pulse_filter, mode="same")
    freq_shaped += freq_offset_hz

    phase = np.cumsum(2 * np.pi * freq_shaped / sample_rate_hz)
    iq = np.exp(1j * phase).astype(np.complex64)
    iq = iq[:n_samples]
    return add_awgn(iq, snr_db, rng=rng)


def verify_4fsk_signal(
    chunks: np.ndarray,
    expected_deviations: tuple[float, float],
    mode_name: str,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
) -> None:
    """Verify 4FSK instantaneous frequency and constant envelope."""
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    phase = np.unwrap(np.angle(iq), axis=1)
    inst_freq = np.diff(phase, axis=1) * sample_rate_hz / (2 * np.pi)

    freq_std = float(inst_freq.std())
    inner_dev, outer_dev = expected_deviations

    if freq_std < inner_dev * 0.3:
        msg = (
            f"{mode_name}: inst_freq std {freq_std:.0f} Hz too low — "
            f"signal may be unmodulated. Expected > {inner_dev * 0.3:.0f} Hz"
        )
        raise ValueError(msg)
    if freq_std > outer_dev * 3.0:
        msg = (
            f"{mode_name}: inst_freq std {freq_std:.0f} Hz too high — "
            f"check deviation parameters. Expected < {outer_dev * 3.0:.0f} Hz"
        )
        raise ValueError(msg)

    envelope = np.abs(iq)
    env_variation = float(envelope.std() / max(envelope.mean(), 1e-12))
    if env_variation > 0.15:
        msg = (
            f"{mode_name}: envelope variation {env_variation:.3f} too high "
            f"for FSK — signal has unexpected AM component"
        )
        raise ValueError(msg)


def _generate_fsk4_protocol_chunk(
    class_name: str,
    *,
    sample_rate_hz: float,
    snr_db: float,
    rng: np.random.Generator,
    apply_channel: bool,
) -> np.ndarray:
    proto = FSK4_PROTOCOL_SPECS[class_name]
    samples_per_symbol = int(round(sample_rate_hz / proto.symbol_rate_hz))
    pulse_filter = _fsk4_pulse_filter(proto, samples_per_symbol)
    freq_offset = 0.0
    level_snr = snr_db
    if apply_channel:
        freq_offset = float(rng.uniform(-DEFAULT_FREQ_OFFSET_HZ, DEFAULT_FREQ_OFFSET_HZ))
    iq = _generate_4fsk_chunk(
        CHUNK_SAMPLES,
        sample_rate_hz,
        proto.symbol_rate_hz,
        proto.symbol_map,
        pulse_filter,
        level_snr if apply_channel else 40.0,
        freq_offset_hz=freq_offset,
        rng=rng,
    )
    return normalise_unit_power(_complex_to_iq_chunk(iq))


def generate_variant_chunks(
    class_name: str,
    n_chunks: int,
    *,
    snr_db: float = 10.0,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
    audio_rate_hz: float = DEFAULT_AUDIO_RATE_HZ,
    tau: float = 0.0,
    max_dev: float | None = None,
    modulation_index: float = DEFAULT_MODULATION_INDEX,
    use_gnuradio: bool = True,
    rng: np.random.Generator | None = None,
    apply_channel: bool = True,
    enforce_params: bool = True,
) -> np.ndarray:
    """Generate n_chunks for one modulation class."""
    spec = VARIANT_SPECS[class_name]
    gen = rng or np.random.default_rng()
    chunks = np.zeros((n_chunks, 2, CHUNK_SAMPLES), dtype=np.float32)

    if spec.kind == "nbfm":
        dev = max_dev if max_dev is not None else spec.max_dev
        if dev is None:
            raise ValueError(f"{class_name}: max_dev required")
        if enforce_params:
            validate_nbfm_params(tau, dev, class_name=class_name)
        for i in range(n_chunks):
            chunks[i] = _generate_nbfm_chunk(
                max_dev=dev,
                tau=tau,
                sample_rate_hz=sample_rate_hz,
                audio_rate_hz=audio_rate_hz,
                class_name=class_name,
                snr_db=snr_db,
                use_gnuradio=use_gnuradio,
                rng=gen,
                apply_channel=apply_channel,
                enforce_params=enforce_params,
            )
    elif spec.kind == "am_air" and spec.am_variant is not None:
        if spec.am_variant == "25k":
            chunks = _generate_aviation_am_chunks(
                n_chunks,
                variant="25k",
                sample_rate_hz=sample_rate_hz,
                audio_rate_hz=audio_rate_hz,
                modulation_index=modulation_index,
                snr_db=snr_db,
                rng=gen,
                apply_channel=apply_channel,
            )
        else:
            chunks = _generate_aviation_am_chunks(
                n_chunks,
                variant="833",
                sample_rate_hz=sample_rate_hz,
                audio_rate_hz=audio_rate_hz,
                modulation_index=modulation_index,
                snr_db=snr_db,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "wbfm":
        dev = max_dev if max_dev is not None else spec.max_dev
        if dev is None:
            raise ValueError(f"{class_name}: max_dev required for WBFM")
        for i in range(n_chunks):
            chunks[i] = _generate_wbfm_chunk(
                max_dev=dev,
                sample_rate_hz=sample_rate_hz,
                snr_db=snr_db,
                use_gnuradio=use_gnuradio,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "psk":
        order = spec.psk_order
        if order is None:
            raise ValueError(f"{class_name}: psk_order required")
        for i in range(n_chunks):
            chunks[i] = _generate_psk_chunk(
                order=order,
                sample_rate_hz=sample_rate_hz,
                snr_db=snr_db,
                use_gnuradio=use_gnuradio,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "fsk4":
        if class_name not in FSK4_PROTOCOL_SPECS:
            raise ValueError(f"{class_name}: missing FSK4 protocol spec")
        for i in range(n_chunks):
            chunks[i] = _generate_fsk4_protocol_chunk(
                class_name,
                sample_rate_hz=sample_rate_hz,
                snr_db=snr_db,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "nbfm_ctcss":
        for i in range(n_chunks):
            chunks[i] = _generate_nfm_ctcss_chunk(
                sample_rate_hz=sample_rate_hz,
                audio_rate_hz=audio_rate_hz,
                snr_db=snr_db,
                use_gnuradio=use_gnuradio,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "nbfm_dcs":
        for i in range(n_chunks):
            chunks[i] = _generate_nfm_dcs_chunk(
                sample_rate_hz=sample_rate_hz,
                audio_rate_hz=audio_rate_hz,
                snr_db=snr_db,
                use_gnuradio=use_gnuradio,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "afsk":
        for i in range(n_chunks):
            chunks[i] = _generate_bell202_chunk(
                sample_rate_hz=sample_rate_hz,
                snr_db=snr_db,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "fsk2":
        for i in range(n_chunks):
            chunks[i] = _generate_g3ruh_chunk(
                sample_rate_hz=sample_rate_hz,
                snr_db=snr_db,
                rng=gen,
                apply_channel=apply_channel,
            )
    elif spec.kind == "gmsk":
        bt = spec.gmsk_bt
        if bt is None:
            raise ValueError(f"{class_name}: gmsk_bt required")
        for i in range(n_chunks):
            chunks[i] = _generate_gmsk_chunk(
                bt=bt,
                sample_rate_hz=sample_rate_hz,
                snr_db=snr_db,
                use_gnuradio=use_gnuradio,
                rng=gen,
                apply_channel=apply_channel,
            )
    else:
        raise ValueError(f"Unknown variant kind for {class_name}")
    return chunks


def generate_synthetic(
    modes: list[SyntheticMode] | None = None,
    *,
    chunks_per_snr: int = DEFAULT_CHUNKS_PER_SNR,
    sample_rate_hz: float = DEFAULT_QUAD_RATE_HZ,
    audio_rate_hz: float = DEFAULT_AUDIO_RATE_HZ,
    snr_levels: list[float] | None = None,
    verify: bool = True,
    use_gnuradio: bool = True,
    seed: int | None = None,
) -> IQDataset:
    """Generate synthetic IQ dataset for selected modes and SNR grid."""
    selected = modes or list(ALL_MODES)
    class_names = [MODE_TO_CLASS[m] for m in selected]
    levels = snr_levels if snr_levels is not None else SNR_DB_LEVELS
    rng = np.random.default_rng(seed)

    samples_list: list[np.ndarray] = []
    labels_list: list[int] = []
    snr_list: list[float] = []

    for mode in selected:
        class_name = MODE_TO_CLASS[mode]
        spec = VARIANT_SPECS[class_name]
        for snr in levels:
            chunks = generate_variant_chunks(
                class_name,
                chunks_per_snr,
                snr_db=float(snr),
                sample_rate_hz=sample_rate_hz,
                audio_rate_hz=audio_rate_hz,
                use_gnuradio=use_gnuradio,
                rng=rng,
            )
            if verify:
                # Bandwidth / 4FSK checks need clean IQ (no AWGN or freq offset).
                # Using loop SNR with apply_channel=True makes noise fill the band.
                ref_chunks = generate_variant_chunks(
                    class_name,
                    min(100, chunks_per_snr),
                    snr_db=20.0,
                    sample_rate_hz=sample_rate_hz,
                    audio_rate_hz=audio_rate_hz,
                    use_gnuradio=use_gnuradio,
                    rng=rng,
                    apply_channel=False,
                )
                # Occupied-bandwidth (-26 dB) is meaningful for analog FM/AM and
                # narrow protocol FSK; not for wideband PSK or 9600 baud G3RUH.
                if spec.kind not in ("psk", "fsk2", "gmsk"):
                    verify_bandwidth(
                        ref_chunks,
                        sample_rate_hz,
                        spec.max_bandwidth_hz,
                        class_name,
                    )
                if spec.kind == "fsk4":
                    proto = FSK4_PROTOCOL_SPECS[class_name]
                    verify_4fsk_signal(
                        ref_chunks,
                        (proto.inner_dev_hz, proto.outer_dev_hz),
                        class_name,
                        sample_rate_hz,
                    )
                    if class_name == "P25":
                        verify_p25(ref_chunks, sample_rate_hz)
                elif spec.kind == "nbfm_ctcss":
                    verify_nfm_ctcss(ref_chunks, sample_rate_hz)
                elif spec.kind == "nbfm_dcs":
                    verify_nfm_dcs(ref_chunks, sample_rate_hz)
                elif spec.kind == "afsk":
                    verify_bell202(ref_chunks, sample_rate_hz)
                elif spec.kind == "fsk2":
                    verify_g3ruh(ref_chunks, sample_rate_hz)
                elif spec.kind == "gmsk":
                    bt_val = spec.gmsk_bt if spec.gmsk_bt is not None else 0.5
                    verify_gmsk_signal(
                        ref_chunks,
                        class_name,
                        sample_rate_hz,
                        bt=bt_val,
                    )
            label = class_names.index(class_name)
            for i in range(chunks_per_snr):
                labels_list.append(label)
                snr_list.append(float(snr))
            samples_list.append(chunks)

    samples = np.concatenate(samples_list, axis=0)
    labels = np.array(labels_list, dtype=np.int32)
    snr_db_arr = np.array(snr_list, dtype=np.float32)
    return IQDataset(
        samples=samples,
        labels=labels,
        snr_db=snr_db_arr,
        class_names=class_names,
        source="synthetic",
    )


def save_synthetic_dataset(output_dir: Path, dataset: IQDataset) -> Path:
    """Write synthetic.npz and meta.json under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "synthetic.npz"
    np.savez_compressed(
        npz_path,
        samples=dataset.samples,
        labels=dataset.labels,
        snr_db=dataset.snr_db,
        class_names=np.array(dataset.class_names, dtype=object),
        source=dataset.source,
    )
    meta = {
        "source": dataset.source,
        "class_names": dataset.class_names,
        "num_samples": int(len(dataset.labels)),
        "snr_min": float(dataset.snr_db.min()) if len(dataset.snr_db) else None,
        "snr_max": float(dataset.snr_db.max()) if len(dataset.snr_db) else None,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d samples)", npz_path, len(dataset.labels))
    return npz_path


def load_synthetic(path: Path | str) -> IQDataset:
    """Load IQDataset from synthetic.npz file or directory containing it."""
    path = Path(path)
    if path.is_dir():
        npz_path = path / "synthetic.npz"
    else:
        npz_path = path
    if not npz_path.is_file():
        msg = f"Synthetic dataset not found: {npz_path}"
        raise FileNotFoundError(msg)
    data = np.load(npz_path, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    return IQDataset(
        samples=data["samples"].astype(np.float32),
        labels=data["labels"].astype(np.int32),
        snr_db=data["snr_db"].astype(np.float32),
        class_names=class_names,
        source=str(data.get("source", "synthetic")),
    )


def measure_audio_bandwidth_hz(
    audio: np.ndarray,
    sample_rate_hz: float,
    threshold_db: float = 40.0,
) -> float:
    """Estimate audio bandwidth from PSD (-threshold_db from peak)."""
    spec = np.abs(np.fft.rfft(audio)) ** 2
    spec_db = 10 * np.log10(spec + 1e-12)
    peak = float(np.max(spec_db))
    mask = spec_db >= peak - threshold_db
    bins = np.where(mask)[0]
    if bins.size == 0:
        return 0.0
    return float(bins[-1] - bins[0]) * sample_rate_hz / (2 * len(audio))


def aviation_carrier_to_sideband_ratio(chunk: np.ndarray) -> float:
    """
    Ratio of carrier (DC) power to AC sideband power for DSB-AM baseband chunk.

    Values >> 1 indicate full carrier (not suppressed carrier).
    """
    env = chunk[0].astype(np.float64)
    carrier_power = float(np.mean(env) ** 2)
    ac = env - np.mean(env)
    sideband_power = float(np.mean(ac**2))
    return carrier_power / max(sideband_power, 1e-12)
