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

IMPORTANT: This module uses GNU Radio built-in blocks and numpy/scipy only.
No OOT (out-of-tree) modules are used. This is intentional —
OOT modules under validation cannot be used to generate reference
data without making the validation circular.

Specifically, the qradiolink OOT module is NOT used here even
though it is installed. Its blocks are the subject of validation,
not a source of ground truth.

Verified sources used:
  - gnuradio.analog.frequency_modulator_fc, gnuradio.filter (NBFM chain)
  - numpy / scipy for direct DSP (aviation AM)

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
]

MODE_TO_CLASS: dict[SyntheticMode, str] = {
    "nbfm25": "NBFM_25",
    "nbfm50": "NBFM_50",
    "am_air_25k": "AM_AIR_25K",
    "am_air_833": "AM_AIR_833",
    "wbfm": "WBFM",
    "bpsk": "BPSK",
    "qpsk": "QPSK",
}

ALL_MODES: tuple[SyntheticMode, ...] = (
    "nbfm25",
    "nbfm50",
    "am_air_25k",
    "am_air_833",
    "wbfm",
    "bpsk",
    "qpsk",
)

DEFAULT_PSK_SAMPLES_PER_SYMBOL = 8

@dataclass(frozen=True)
class VariantSpec:
    """Per-modulation generation and verification limits."""

    class_name: str
    max_bandwidth_hz: float
    kind: Literal["nbfm", "am_air", "wbfm", "psk"]
    max_dev: float | None = None
    am_variant: Literal["25k", "833"] | None = None
    psk_order: int | None = None


VARIANT_SPECS: dict[str, VariantSpec] = {
    "NBFM_25": VariantSpec("NBFM_25", 7000.0, "nbfm", max_dev=2500.0),
    "NBFM_50": VariantSpec("NBFM_50", 13000.0, "nbfm", max_dev=5000.0),
    "AM_AIR_25K": VariantSpec("AM_AIR_25K", 8000.0, "am_air", am_variant="25k"),
    "AM_AIR_833": VariantSpec("AM_AIR_833", 6500.0, "am_air", am_variant="833"),
    "WBFM": VariantSpec("WBFM", 200000.0, "wbfm", max_dev=75000.0),
    "BPSK": VariantSpec("BPSK", 12000.0, "psk", psk_order=2),
    "QPSK": VariantSpec("QPSK", 12000.0, "psk", psk_order=4),
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


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add complex AWGN at the given SNR (dB) relative to signal power."""
    signal_power = float(np.mean(np.abs(signal) ** 2))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
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
                ref_chunks = generate_variant_chunks(
                    class_name,
                    min(100, chunks_per_snr),
                    snr_db=float(snr),
                    sample_rate_hz=sample_rate_hz,
                    audio_rate_hz=audio_rate_hz,
                    use_gnuradio=use_gnuradio,
                    rng=rng,
                    apply_channel=False,
                )
                verify_bandwidth(
                    ref_chunks,
                    sample_rate_hz,
                    spec.max_bandwidth_hz,
                    class_name,
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


def load_synthetic(path: Path) -> IQDataset:
    """Load IQDataset from synthetic.npz file or directory containing it."""
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
