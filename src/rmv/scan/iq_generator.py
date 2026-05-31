"""Generate reference IQ from GNU Radio built-ins or numpy (never OOT project blocks)."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, lfilter

from rmv.dataset.preprocess import normalise_unit_power
from rmv.dataset.synthetic import _generate_nbfm_chunk, _generate_psk_chunk
from rmv.scan.class_vocab import ClassifierVocab, resolve_classifier_labels
from rmv.scan.discover import GRProject
from rmv.scan.exclusions import mode_exclusion_reason
from rmv.scan.mode_table import ModeSpec, lookup_mode
from rmv.scan.readme_parser import ReadmeSummary
from rmv.plugins.sleipnir_8qpsk import (
    DEFAULT_SAMPLE_RATE_HZ,
    EXPECTED_CARRIER_HZ,
    _rrc_taps,
)

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 1024
SAMPLE_RATE_HZ = 48000.0
N_CHUNKS = 16
# Symmetric tone spacing for M-FSK (Hz); matches training-data frequency deviation scale.
FSK_DEVIATION_HZ = 500.0
FSK_SAMPLES_PER_SYMBOL = 8
# MSK/GMSK reference: keep peak frequency deviation near FSK_DEVIATION_HZ so the
# family CNN sees discrete CPFSK rather than wideband FM (sps=8 was ~1500 Hz dev).
GMSK_SAMPLES_PER_SYMBOL = max(
    FSK_SAMPLES_PER_SYMBOL,
    int(SAMPLE_RATE_HZ / (2.0 * FSK_DEVIATION_HZ)),
)


@dataclass
class GeneratedIQ:
    iq_path: Path
    sidecar_path: Path
    block_name: str
    expected_family: str
    expected_order: str
    generation_method: str
    gr_env_used: str
    mode_name: str
    spec_note: str = ""
    protocol_only: bool = False
    skipped: bool = False
    skip_reason: str = ""


def _write_iq_and_sidecar(
    output_dir: Path,
    block_name: str,
    chunks: np.ndarray,
    *,
    expected_family: str,
    expected_order: str,
    project_name: str,
    generation_method: str,
    gr_env_used: str,
    notes: str,
) -> GeneratedIQ:
    out_dir = output_dir / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    iq_path = out_dir / f"{block_name}.iq"
    parts: list[np.ndarray] = []
    for i in range(chunks.shape[0]):
        c = chunks[i]
        parts.append(np.stack([c[0], c[1]], axis=1).reshape(-1))
    flat = np.concatenate(parts).astype("<f4")
    flat.tofile(iq_path)
    sidecar_path = out_dir / f"{block_name}.json"
    sidecar = {
        "source": project_name,
        "block_name": block_name,
        "expected_family": expected_family,
        "expected_order": expected_order,
        "sample_rate_hz": int(SAMPLE_RATE_HZ),
        "center_freq_hz": 0,
        "snr_db": None,
        "notes": notes,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return GeneratedIQ(
        iq_path=iq_path,
        sidecar_path=sidecar_path,
        block_name=block_name,
        expected_family=expected_family,
        expected_order=expected_order,
        generation_method=generation_method,
        gr_env_used=gr_env_used,
        mode_name=block_name.replace("mod_", ""),
        spec_note=notes,
    )


def _chunks_from_complex(signal: np.ndarray, n_chunks: int = N_CHUNKS) -> np.ndarray:
    need = n_chunks * CHUNK_SAMPLES
    if len(signal) < need:
        reps = int(np.ceil(need / max(len(signal), 1)))
        signal = np.tile(signal, reps)[:need]
    signal = signal[:need]
    chunks = np.zeros((n_chunks, 2, CHUNK_SAMPLES), dtype=np.float32)
    for i in range(n_chunks):
        seg = signal[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        chunks[i] = normalise_unit_power(
            np.stack([seg.real, seg.imag], axis=0).astype(np.float32)
        )
    return chunks


def _gen_psk_scan_chunks(order: int, *, use_gnuradio: bool) -> np.ndarray:
    """BPSK/QPSK/8PSK via synthetic generator (proper I/Q, not real-only)."""
    rng = np.random.default_rng(42)
    chunks = np.zeros((N_CHUNKS, 2, CHUNK_SAMPLES), dtype=np.float32)
    for i in range(N_CHUNKS):
        chunks[i] = _generate_psk_chunk(
            order=order,
            sample_rate_hz=SAMPLE_RATE_HZ,
            snr_db=20.0,
            use_gnuradio=use_gnuradio,
            rng=rng,
            apply_channel=True,
        )
    return chunks


def _gen_nbfm_scan_chunks(*, use_gnuradio: bool) -> np.ndarray:
    """NBFM_25 chunks via the same path as rmv.dataset.synthetic (verified)."""
    rng = np.random.default_rng(42)
    chunks = np.zeros((N_CHUNKS, 2, CHUNK_SAMPLES), dtype=np.float32)
    for i in range(N_CHUNKS):
        chunks[i] = _generate_nbfm_chunk(
            max_dev=2500.0,
            tau=0.0,
            sample_rate_hz=SAMPLE_RATE_HZ,
            audio_rate_hz=8000,
            class_name="NBFM_25",
            snr_db=20.0,
            use_gnuradio=use_gnuradio,
            rng=rng,
            apply_channel=True,
        )
    return chunks


def _gen_wbfm_numpy() -> np.ndarray:
    t = np.arange(N_CHUNKS * CHUNK_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    audio = 0.7 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    k = 2 * math.pi * 75000.0 / SAMPLE_RATE_HZ
    phase = np.cumsum(k * audio)
    return np.exp(1j * phase).astype(np.complex64)


def _gen_am_dsb_numpy() -> np.ndarray:
    t = np.arange(N_CHUNKS * CHUNK_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    audio = 0.8 * np.sin(2 * np.pi * 1000 * t)
    return (1.0 + 0.8 * audio).astype(np.complex64)


def _gen_ssb_numpy() -> np.ndarray:
    t = np.arange(N_CHUNKS * CHUNK_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    audio = (
        0.8 * np.sin(2 * np.pi * 300 * t)
        + 0.4 * np.sin(2 * np.pi * 450 * t)
        + 0.2 * np.sin(2 * np.pi * 1200 * t)
    )
    # Hilbert of a pure tone is constant-envelope (looks like PSK); form RF DSB then
    # take analytic signal so envelope varies like real SSB.
    fc = 1500.0
    dsb = (1.0 + 0.7 * audio) * np.cos(2 * np.pi * fc * t)
    return hilbert(dsb).astype(np.complex64)


def _gen_fsk_numpy(
    n_tones: int,
    *,
    deviation_hz: float = FSK_DEVIATION_HZ,
    samples_per_symbol: int = FSK_SAMPLES_PER_SYMBOL,
    seed: int = 42,
) -> np.ndarray:
    """
    M-FSK with symmetric tone frequencies around 0 Hz (complex baseband).

    Positive-only tone frequencies (e.g. 1500/2000 Hz) produced a large IF
    offset that the family classifier confused with AM despite constant envelope.
    """
    n = N_CHUNKS * CHUNK_SAMPLES
    rng = np.random.default_rng(seed)
    if n_tones < 2:
        msg = f"FSK requires at least 2 tones, got {n_tones}"
        raise ValueError(msg)
    tone_freqs = np.linspace(-deviation_hz, deviation_hz, n_tones)
    n_symbols = n // samples_per_symbol + 2
    symbols = rng.integers(0, n_tones, size=n_symbols)
    inst_freq = np.repeat(tone_freqs[symbols], samples_per_symbol)[:n]
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / SAMPLE_RATE_HZ
    return np.exp(1j * phase).astype(np.complex64)


def _gen_gmsk_numpy(
    samples_per_symbol: int = GMSK_SAMPLES_PER_SYMBOL,
    seed: int = 42,
) -> np.ndarray:
    """GMSK: Gaussian-smoothed NRZ driving phase."""
    n = N_CHUNKS * CHUNK_SAMPLES
    rng = np.random.default_rng(seed)
    n_symbols = n // samples_per_symbol + 4
    bits = rng.integers(0, 2, size=n_symbols)
    nrz = 2 * bits.astype(np.float64) - 1.0
    upsampled = np.repeat(nrz, samples_per_symbol)[:n]
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    smoothed = np.convolve(upsampled, kernel, mode="same")
    # pi/2 phase step per symbol, spread across samples_per_symbol
    phase_inc = (np.pi / 2.0 / samples_per_symbol) * smoothed
    phase = np.cumsum(phase_inc)
    return np.exp(1j * phase).astype(np.complex64)


def _gen_sleipnir_numpy() -> np.ndarray:
    """Eight parallel 900-baud QPSK carriers (Sleipnir composite reference)."""
    n = N_CHUNKS * CHUNK_SAMPLES
    t = np.arange(n, dtype=np.float64) / DEFAULT_SAMPLE_RATE_HZ
    composite = np.zeros(n, dtype=np.complex128)
    baud_hz = 900.0
    sps = int(round(DEFAULT_SAMPLE_RATE_HZ / baud_hz))
    taps = _rrc_taps(sps, 0.35, 8 * sps + 1)
    delay = len(taps) // 2
    for idx, fc in enumerate(EXPECTED_CARRIER_HZ):
        rng = np.random.default_rng(7 + idx)
        n_sym = n // sps + 8
        bits_i = rng.integers(0, 2, size=n_sym)
        bits_q = rng.integers(0, 2, size=n_sym)
        symbols = ((2 * bits_i - 1) + 1j * (2 * bits_q - 1)) / np.sqrt(2)
        upsampled = np.repeat(symbols, sps)
        shaped = lfilter(taps, 1.0, upsampled)[delay : delay + n]
        composite += shaped * np.exp(2j * np.pi * fc * t)
    peak = float(np.max(np.abs(composite))) or 1.0
    return (composite / peak * 0.8).astype(np.complex64)


def _generate_signal(
    spec: ModeSpec,
    gr3_env: dict[str, str] | None,
) -> tuple[np.ndarray, str, str]:
    """Return (complex signal, generation_method, gr_env_used)."""
    if spec.expected_order == "NBFM_25" or spec.mode_name == "NBFM":
        msg = "NBFM_25 uses _gen_nbfm_scan_chunks(); call that path directly"
        raise RuntimeError(msg)
    if spec.mode_name == "WBFM":
        return _gen_wbfm_numpy(), "numpy", "none"
    if spec.mode_name == "AM":
        return _gen_am_dsb_numpy(), "numpy", "none"
    if spec.mode_name == "SSB":
        return _gen_ssb_numpy(), "numpy", "none"
    if spec.mode_name == "BPSK":
        msg = "BPSK uses _gen_psk_scan_chunks(); call that path directly"
        raise RuntimeError(msg)
    if spec.mode_name in ("QPSK", "SOQPSK", "8PSK"):
        msg = "PSK orders use _gen_psk_scan_chunks(); call that path directly"
        raise RuntimeError(msg)
    if spec.mode_name == "GMSK" or spec.mode_name == "FreeDV":
        return _gen_gmsk_numpy(), "numpy", "none"
    if spec.mode_name in ("2FSK", "DMR", "M17", "YSF", "NXDN", "dPMR", "P25"):
        n_tones = 2 if spec.mode_name == "2FSK" else 4
        return _gen_fsk_numpy(n_tones), "numpy", "none"
    if spec.mode_name == "4FSK":
        return _gen_fsk_numpy(4), "numpy", "none"
    if spec.mode_name == "8FSK":
        return _gen_fsk_numpy(8), "numpy", "none"
    if spec.mode_name == "D-Star":
        return _gen_gmsk_numpy(), "numpy", "none"
    msg = f"No built-in generator for mode {spec.mode_name}"
    raise RuntimeError(msg)


def generate_iq_for_project(
    project: GRProject,
    summary: ReadmeSummary,
    output_dir: Path,
    *,
    gr3_env: dict[str, str] | None,
    gr4_env: dict[str, str] | None,
    vocab: ClassifierVocab | None = None,
) -> list[GeneratedIQ]:
    """
    Generate reference IQ for modes listed in README.

    Never uses the scanned project's OOT blocks.
    """
    del gr4_env  # reserved for future GR4 built-in generators
    results: list[GeneratedIQ] = []
    gr_env_used_label = "gr3" if gr3_env else "none"

    if project.gr_version == "4" and gr3_env is None and not summary.modulation_modes:
        return results

    for mode_name in summary.modulation_modes:
        spec = lookup_mode(mode_name)
        if spec is None:
            results.append(
                GeneratedIQ(
                    iq_path=Path(),
                    sidecar_path=Path(),
                    block_name=f"mod_{mode_name.lower().replace('.', '_')}",
                    expected_family="",
                    expected_order="",
                    generation_method="skip",
                    gr_env_used="none",
                    mode_name=mode_name,
                    skipped=True,
                    skip_reason=f"Unknown mode: {mode_name}",
                )
            )
            continue

        block_name = f"mod_{spec.mode_name.lower().replace('.', '_').replace('-', '_')}"

        if spec.generation_method == "skip":
            skip_reason = mode_exclusion_reason(spec.mode_name) or spec.note or "No built-in equivalent"
            results.append(
                GeneratedIQ(
                    iq_path=Path(),
                    sidecar_path=Path(),
                    block_name=block_name,
                    expected_family=spec.expected_family,
                    expected_order=spec.expected_order,
                    generation_method="skip",
                    gr_env_used="none",
                    mode_name=spec.mode_name,
                    skipped=True,
                    skip_reason=skip_reason,
                )
            )
            continue

        if spec.generation_method == "plugin":
            if spec.expected_order == "sleipnir_8qpsk":
                signal = _gen_sleipnir_numpy()
                chunks = _chunks_from_complex(signal)
                note = spec.note + " Reference composite IQ for plugin validation."
                results.append(
                    _write_iq_and_sidecar(
                        output_dir,
                        block_name,
                        chunks,
                        expected_family="custom",
                        expected_order="sleipnir_8qpsk",
                        project_name=project.name,
                        generation_method="plugin",
                        gr_env_used=gr_env_used_label,
                        notes=note,
                    )
                )
            continue

        labels: tuple[str, str] | None = None
        if vocab is not None:
            labels = resolve_classifier_labels(spec, vocab)
            if labels is None:
                skip_reason = (
                    f"Mode labels not in classifier vocabulary: "
                    f"family={spec.expected_family!r} order={spec.expected_order!r}"
                )
                logger.warning("%s: %s", block_name, skip_reason)
                results.append(
                    GeneratedIQ(
                        iq_path=Path(),
                        sidecar_path=Path(),
                        block_name=block_name,
                        expected_family=spec.expected_family,
                        expected_order=spec.expected_order,
                        generation_method="skip",
                        gr_env_used="none",
                        mode_name=spec.mode_name,
                        skipped=True,
                        skip_reason=skip_reason,
                    )
                )
                continue
        expected_family, expected_order = labels or (
            spec.expected_family,
            spec.expected_order,
        )

        if project.gr_version == "4" and spec.generation_method != "plugin":
            results.append(
                GeneratedIQ(
                    iq_path=Path(),
                    sidecar_path=Path(),
                    block_name=block_name,
                    expected_family=expected_family,
                    expected_order=expected_order,
                    generation_method="skip",
                    gr_env_used="none",
                    mode_name=spec.mode_name,
                    skipped=True,
                    skip_reason="GR4 project: use captured IQ or plugin; no GR4 built-in generator.",
                )
            )
            continue

        use_gr = gr3_env is not None and spec.generation_method == "gr3_builtin"
        if expected_order == "NBFM_25":
            chunks = _gen_nbfm_scan_chunks(use_gnuradio=use_gr)
            method = "gr3_builtin" if use_gr else "numpy"
            env_label = "gr3" if use_gr else "none"
        elif spec.mode_name == "BPSK":
            chunks = _gen_psk_scan_chunks(2, use_gnuradio=use_gr)
            method = "gr3_builtin" if use_gr else "numpy"
            env_label = "gr3" if use_gr else "none"
        elif spec.mode_name in ("QPSK", "SOQPSK"):
            chunks = _gen_psk_scan_chunks(4, use_gnuradio=use_gr)
            method = "gr3_builtin" if use_gr else "numpy"
            env_label = "gr3" if use_gr else "none"
        elif spec.mode_name == "8PSK":
            chunks = _gen_psk_scan_chunks(8, use_gnuradio=use_gr)
            method = "gr3_builtin" if use_gr else "numpy"
            env_label = "gr3" if use_gr else "none"
        else:
            signal, method, env_label = _generate_signal(spec, gr3_env)
            chunks = _chunks_from_complex(signal)
        notes = (
            "Generated by rmv scan using built-in GNU Radio / numpy reference only. "
            "Project OOT blocks were not used."
        )
        if spec.note:
            notes += " " + spec.note
        if spec.protocol_only:
            notes += " Underlying modulation validated only; protocol framing not verified."

        results.append(
            _write_iq_and_sidecar(
                output_dir,
                block_name,
                chunks,
                expected_family=expected_family,
                expected_order=expected_order,
                project_name=project.name,
                generation_method=method,
                gr_env_used=env_label,
                notes=notes,
            )
        )

    return results
