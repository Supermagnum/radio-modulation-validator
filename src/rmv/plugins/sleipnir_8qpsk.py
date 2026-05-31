"""Sleipnir 8-carrier parallel QPSK composite mode validator."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.signal import lfilter

from rmv.plugins.base import CustomModeResult, CustomModeValidator

CHUNK_SAMPLES = 1024
DEFAULT_SAMPLE_RATE_HZ = 48_000.0
EXPECTED_CARRIER_HZ = [-4550.0, -3250.0, -1950.0, -650.0, 650.0, 1950.0, 3250.0, 4550.0]
CARRIER_MATCH_TOLERANCE_HZ = 200.0
SPACING_TARGET_HZ = 1300.0
SPACING_TOLERANCE_HZ = 150.0
SPACING_STD_MAX_HZ = 160.0
CARRIER_BW_MIN_HZ = 800.0
CARRIER_BW_MAX_HZ = 1400.0
SYMBOL_RATE_TARGET_HZ = 900.0
SYMBOL_RATE_TOLERANCE_HZ = 50.0
SYMBOL_RATE_MIN_PASS_COUNT = 6
QPSK_MIN_PASS_COUNT = 6
TOTAL_BW_MIN_HZ = 9000.0
TOTAL_BW_MAX_HZ = 12000.0
BANDPASS_HALF_WIDTH_HZ = 550.0
RRRC_ALPHA = 0.35


def _chunks_to_stream(samples: np.ndarray, max_samples: int = 192_000) -> np.ndarray:
    """Flatten (N, 2, 1024) chunks to 1-D complex baseband (trimmed for speed)."""
    if samples.ndim != 3 or samples.shape[1] != 2:
        msg = f"Expected (N, 2, 1024) samples, got {samples.shape}"
        raise ValueError(msg)
    i = samples[:, 0, :].reshape(-1)
    q = samples[:, 1, :].reshape(-1)
    stream = (i + 1j * q).astype(np.complex64)
    if len(stream) > max_samples:
        return stream[:max_samples]
    return stream


def _average_spectrum(
    samples: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average |FFT| across chunks; return frequencies (Hz) and magnitude."""
    n = samples.shape[-1]
    mags: list[np.ndarray] = []
    for chunk in samples:
        x = chunk[0] + 1j * chunk[1]
        spec = np.fft.fftshift(np.fft.fft(x))
        mags.append(np.abs(spec))
    avg = np.mean(mags, axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate_hz))
    return freqs.astype(np.float64), avg.astype(np.float64)


def _match_carriers(
    freqs_hz: np.ndarray,
    spectrum: np.ndarray,
    expected: list[float],
    tolerance_hz: float,
) -> tuple[list[float], list[float]]:
    """Find strongest spectral peak near each expected carrier frequency."""
    matched_detected: list[float] = []
    matched_expected: list[float] = []
    for exp in expected:
        mask = np.abs(freqs_hz - exp) <= tolerance_hz
        if not np.any(mask):
            continue
        sub_spec = spectrum[mask]
        sub_freqs = freqs_hz[mask]
        peak_i = int(np.argmax(sub_spec))
        peak_strength = float(sub_spec[peak_i])
        if peak_strength < float(np.max(spectrum)) * 0.12:
            continue
        if peak_strength < float(np.median(spectrum)) * 4.0:
            continue
        matched_detected.append(float(sub_freqs[peak_i]))
        matched_expected.append(exp)

    order = np.argsort(matched_detected)
    sorted_detected = [matched_detected[i] for i in order]
    sorted_expected = [matched_expected[i] for i in order]
    return sorted_detected, sorted_expected


def _spacing_stats(positions_hz: list[float]) -> tuple[float, float]:
    if len(positions_hz) < 2:
        return 0.0, 999.0
    spacings = np.diff(np.sort(positions_hz))
    return float(np.mean(spacings)), float(np.std(spacings))


def _bandwidth_3db(freqs_hz: np.ndarray, spectrum: np.ndarray, center_hz: float) -> float:
    """Estimate 3 dB bandwidth around a carrier peak in the averaged spectrum."""
    idx = int(np.argmin(np.abs(freqs_hz - center_hz)))
    peak_val = float(spectrum[idx])
    if peak_val <= 0:
        return 0.0
    threshold = peak_val * 10 ** (-6.0 / 10.0)
    left = idx
    while left > 0 and spectrum[left] >= threshold:
        left -= 1
    right = idx
    while right < len(spectrum) - 1 and spectrum[right] >= threshold:
        right += 1
    return float(freqs_hz[right] - freqs_hz[left])


def _bandpass_carrier(
    stream: np.ndarray,
    sample_rate_hz: float,
    center_hz: float,
) -> np.ndarray:
    """Extract narrowband complex baseband for one carrier (FFT mask)."""
    n = len(stream)
    spec = np.fft.fft(stream)
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate_hz)
    mask = np.abs(freqs - center_hz) <= BANDPASS_HALF_WIDTH_HZ
    spec = spec * mask
    band = np.fft.ifft(spec).astype(np.complex64)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    return (band * np.exp(-2j * np.pi * center_hz * t)).astype(np.complex64)


def _estimate_symbol_rate_hz(
    carrier_bb: np.ndarray,
    sample_rate_hz: float,
) -> float:
    """Estimate symbol rate from envelope autocorrelation near expected baud."""
    env = np.abs(carrier_bb).astype(np.float64)
    env -= np.mean(env)
    n = len(env)
    if n < 256:
        return 0.0
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(env, n=n_fft)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac = ac / (ac[0] + 1e-12)
    expected_lag = int(round(sample_rate_hz / SYMBOL_RATE_TARGET_HZ))
    half_win = max(8, int(expected_lag * 0.25))
    lo = max(8, expected_lag - half_win)
    hi = min(n - 1, expected_lag + half_win)
    if lo >= hi:
        return 0.0
    segment = ac[lo : hi + 1]
    peak_lag = lo + int(np.argmax(segment))
    best_rate = float(sample_rate_hz / peak_lag) if peak_lag > 0 else 0.0
    for lag in range(lo, hi + 1):
        rate = float(sample_rate_hz / lag)
        if abs(rate - SYMBOL_RATE_TARGET_HZ) < abs(best_rate - SYMBOL_RATE_TARGET_HZ):
            best_rate = rate
    return best_rate


def _rrc_taps(sps: int, alpha: float, num_taps: int) -> np.ndarray:
    """Root raised cosine filter taps (odd length)."""
    n = num_taps
    t = (np.arange(n) - (n - 1) / 2.0) / float(sps)
    h = np.zeros(n, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - alpha + 4.0 * alpha / np.pi
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h**2))
    return h.astype(np.float64)


def _qpsk_constellation_check(carrier_bb: np.ndarray, sample_rate_hz: float) -> tuple[bool, float]:
    """k-means k=4 cluster balance for QPSK-like constellation."""
    baud = SYMBOL_RATE_TARGET_HZ
    sps = max(2, int(round(sample_rate_hz / baud)))
    x = carrier_bb.astype(np.complex64)
    if len(x) < 80:
        return False, 0.0
    taps = _rrc_taps(sps, RRRC_ALPHA, num_taps=8 * sps + 1)
    shaped = lfilter(taps, 1.0, x)
    symbols = shaped[sps * 4 :: sps]
    if len(symbols) < 40:
        return False, 0.0
    if len(symbols) > 400:
        symbols = symbols[:: max(1, len(symbols) // 400)]
    pts = np.column_stack([symbols.real, symbols.imag]).astype(np.float64)
    scale = float(np.percentile(np.abs(pts), 95)) + 1e-9
    pts /= scale
    try:
        centroids, labels = kmeans2(pts, 4, minit="points", iter=8)
    except Exception:
        return False, 0.0
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    balance = float(np.min(counts) / (np.max(counts) + 1e-9))
    centres_ok = all(0.35 <= float(np.linalg.norm(c)) <= 1.2 for c in centroids)
    frac_ok = all(0.15 <= c / len(labels) <= 0.35 for c in counts)
    passed = balance >= 0.4 and frac_ok and centres_ok
    return passed, balance


def _total_occupied_bandwidth_hz(
    freqs_hz: np.ndarray,
    spectrum: np.ndarray,
) -> float:
    peak = float(np.max(spectrum))
    if peak <= 0:
        return 0.0
    threshold = peak * 10 ** (-20.0 / 20.0)
    occupied = freqs_hz[spectrum >= threshold]
    if len(occupied) < 2:
        return 0.0
    return float(occupied[-1] - occupied[0])


def _score_carrier_count(count: int) -> float:
    if count == 8:
        return 1.0
    return max(0.0, min(1.0, count / 8.0))


def _score_spacing(mean_hz: float, std_hz: float) -> float:
    mean_ok = abs(mean_hz - SPACING_TARGET_HZ) <= SPACING_TOLERANCE_HZ
    std_ok = std_hz <= SPACING_STD_MAX_HZ
    if mean_ok and std_ok:
        return 1.0
    mean_part = max(0.0, 1.0 - abs(mean_hz - SPACING_TARGET_HZ) / SPACING_TOLERANCE_HZ)
    std_part = max(0.0, 1.0 - std_hz / SPACING_STD_MAX_HZ)
    return 0.5 * mean_part + 0.5 * std_part


def _score_list_fraction(passes: list[bool], minimum: int) -> float:
    n_pass = sum(passes)
    if n_pass >= minimum:
        return 1.0
    return n_pass / float(minimum)


class Sleipnir8QPSKValidator(CustomModeValidator):
    """Eight parallel QPSK carriers in one composite wideband IQ stream."""

    mode_id = "sleipnir_8qpsk"
    description = (
        "Validates Sleipnir-style 8-carrier parallel QPSK (900 baud per carrier, "
        "1300 Hz spacing) on composite IQ without per-carrier file splits."
    )

    def describe(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "description": self.description,
            "measurements": [
                "Spectral carrier detection (8 peaks, ~1300 Hz spacing)",
                "Per-carrier 3 dB bandwidth (800-1400 Hz)",
                "Per-carrier symbol rate via cyclostationary |x|^2 peak (~900 Hz)",
                "Per-carrier QPSK 4-cluster constellation balance",
                "Total occupied bandwidth (-20 dB, 9-12 kHz)",
            ],
            "pass_criteria": {
                "carrier_count": 8,
                "carrier_spacing_mean_hz": f"{SPACING_TARGET_HZ} +/- {SPACING_TOLERANCE_HZ}",
                "carrier_spacing_std_hz": f"<= {SPACING_STD_MAX_HZ} (nominal 100 Hz)",
                "symbol_rate_pass": f">= {SYMBOL_RATE_MIN_PASS_COUNT}/8 carriers",
                "qpsk_pass": f">= {QPSK_MIN_PASS_COUNT}/8 carriers",
                "total_bandwidth_hz": f"{TOTAL_BW_MIN_HZ}-{TOTAL_BW_MAX_HZ}",
            },
            "default_sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
            "expected_carrier_positions_hz": EXPECTED_CARRIER_HZ,
        }

    def validate(
        self,
        samples: np.ndarray,
        sample_rate_hz: float,
        metadata: dict[str, Any],
    ) -> CustomModeResult:
        del metadata
        fs = float(sample_rate_hz)
        freqs_hz, avg_spec = _average_spectrum(samples, fs)
        positions, _ = _match_carriers(
            freqs_hz,
            avg_spec,
            EXPECTED_CARRIER_HZ,
            CARRIER_MATCH_TOLERANCE_HZ,
        )
        carrier_count = len(positions)
        spacing_mean, spacing_std = _spacing_stats(positions)

        carrier_bws = [_bandwidth_3db(freqs_hz, avg_spec, p) for p in positions]
        bw_pass = all(CARRIER_BW_MIN_HZ <= bw <= CARRIER_BW_MAX_HZ for bw in carrier_bws)

        stream = _chunks_to_stream(samples)

        symbol_rates: list[float] = []
        symbol_pass: list[bool] = []
        qpsk_balance: list[float] = []
        qpsk_pass: list[bool] = []

        for pos in positions:
            bb = _bandpass_carrier(stream, fs, pos)
            rate = _estimate_symbol_rate_hz(bb, fs)
            symbol_rates.append(rate)
            symbol_pass.append(abs(rate - SYMBOL_RATE_TARGET_HZ) <= SYMBOL_RATE_TOLERANCE_HZ)
            qp, bal = _qpsk_constellation_check(bb, fs)
            qpsk_pass.append(qp)
            qpsk_balance.append(bal)

        while len(symbol_rates) < 8:
            symbol_rates.append(0.0)
            symbol_pass.append(False)
            qpsk_balance.append(0.0)
            qpsk_pass.append(False)

        total_bw = _total_occupied_bandwidth_hz(freqs_hz, avg_spec)
        total_bw_ok = TOTAL_BW_MIN_HZ <= total_bw <= TOTAL_BW_MAX_HZ

        spacing_ok = (
            carrier_count >= 2
            and abs(spacing_mean - SPACING_TARGET_HZ) <= SPACING_TOLERANCE_HZ
            and spacing_std <= SPACING_STD_MAX_HZ
        )
        symbol_ok = sum(symbol_pass) >= SYMBOL_RATE_MIN_PASS_COUNT
        qpsk_ok = sum(qpsk_pass) >= QPSK_MIN_PASS_COUNT

        pass_overall = (
            carrier_count == 8
            and spacing_ok
            and bw_pass
            and symbol_ok
            and qpsk_ok
            and total_bw_ok
        )

        score_count = _score_carrier_count(carrier_count)
        score_spacing = _score_spacing(spacing_mean, spacing_std) if carrier_count >= 2 else 0.0
        score_symbol = _score_list_fraction(symbol_pass, SYMBOL_RATE_MIN_PASS_COUNT)
        score_qpsk = _score_list_fraction(qpsk_pass, QPSK_MIN_PASS_COUNT)
        confidence = (
            0.3 * score_count + 0.2 * score_spacing + 0.25 * score_symbol + 0.25 * score_qpsk
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        notes_parts: list[str] = []
        if carrier_count != 8:
            notes_parts.append(f"carrier_count={carrier_count}, expected 8")
        if not spacing_ok:
            notes_parts.append(
                f"spacing mean={spacing_mean:.1f} Hz std={spacing_std:.1f} Hz"
            )
        if not bw_pass:
            notes_parts.append("carrier bandwidth out of range")
        if not symbol_ok:
            notes_parts.append(f"symbol rate pass {sum(symbol_pass)}/8")
        if not qpsk_ok:
            notes_parts.append(f"qpsk pass {sum(qpsk_pass)}/8")
        if not total_bw_ok:
            notes_parts.append(f"total_bandwidth_hz={total_bw:.1f}")

        return CustomModeResult(
            mode_id=self.mode_id,
            pass_overall=pass_overall,
            confidence=confidence,
            metrics={
                "carrier_count": carrier_count,
                "carrier_positions_hz": [round(p, 1) for p in positions],
                "carrier_spacing_mean_hz": round(spacing_mean, 1),
                "carrier_spacing_std_hz": round(spacing_std, 1),
                "carrier_bandwidths_hz": [round(b, 1) for b in carrier_bws],
                "symbol_rate_estimates_hz": [round(r, 1) for r in symbol_rates[:8]],
                "symbol_rate_pass": symbol_pass[:8],
                "qpsk_cluster_balance": [round(b, 3) for b in qpsk_balance[:8]],
                "qpsk_pass": qpsk_pass[:8],
                "total_bandwidth_hz": round(total_bw, 1),
            },
            notes="; ".join(notes_parts),
        )
