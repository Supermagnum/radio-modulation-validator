"""IQ preprocessing: normalisation, windowing, format conversion."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.signal import resample

from rmv.constants import CHUNK_SAMPLES

logger = logging.getLogger(__name__)


def normalise_unit_power(samples: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalise IQ samples to unit average power per sample.

    For (N, 2, L) arrays, each of the N windows is scaled independently.
    """
    del axis  # per-sample normalisation always uses time axis for (N, 2, L)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim < 2:
        msg = f"Expected at least 2D array, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.ndim == 3 and arr.shape[1] == 2:
        power = np.mean(np.sum(arr**2, axis=1), axis=-1)
        power = np.maximum(power, 1e-12)
        scale = np.sqrt(power).reshape(-1, 1, 1)
        return (arr / scale).astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 2:
        power = np.mean(np.sum(arr**2, axis=0))
        power = max(float(power), 1e-12)
        return (arr / np.sqrt(power)).astype(np.float32)
    power = np.mean(arr**2)
    power = max(float(power), 1e-12)
    return (arr / np.sqrt(power)).astype(np.float32)


def upsample_iq_128_to_1024(samples: np.ndarray) -> np.ndarray:
    """
    Upsample RadioML 128-sample windows to 1024 using scipy resample.

    Input shape: (N, 2, 128) or (2, 128)
    Output shape: (N, 2, 1024) or (2, 1024)
    """
    single = samples.ndim == 2
    if single:
        samples = samples[np.newaxis, ...]
    n = samples.shape[0]
    out = np.zeros((n, 2, CHUNK_SAMPLES), dtype=np.float32)
    for i in range(n):
        for ch in range(2):
            out[i, ch] = resample(samples[i, ch], CHUNK_SAMPLES).astype(np.float32)
    if single:
        return out[0]
    return out


def chunk_iq_file(data: np.ndarray, chunk_samples: int = CHUNK_SAMPLES) -> np.ndarray:
    """
    Split flat interleaved I/Q float32 into (N, 2, chunk_samples).

    data: shape (2 * chunk_samples * N,) interleaved I,Q
    """
    if data.size % (2 * chunk_samples) != 0:
        msg = (
            f"IQ data length {data.size} is not a multiple of "
            f"{2 * chunk_samples} (interleaved float32 I/Q pairs)"
        )
        raise ValueError(msg)
    n_chunks = data.size // (2 * chunk_samples)
    reshaped = data.reshape(n_chunks, 2, chunk_samples)
    return normalise_unit_power(reshaped, axis=-1)


def interleaved_to_iq(data: np.ndarray) -> np.ndarray:
    """Convert interleaved float32 I/Q to (2, L)."""
    if data.size % 2 != 0:
        msg = "Interleaved IQ must have even number of float32 values"
        raise ValueError(msg)
    pairs = data.reshape(-1, 2)
    return pairs.T.astype(np.float32)


def tim_to_chunks(
    tim_data: np.ndarray,
    chunk_samples: int = CHUNK_SAMPLES,
    max_chunks: int | None = None,
) -> np.ndarray:
    """
    Convert CSPB .tim interleaved data to fixed-size chunks.

    tim_data: 1D float32 interleaved real/imag
    """
    iq = interleaved_to_iq(tim_data)
    length = iq.shape[1]
    n_full = length // chunk_samples
    if n_full == 0:
        msg = f"Signal too short for chunk size {chunk_samples}: length={length}"
        raise ValueError(msg)
    trimmed = iq[:, : n_full * chunk_samples]
    chunks = trimmed.reshape(2, n_full, chunk_samples).transpose(1, 0, 2)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return normalise_unit_power(chunks.astype(np.float32), axis=-1)


def cache_path_for_source(cache_dir: Path, source: str, key: str) -> Path:
    """Build cache file path for preprocessed dataset shard."""
    safe_key = key.replace("/", "_").replace(" ", "_")
    return cache_dir / source / f"{safe_key}.npy"


def save_cache_shard(path: Path, samples: np.ndarray, labels: np.ndarray, snr: np.ndarray) -> None:
    """Save preprocessed shard to .npy bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, samples=samples, labels=labels, snr_db=snr)


def load_cache_shard(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed shard from cache."""
    if not path.is_file():
        msg = f"Cache file not found: {path}"
        raise FileNotFoundError(msg)
    data = np.load(path)
    return (
        data["samples"].astype(np.float32),
        data["labels"].astype(np.int32),
        data["snr_db"].astype(np.float32),
    )
