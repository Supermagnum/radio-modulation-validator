"""IQ preprocessing: normalisation, windowing, format conversion."""

from __future__ import annotations

import hashlib
import inspect
import logging
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import resample

from rmv.constants import CHUNK_SAMPLES

logger = logging.getLogger(__name__)


class CacheLoadError(OSError):
    """Raised when a preprocessed cache shard cannot be read."""


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

    data: interleaved I0, Q0, I1, Q1, ... (not reshape-safe; use de-interleave first).
    """
    iq = interleaved_to_iq(np.asarray(data, dtype=np.float32))
    if iq.shape[1] % chunk_samples != 0:
        msg = (
            f"IQ sample count {iq.shape[1]} is not a multiple of "
            f"{chunk_samples} (interleaved float32 I/Q pairs)"
        )
        raise ValueError(msg)
    n_chunks = iq.shape[1] // chunk_samples
    trimmed = iq[:, : n_chunks * chunk_samples]
    chunks = trimmed.reshape(2, n_chunks, chunk_samples).transpose(1, 0, 2)
    return normalise_unit_power(chunks.astype(np.float32), axis=-1)


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


def _compute_processing_version() -> str:
    """Hash of upsample/normalise source; changes when processing pipeline changes."""
    src = inspect.getsource(upsample_iq_128_to_1024) + inspect.getsource(normalise_unit_power)
    return hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]


PROCESSING_VERSION: str = _compute_processing_version()

CACHE_FILE_SUFFIX = f"_{PROCESSING_VERSION}.npy.npz"


def cache_path_for_source(cache_dir: Path, source: str, key: str) -> Path:
    """Build versioned cache path for a preprocessed dataset shard."""
    safe_key = key.replace("/", "_").replace(" ", "_")
    return cache_dir / source / f"{safe_key}{CACHE_FILE_SUFFIX}"


def is_current_cache_file(path: Path) -> bool:
    """True if path matches the current processing pipeline version."""
    return path.name.endswith(CACHE_FILE_SUFFIX)


def _is_cache_artifact(path: Path) -> bool:
    name = path.name
    return name.endswith(".npz") or name.endswith(".npy") or name.endswith(".npy.npz")


def clean_stale_cache(cache_dir: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """
    Remove cache files that do not match PROCESSING_VERSION.

    Returns (removed_count, kept_count).
    """
    if not cache_dir.is_dir():
        return 0, 0
    removed = 0
    kept = 0
    for path in sorted(cache_dir.rglob("*")):
        if not path.is_file() or not _is_cache_artifact(path):
            continue
        if is_current_cache_file(path):
            kept += 1
            continue
        if not dry_run:
            path.unlink()
        removed += 1
    return removed, kept


def save_cache_shard(path: Path, samples: np.ndarray, labels: np.ndarray, snr: np.ndarray) -> None:
    """Save preprocessed shard atomically (temp .npz file then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends ".npz" when the path does not already end with it.
    tmp_path = path.parent / f".{path.name}.part.npz"
    try:
        np.savez_compressed(tmp_path, samples=samples, labels=labels, snr_db=snr)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_cache_shard(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed shard from cache."""
    if not path.is_file():
        msg = f"Cache file not found: {path}"
        raise FileNotFoundError(msg)
    try:
        with np.load(path) as data:
            required = ("samples", "labels", "snr_db")
            missing = [key for key in required if key not in data]
            if missing:
                msg = f"Cache file {path} missing arrays: {missing}"
                raise CacheLoadError(msg)
            return (
                data["samples"].astype(np.float32),
                data["labels"].astype(np.int32),
                data["snr_db"].astype(np.float32),
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        msg = f"Failed to read cache file {path}: {exc}"
        raise CacheLoadError(msg) from exc


def try_load_cache_shard(
    path: Path,
    *,
    remove_on_failure: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load cache shard, or None if missing/corrupt.

    Corrupt files are removed so the next load can rebuild them.
    """
    if not path.is_file():
        return None
    try:
        return load_cache_shard(path)
    except CacheLoadError as exc:
        logger.warning("%s", exc)
        if remove_on_failure:
            path.unlink(missing_ok=True)
        return None
