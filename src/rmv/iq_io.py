"""Load contributed IQ files (.iq binary or SigMF)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from rmv.constants import MAX_IQ_FILE_BYTES
from rmv.dataset.loader import read_iq_binary
from rmv.dataset.preprocess import chunk_iq_file

logger = logging.getLogger(__name__)


def resolve_iq_path(path: Path) -> Path:
    """Resolve .iq, .sigmf-data, or SigMF basename to a readable data file."""
    if path.suffix == ".iq" and path.is_file():
        return path
    if path.suffix == ".sigmf-data" and path.is_file():
        return path
    sigmf_data = path.with_suffix(".sigmf-data")
    if sigmf_data.is_file():
        return sigmf_data
    if path.is_file():
        return path
    msg = f"IQ data file not found: {path}"
    raise FileNotFoundError(msg)


def read_sigmf_meta(data_path: Path) -> dict[str, object]:
    """Read SigMF metadata JSON if present."""
    meta_path = data_path.with_suffix(".sigmf-meta")
    if not meta_path.is_file():
        meta_path = data_path.parent / f"{data_path.stem}.sigmf-meta"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_iq_chunks_from_path(path: Path, chunk_samples: int = 1024) -> np.ndarray:
    """
    Load IQ file as (N, 2, chunk_samples) float32 chunks.

    Supports .iq (interleaved float32) and SigMF .sigmf-data (cf32 or f32 LE).
    """
    data_path = resolve_iq_path(path)
    size = data_path.stat().st_size
    if size > MAX_IQ_FILE_BYTES:
        msg = f"IQ file exceeds 50 MB limit: {data_path}"
        raise ValueError(msg)

    if data_path.suffix in (".sigmf-data",):
        meta = read_sigmf_meta(data_path)
        dtype = "cf32_le"
        global_meta = meta.get("global", {}) if isinstance(meta.get("global"), dict) else {}
        if isinstance(global_meta, dict) and "core:datatype" in global_meta:
            dtype = str(global_meta["core:datatype"])
        raw_bytes = data_path.read_bytes()
        if dtype.startswith("cf32"):
            n = len(raw_bytes) // 8
            complex_samples = np.frombuffer(raw_bytes[: n * 8], dtype="<f4").reshape(-1, 2)
            interleaved = np.stack([complex_samples[:, 0], complex_samples[:, 1]], axis=1).reshape(
                -1
            )
        else:
            interleaved = np.frombuffer(raw_bytes, dtype="<f4")
        return chunk_iq_file(interleaved.astype(np.float32), chunk_samples=chunk_samples)

    raw = read_iq_binary(data_path, max_bytes=MAX_IQ_FILE_BYTES)
    return chunk_iq_file(raw, chunk_samples=chunk_samples)
