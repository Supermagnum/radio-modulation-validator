"""Loaders for RadioML 2016.10A, HISARMOD 2019.1, and CSPB.ML.2018R2."""

from __future__ import annotations

import hashlib
import logging
import pickle
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

from rmv.constants import (
    CHUNK_SAMPLES,
    CSPB_CLASS_ALIASES,
    CSPB_CLASSES,
    HISARMOD_CLASSES,
    RADIOML_CLASSES,
)
from rmv.dataset.preprocess import (
    cache_path_for_source,
    load_cache_shard,
    normalise_unit_power,
    save_cache_shard,
    tim_to_chunks,
    upsample_iq_128_to_1024,
)
from rmv.types import IQDataset

logger = logging.getLogger(__name__)


def verify_checksum(path: Path, checksum_file: Path | None) -> None:
    """Verify file integrity against user-supplied checksum file if provided."""
    if checksum_file is None or not checksum_file.is_file():
        return
    expected: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            expected[parts[-1]] = parts[0].lower()
    key = path.name
    if key not in expected:
        logger.warning("No checksum entry for %s in %s; skipping verify", key, checksum_file)
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected[key]:
        msg = f"Checksum mismatch for {path}: expected {expected[key]}, got {actual}"
        raise ValueError(msg)


def _resolve_radioml_pickle(path: Path) -> Path:
    """Return path to RML2016.10a_dict.pkl only (strict; no arbitrary .pkl fallback)."""
    from rmv.dataset.paths import RADIOML_PKL_NAME
    from rmv.dataset.radioml_resolve import extract_radioml_tar_strict, find_radioml_pickle

    if path.suffix == ".pkl" and path.is_file():
        if path.name != RADIOML_PKL_NAME:
            logger.warning("Expected %s, got %s", RADIOML_PKL_NAME, path.name)
        return path
    if path.name.endswith(".tar.bz2") or path.suffixes[-2:] == [".tar", ".bz2"]:
        found = find_radioml_pickle(path.parent)
        if found is not None:
            return found
        return extract_radioml_tar_strict(path, path.parent)
    if path.is_dir():
        found = find_radioml_pickle(path)
        if found is not None:
            return found
        msg = f"{RADIOML_PKL_NAME} not found under {path}"
        raise FileNotFoundError(msg)
    msg = f"Unsupported RadioML path: {path}"
    raise FileNotFoundError(msg)


def load_radioml_streaming(
    path: Path,
    *,
    cache_dir: Path | None = None,
    checksum_file: Path | None = None,
    load_all: bool = False,
    batch_size: int = 512,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (samples, labels, snr) batches from RadioML 2016.10A."""
    pkl_path = _resolve_radioml_pickle(path)
    verify_checksum(pkl_path, checksum_file)
    cache_key = f"radioml_{pkl_path.stat().st_size}"
    if cache_dir is not None:
        cache_path = cache_path_for_source(cache_dir, "radioml2016", cache_key)
        if cache_path.is_file():
            samples, labels, snr = load_cache_shard(cache_path)
            if load_all:
                yield samples, labels, snr
                return
            for start in range(0, len(labels), batch_size):
                end = min(start + batch_size, len(labels))
                yield samples[start:end], labels[start:end], snr[start:end]
            return

    with pkl_path.open("rb") as f:
        data: dict[tuple[str, str], np.ndarray] = pickle.load(f)

    class_to_idx = {c: i for i, c in enumerate(RADIOML_CLASSES)}
    all_samples: list[np.ndarray] = []
    all_labels: list[int] = []
    all_snr: list[float] = []

    for (mod, snr_str), arr in data.items():
        if mod not in class_to_idx:
            continue
        label = class_to_idx[mod]
        snr_val = float(snr_str)
        upsampled = upsample_iq_128_to_1024(arr.astype(np.float32))
        upsampled = normalise_unit_power(upsampled, axis=-1)
        n = upsampled.shape[0]
        all_samples.append(upsampled)
        all_labels.extend([label] * n)
        all_snr.extend([snr_val] * n)

    samples = np.concatenate(all_samples, axis=0)
    labels = np.array(all_labels, dtype=np.int32)
    snr_arr = np.array(all_snr, dtype=np.float32)

    if cache_dir is not None:
        save_cache_shard(cache_path_for_source(cache_dir, "radioml2016", cache_key), samples, labels, snr_arr)

    if load_all:
        yield samples, labels, snr_arr
        return
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        yield samples[start:end], labels[start:end], snr_arr[start:end]


def load_radioml(
    path: Path,
    *,
    cache_dir: Path | None = None,
    checksum_file: Path | None = None,
) -> IQDataset:
    """Load full RadioML 2016.10A dataset into memory."""
    for samples, labels, snr in load_radioml_streaming(
        path, cache_dir=cache_dir, checksum_file=checksum_file, load_all=True
    ):
        return IQDataset(
            samples=samples,
            labels=labels,
            snr_db=snr,
            class_names=RADIOML_CLASSES,
            source="radioml2016",
        )
    msg = "RadioML loader produced no data"
    raise RuntimeError(msg)


def load_hisarmod_streaming(
    path: Path,
    *,
    cache_dir: Path | None = None,
    checksum_file: Path | None = None,
    load_all: bool = False,
    batch_size: int = 512,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield batches from HISARMOD HDF5."""
    if not path.is_file():
        msg = f"HISARMOD HDF5 not found: {path}"
        raise FileNotFoundError(msg)
    verify_checksum(path, checksum_file)
    cache_key = f"hisarmod_{path.stat().st_size}"
    if cache_dir is not None:
        cache_path = cache_path_for_source(cache_dir, "hisarmod", cache_key)
        if cache_path.is_file():
            samples, labels, snr = load_cache_shard(cache_path)
            if load_all:
                yield samples, labels, snr
                return
            for start in range(0, len(labels), batch_size):
                end = min(start + batch_size, len(labels))
                yield samples[start:end], labels[start:end], snr[start:end]
            return

    class_to_idx = {c: i for i, c in enumerate(HISARMOD_CLASSES)}
    all_samples: list[np.ndarray] = []
    all_labels: list[int] = []
    all_snr: list[float] = []

    with h5py.File(path, "r") as hf:
        keys = sorted(hf.keys())
        for key in keys:
            grp = hf[key]
            mod_name = _hisarmod_key_to_class(key)
            if mod_name not in class_to_idx:
                logger.warning("Unknown HISARMOD group %s -> %s", key, mod_name)
                continue
            label = class_to_idx[mod_name]
            for snr_key in grp.keys():
                snr_val = _parse_snr_key(snr_key)
                ds = grp[snr_key][()]
                arr = np.asarray(ds, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == 2:
                    arr = arr[np.newaxis, ...]
                elif arr.ndim == 3 and arr.shape[1] != 2 and arr.shape[2] == 2:
                    arr = np.transpose(arr, (0, 2, 1))
                arr = normalise_unit_power(arr, axis=-1)
                n = arr.shape[0]
                all_samples.append(arr)
                all_labels.extend([label] * n)
                all_snr.extend([snr_val] * n)

    samples = np.concatenate(all_samples, axis=0)
    labels = np.array(all_labels, dtype=np.int32)
    snr_arr = np.array(all_snr, dtype=np.float32)

    if cache_dir is not None:
        save_cache_shard(cache_path_for_source(cache_dir, "hisarmod", cache_key), samples, labels, snr_arr)

    if load_all:
        yield samples, labels, snr_arr
        return
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        yield samples[start:end], labels[start:end], snr_arr[start:end]


def _hisarmod_key_to_class(key: str) -> str:
    """Map HDF5 group key to standard class name."""
    mapping = {c.upper().replace("-", ""): c for c in HISARMOD_CLASSES}
    normalized = key.upper().replace("_", "").replace("-", "")
    for k, v in mapping.items():
        if k in normalized or normalized in k:
            return v
    return key


def _parse_snr_key(key: str) -> float:
    """Parse SNR from HDF5 subgroup key like 'snr_10' or '-10dB'."""
    s = key.lower().replace("db", "").replace("snr", "").replace("_", "").strip()
    try:
        return float(s)
    except ValueError:
        digits = "".join(c if c.isdigit() or c in ".-" else " " for c in key).split()
        if digits:
            return float(digits[0])
        return 0.0


def load_hisarmod(
    path: Path,
    *,
    cache_dir: Path | None = None,
    checksum_file: Path | None = None,
) -> IQDataset:
    """Load full HISARMOD dataset."""
    for samples, labels, snr in load_hisarmod_streaming(
        path, cache_dir=cache_dir, checksum_file=checksum_file, load_all=True
    ):
        return IQDataset(
            samples=samples,
            labels=labels,
            snr_db=snr,
            class_names=HISARMOD_CLASSES,
            source="hisarmod",
        )
    msg = "HISARMOD loader produced no data"
    raise RuntimeError(msg)


def _find_truth_file(directory: Path) -> Path:
    """Locate CSPB truth file in batch directory."""
    from rmv.dataset.paths import find_cspb_truth_file

    found = find_cspb_truth_file(directory)
    if found is not None:
        return found
    msg = f"No CSPB truth file (signal_record.txt) under {directory}"
    raise FileNotFoundError(msg)


def _parse_cspb_truth_line(line: str) -> tuple[int, str] | None:
    """Parse truth file line -> (signal_index, modulation_class)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        idx = int(parts[0].rstrip(".").rstrip(","))
    except ValueError:
        return None
    mod_raw = parts[1].lower()
    mod = CSPB_CLASS_ALIASES.get(mod_raw, mod_raw.upper())
    return idx, mod


def _read_tim_file(path: Path) -> np.ndarray:
    """Read CSPB .tim binary (interleaved float32 real/imag)."""
    return np.fromfile(path, dtype=np.float32)


def load_cspb_streaming(
    path: Path,
    *,
    cache_dir: Path | None = None,
    checksum_file: Path | None = None,
    load_all: bool = False,
    batch_size: int = 256,
    chunks_per_signal: int = 4,
    progress: Progress | None = None,
    progress_task: TaskID | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yield batches from CSPB.ML.2018R2 directory.

    Use R2 not the original CSPB.ML.2018 release (RNG flaw in original).
    """
    if not path.is_dir():
        msg = f"CSPB directory not found: {path}"
        raise FileNotFoundError(msg)
    verify_checksum(path, checksum_file)

    truth_path = _find_truth_file(path)
    truth: dict[int, str] = {}
    for line in truth_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_cspb_truth_line(line)
        if parsed:
            truth[parsed[0]] = parsed[1]

    from rmv.dataset.paths import list_cspb_tim_files

    class_to_idx = {c: i for i, c in enumerate(CSPB_CLASSES)}
    unique_tim = list_cspb_tim_files(path)
    total_tim = len(unique_tim)
    if progress is not None and progress_task is not None:
        progress.update(
            progress_task,
            total=total_tim,
            completed=0,
            description=f"CSPB: reading .tim files (0/{total_tim:,})",
        )

    batch_samples: list[np.ndarray] = []
    batch_labels: list[int] = []
    batch_snr: list[float] = []

    def flush() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not batch_samples:
            return None
        s = np.concatenate(batch_samples, axis=0)
        l = np.array(batch_labels, dtype=np.int32)
        sn = np.array(batch_snr, dtype=np.float32)
        batch_samples.clear()
        batch_labels.clear()
        batch_snr.clear()
        return s, l, sn

    for file_idx, tim_path in enumerate(unique_tim):
        if progress is not None and progress_task is not None and (
            file_idx == 0 or file_idx % 500 == 0 or file_idx == total_tim - 1
        ):
            progress.update(
                progress_task,
                completed=file_idx + 1,
                description=f"CSPB: reading .tim files ({file_idx + 1:,}/{total_tim:,})",
            )
        stem = tim_path.stem
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        mod = truth.get(idx)
        if mod is None or mod not in class_to_idx:
            continue
        label = class_to_idx[mod]
        try:
            raw = _read_tim_file(tim_path)
            chunks = tim_to_chunks(raw, max_chunks=chunks_per_signal)
        except (ValueError, OSError) as exc:
            logger.warning("Skip %s: %s", tim_path, exc)
            continue
        for ch in chunks:
            batch_samples.append(ch[np.newaxis, ...])
            batch_labels.append(label)
            batch_snr.append(0.0)
        if len(batch_labels) >= batch_size:
            out = flush()
            if out:
                yield out

    out = flush()
    if out:
        yield out


def load_cspb(
    path: Path,
    *,
    cache_dir: Path | None = None,
    checksum_file: Path | None = None,
    progress: Progress | None = None,
    progress_task: TaskID | None = None,
) -> IQDataset:
    """Load CSPB.ML.2018R2 (use R2, not original)."""
    from rmv.dataset.paths import find_cspb_truth_file

    truth = find_cspb_truth_file(path)
    truth_tag = str(truth.stat().st_size) if truth is not None else "0"
    cache_key = f"cspb_{path.stat().st_size}_{truth_tag}"
    if cache_dir is not None:
        cache_path = cache_path_for_source(cache_dir, "cspb", cache_key)
        if cache_path.is_file():
            samples, labels, snr = load_cache_shard(cache_path)
            if progress is not None and progress_task is not None:
                progress.update(
                    progress_task,
                    completed=1,
                    total=1,
                    description=f"CSPB: cache hit ({len(labels):,} chunks)",
                )
            return IQDataset(
                samples=samples,
                labels=labels,
                snr_db=snr,
                class_names=CSPB_CLASSES,
                source="cspb",
            )

    all_s: list[np.ndarray] = []
    all_l: list[np.ndarray] = []
    all_sn: list[np.ndarray] = []
    for samples, labels, snr in load_cspb_streaming(
        path,
        cache_dir=cache_dir,
        checksum_file=checksum_file,
        progress=progress,
        progress_task=progress_task,
    ):
        all_s.append(samples)
        all_l.append(labels)
        all_sn.append(snr)
    if not all_s:
        msg = "CSPB loader produced no data"
        raise RuntimeError(msg)
    merged_samples = np.concatenate(all_s, axis=0)
    merged_labels = np.concatenate(all_l, axis=0)
    merged_snr = np.concatenate(all_sn, axis=0)
    if cache_dir is not None:
        save_cache_shard(
            cache_path_for_source(cache_dir, "cspb", cache_key),
            merged_samples,
            merged_labels,
            merged_snr,
        )
    return IQDataset(
        samples=merged_samples,
        labels=merged_labels,
        snr_db=merged_snr,
        class_names=CSPB_CLASSES,
        source="cspb",
    )


def read_iq_binary(path: Path, max_bytes: int | None = None) -> np.ndarray:
    """Read raw .iq interleaved float32 file."""
    if not path.is_file():
        msg = f"IQ file not found: {path}"
        raise FileNotFoundError(msg)
    size = path.stat().st_size
    limit = max_bytes or size
    if size > limit:
        msg = f"IQ file exceeds size limit ({size} > {limit} bytes): {path}"
        raise ValueError(msg)
    return np.fromfile(path, dtype="<f4", count=size // 4)
