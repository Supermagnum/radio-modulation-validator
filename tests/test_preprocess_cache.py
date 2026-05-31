"""Tests for versioned dataset cache paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rmv.dataset.preprocess import (
    CACHE_FILE_SUFFIX,
    PROCESSING_VERSION,
    cache_path_for_source,
    clean_stale_cache,
    is_current_cache_file,
    save_cache_shard,
    try_load_cache_shard,
)


def test_cache_path_includes_processing_version(tmp_path: Path) -> None:
    path = cache_path_for_source(tmp_path, "radioml2016", "radioml_12345")
    assert PROCESSING_VERSION in path.name
    assert path.name.endswith(CACHE_FILE_SUFFIX)
    assert path.parent == tmp_path / "radioml2016"


def test_clean_stale_cache_removes_old_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "radioml2016"
    source_dir.mkdir(parents=True)
    stale = source_dir / "radioml_old.npy.npz"
    stale.write_bytes(b"stale")
    legacy = source_dir / "radioml_legacy.npy"
    legacy.write_bytes(b"legacy")

    current = cache_path_for_source(tmp_path, "radioml2016", "radioml_new")
    samples = np.zeros((2, 2, 1024), dtype=np.float32)
    save_cache_shard(current, samples, np.array([0, 1], dtype=np.int32), np.zeros(2, dtype=np.float32))

    removed, kept = clean_stale_cache(tmp_path)
    assert removed == 2
    assert kept == 1
    assert not stale.is_file()
    assert not legacy.is_file()
    assert current.is_file()


def test_is_current_cache_file() -> None:
    current = Path(f"shard_abc12345{CACHE_FILE_SUFFIX}")
    assert is_current_cache_file(current)
    assert not is_current_cache_file(Path("shard_old.npy.npz"))


def test_try_load_cache_shard_removes_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / f"bad{CACHE_FILE_SUFFIX}"
    corrupt.write_bytes(b"not a valid npz")
    assert try_load_cache_shard(corrupt) is None
    assert not corrupt.is_file()


def test_save_cache_shard_roundtrip(tmp_path: Path) -> None:
    from rmv.dataset.preprocess import load_cache_shard

    path = cache_path_for_source(tmp_path, "test", "shard")
    samples = np.random.randn(4, 2, 1024).astype(np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    snr = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float32)
    save_cache_shard(path, samples, labels, snr)
    loaded = load_cache_shard(path)
    np.testing.assert_array_equal(loaded[0], samples)
    np.testing.assert_array_equal(loaded[1], labels)
    np.testing.assert_array_equal(loaded[2], snr)
