"""Tests for dataset loaders using synthetic/mock data."""

from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest

from rmv.constants import RADIOML_CLASSES
from rmv.dataset.loader import (
    load_cspb,
    load_hisarmod,
    load_radioml,
    load_radioml_streaming,
)
from rmv.dataset.preprocess import chunk_iq_file, normalise_unit_power, upsample_iq_128_to_1024
from tests.fixtures.synthetic_iq import cspb_mock_dir, hisarmod_mock_hdf5, radioml_mock_pickle


def test_normalise_unit_power() -> None:
    x = np.random.randn(4, 2, 1024).astype(np.float32)
    y = normalise_unit_power(x)
    per_sample_power = np.mean(np.sum(y**2, axis=1), axis=1)
    assert np.allclose(per_sample_power, 1.0, atol=1e-5)


def test_upsample_128_to_1024() -> None:
    x = np.random.randn(2, 2, 128).astype(np.float32)
    y = upsample_iq_128_to_1024(x)
    assert y.shape == (2, 2, 1024)


def test_chunk_iq_file_deinterleaves_correctly() -> None:
    from tests.fixtures.synthetic_iq import generate_bpsk, interleave_iq

    source = normalise_unit_power(generate_bpsk()[np.newaxis, ...])[0]
    raw = interleave_iq(source)
    loaded = chunk_iq_file(raw)
    assert loaded.shape == (1, 2, 1024)
    np.testing.assert_allclose(loaded[0], source, atol=1e-5)


def test_chunk_iq_file_multiple_chunks() -> None:
    from tests.fixtures.synthetic_iq import generate_bpsk, interleave_iq

    n_chunks = 3
    parts = [
        normalise_unit_power(generate_bpsk()[np.newaxis, ...])[0] for _ in range(n_chunks)
    ]
    raw = np.concatenate([interleave_iq(p) for p in parts])
    chunks = chunk_iq_file(raw)
    assert chunks.shape == (n_chunks, 2, 1024)


def test_radioml_loader(tmp_path: Path) -> None:
    pkl = tmp_path / "RML2016.10a_dict.pkl"
    data: dict[tuple[str, str], np.ndarray] = {}
    for mod in RADIOML_CLASSES[:3]:
        data[(mod, "0")] = np.random.randn(2, 2, 128).astype(np.float32)
    with pkl.open("wb") as f:
        pickle.dump(data, f)
    ds = load_radioml(pkl, cache_dir=tmp_path / "cache")
    assert ds.source == "radioml2016"
    assert ds.samples.shape[1:] == (2, 1024)
    assert len(ds.labels) == ds.samples.shape[0]


def test_radioml_streaming(tmp_path: Path) -> None:
    pkl = tmp_path / "mock.pkl"
    radioml_mock_pickle(pkl)
    batches = list(load_radioml_streaming(pkl, batch_size=2))
    assert len(batches) >= 1
    samples, labels, snr = batches[0]
    assert samples.ndim == 3


def test_hisarmod_loader(tmp_path: Path) -> None:
    h5 = tmp_path / "hisar.h5"
    hisarmod_mock_hdf5(h5)
    with h5py.File(h5, "a") as hf:
        if "BPSK" in hf:
            grp = hf["BPSK"]
            if "snr0" in grp:
                del grp["snr0"]
            grp.create_dataset("snr0", data=np.random.randn(3, 2, 1024).astype(np.float32))
    ds = load_hisarmod(h5, cache_dir=tmp_path / "cache")
    assert ds.source == "hisarmod"
    assert ds.samples.shape[2] == 1024


def test_cspb_loader(tmp_path: Path) -> None:
    cspb_mock_dir(tmp_path / "cspb")
    with pytest.raises(FileNotFoundError):
        load_cspb(tmp_path / "cspb_empty")
    cspb_mock_dir(tmp_path / "cspb2")
    ds = load_cspb(tmp_path / "cspb2", cache_dir=tmp_path / "cache")
    assert ds.source == "cspb"
    assert ds.samples.ndim == 3
