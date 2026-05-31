"""Generate synthetic IQ data for unit tests (no trained models required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
def generate_bpsk(n_samples: int = 1024, sps: int = 8) -> np.ndarray:
    """Generate BPSK IQ chunk shape (2, n_samples)."""
    bits = np.random.randint(0, 2, n_samples // sps + 1)
    symbols = 2 * bits - 1
    pulse = np.repeat(symbols, sps)[:n_samples]
    i = pulse.astype(np.float32)
    q = np.zeros_like(i)
    return np.stack([i, q], axis=0)


def generate_qpsk(n_samples: int = 1024, sps: int = 8) -> np.ndarray:
    """Generate QPSK IQ chunk."""
    bits = np.random.randint(0, 4, n_samples // sps + 1)
    mapping = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    symbols = mapping[bits]
    pulse = np.repeat(symbols, sps)[:n_samples]
    return np.stack([pulse.real, pulse.imag], axis=0).astype(np.float32)


def generate_fm(n_samples: int = 1024, fs: float = 48000.0) -> np.ndarray:
    """Generate narrowband FM-like IQ chunk."""
    t = np.arange(n_samples) / fs
    freq = 1000.0 * np.sin(2 * np.pi * 200 * t)
    phase = 2 * np.pi * np.cumsum(freq) / fs
    i = np.cos(phase).astype(np.float32)
    q = np.sin(phase).astype(np.float32)
    return np.stack([i, q], axis=0)


def generate_2fsk(n_samples: int = 1024, sps: int = 32) -> np.ndarray:
    """Generate 2FSK IQ chunk."""
    bits = np.random.randint(0, 2, n_samples // sps + 1)
    f0, f1 = 2000.0, 4000.0
    fs = 48000.0
    t = np.arange(n_samples) / fs
    freqs = np.where(np.repeat(bits, sps)[:n_samples], f1, f0)
    phase = 2 * np.pi * np.cumsum(freqs) / fs
    i = np.cos(phase).astype(np.float32)
    q = np.sin(phase).astype(np.float32)
    return np.stack([i, q], axis=0)


def interleave_iq(iq: np.ndarray) -> np.ndarray:
    """Convert (2, L) to interleaved float32."""
    return np.stack([iq[0], iq[1]], axis=1).reshape(-1).astype("<f4")


def write_test_iq_file(
    path: Path,
    sidecar: dict[str, object],
    chunks: list[np.ndarray],
) -> None:
    """Write .iq file and .json sidecar for validation tests."""
    data = np.concatenate([interleave_iq(c) for c in chunks])
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)
    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def radioml_mock_pickle(path: Path) -> None:
    """Write minimal RadioML-format pickle for loader tests."""
    import pickle

    classes = ["BPSK", "QPSK"]
    data: dict[tuple[str, str], np.ndarray] = {}
    for mod in classes:
        for snr in ("0", "10"):
            arr = np.random.randn(4, 2, 128).astype(np.float32)
            data[(mod, snr)] = arr
    with path.open("wb") as f:
        pickle.dump(data, f)


def hisarmod_mock_hdf5(path: Path) -> None:
    """Write minimal HISARMOD-like HDF5."""
    import h5py

    with h5py.File(path, "w") as hf:
        grp = hf.create_group("BPSK")
        ds = grp.create_dataset("snr0", data=np.random.randn(2, 2, 1024).astype(np.float32))


def cspb_mock_dir(path: Path) -> None:
    """Write minimal CSPB batch with truth file and one .tim signal."""
    path.mkdir(parents=True, exist_ok=True)
    truth = path / "signal_record.txt"
    truth.write_text(
        "1 bpsk 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )
    n = 1024 * 4
    raw = np.random.randn(n * 2).astype(np.float32)
    (path / "signal_1.tim").write_bytes(raw.tobytes())
