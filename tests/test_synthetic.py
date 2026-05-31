"""Tests for synthetic NBFM and aviation AM generation."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from rmv.dataset.synthetic import (
    BROADCAST_FM_TAU,
    MODE_TO_CLASS,
    PROTOCOL_4FSK_ORDERS,
    VARIANT_SPECS,
    aviation_carrier_to_sideband_ratio,
    generate_aviation_am_25k,
    generate_aviation_am_833,
    generate_synthetic,
    generate_variant_chunks,
    load_synthetic,
    measure_audio_bandwidth_hz,
    save_synthetic_dataset,
    validate_nbfm_params,
    verify_4fsk_signal,
    verify_bandwidth,
)
from rmv.types import IQDataset


def _broadcast_am_dsb_chunk(rng: np.random.Generator) -> np.ndarray:
    """Wideband broadcast-style AM-DSB for comparison (not aviation)."""
    n = 1024
    t = np.arange(n, dtype=np.float64) / 48000.0
    audio = rng.standard_normal(n).astype(np.float32)
    sos = __import__("scipy.signal", fromlist=["butter"]).butter(
        4, [200.0, 8000.0], btype="bandpass", fs=48000.0, output="sos"
    )
    audio = __import__("scipy.signal", fromlist=["sosfilt"]).sosfilt(sos, audio).astype(np.float32)
    audio = audio / max(float(np.max(np.abs(audio))), 1e-6) * 0.9
    env = (1.0 + 0.9 * audio).astype(np.complex64)
    chunk = np.stack([env.real, env.imag], axis=0).astype(np.float32)
    return chunk


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.mark.parametrize(
    ("class_name", "limit_hz"),
    [
        ("NBFM_25", 7000.0),
        ("NBFM_50", 13000.0),
        ("AM_AIR_25K", 8000.0),
        ("AM_AIR_833", 6500.0),
    ],
)
def test_bandwidth_within_limit(
    class_name: str,
    limit_hz: float,
    rng: np.random.Generator,
) -> None:
    chunks = generate_variant_chunks(
        class_name, 80, apply_channel=False, use_gnuradio=False, rng=rng
    )
    verify_bandwidth(chunks, 48000.0, limit_hz, class_name)


def test_wrong_tau_raises() -> None:
    with pytest.raises(ValueError, match="broadcast FM preemphasis"):
        validate_nbfm_params(BROADCAST_FM_TAU, 2500.0, class_name="NBFM_25")


def test_wrong_nbfm_deviation_raises(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="max_dev"):
        validate_nbfm_params(0.0, 75000.0, class_name="NBFM_25")
    chunks = generate_variant_chunks(
        "NBFM_25",
        40,
        max_dev=75000.0,
        apply_channel=False,
        use_gnuradio=False,
        enforce_params=False,
        rng=rng,
    )
    with pytest.raises(ValueError, match="bandwidth"):
        verify_bandwidth(chunks, 48000.0, 7000.0, "NBFM_25")


def test_am_air_833_narrower_than_25k(rng: np.random.Generator) -> None:
    c25 = generate_aviation_am_25k(60, rng=rng, apply_channel=False)
    c833 = generate_aviation_am_833(60, rng=rng, apply_channel=False)
    verify_bandwidth(c25, 48000.0, 8000.0, "AM_AIR_25K")
    verify_bandwidth(c833, 48000.0, 6500.0, "AM_AIR_833")
    from rmv.dataset.synthetic import _occupied_bandwidth_hz

    bw25 = np.percentile(
        [_occupied_bandwidth_hz(c25[i, 0] + 1j * c25[i, 1], 48000.0) for i in range(60)],
        95,
    )
    bw833 = np.percentile(
        [_occupied_bandwidth_hz(c833[i, 0] + 1j * c833[i, 1], 48000.0) for i in range(60)],
        95,
    )
    assert bw833 < bw25


def test_am_air_audio_bandlimit_25k(rng: np.random.Generator) -> None:
    audio_rate = 8000
    n = audio_rate
    audio = rng.standard_normal(n).astype(np.float32)
    from rmv.dataset.synthetic import _bandlimit_audio_aviation

    limited = _bandlimit_audio_aviation(audio, float(audio_rate), "25k")
    bw = measure_audio_bandwidth_hz(limited, float(audio_rate))
    assert bw <= 3600.0


def test_am_air_audio_bandlimit_833(rng: np.random.Generator) -> None:
    audio_rate = 8000
    n = audio_rate
    audio = rng.standard_normal(n).astype(np.float32)
    from rmv.dataset.synthetic import _bandlimit_audio_aviation

    limited = _bandlimit_audio_aviation(audio, float(audio_rate), "833")
    bw = measure_audio_bandwidth_hz(limited, float(audio_rate))
    assert bw <= 2800.0


def test_am_air_full_carrier_present(rng: np.random.Generator) -> None:
    chunks = generate_aviation_am_25k(20, rng=rng, apply_channel=False)
    ratios = [aviation_carrier_to_sideband_ratio(chunks[i]) for i in range(20)]
    assert all(r > 2.0 for r in ratios)


def _envelope_modulation_bandwidth_hz(chunk: np.ndarray, sample_rate_hz: float = 48000.0) -> float:
    env = chunk[0].astype(np.float64)
    ac = env - np.mean(env)
    return measure_audio_bandwidth_hz(ac.astype(np.float32), sample_rate_hz)


def test_am_air_distinct_from_radioml_am_dsb(rng: np.random.Generator) -> None:
    air = generate_aviation_am_25k(1, rng=rng, apply_channel=False)[0]
    broadcast = _broadcast_am_dsb_chunk(rng)
    bw_air = _envelope_modulation_bandwidth_hz(air)
    bw_bc = _envelope_modulation_bandwidth_hz(broadcast)
    assert bw_bc > bw_air * 1.2
    assert "AM-DSB" != "AM_AIR_25K"


def test_output_shape(rng: np.random.Generator) -> None:
    for name in VARIANT_SPECS:
        chunks = generate_variant_chunks(name, 3, rng=rng, use_gnuradio=False)
        assert chunks.shape == (3, 2, 1024)
        assert chunks.dtype == np.float32


def test_source_field(tmp_path: Path, rng: np.random.Generator) -> None:
    ds = generate_synthetic(
        ["nbfm25"],
        chunks_per_snr=2,
        snr_levels=[0.0, 10.0],
        verify=False,
        use_gnuradio=False,
        seed=1,
    )
    assert ds.source == "synthetic"
    save_synthetic_dataset(tmp_path, ds)
    loaded = load_synthetic(tmp_path)
    assert loaded.source == "synthetic"
    assert loaded.samples.shape[1:] == (2, 1024)


OOT_MODULE_FORBIDDEN = (
    "qradiolink",
    "gr_qradiolink",
    "packet_protocols",
    "gr_packet_protocols",
    "sleipnir",
    "gr_sleipnir",
)


def test_no_oot_imports() -> None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "rmv" / "dataset" / "synthetic.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.lower()
                for forbidden in OOT_MODULE_FORBIDDEN:
                    assert forbidden not in name
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.lower()
            for forbidden in OOT_MODULE_FORBIDDEN:
                assert forbidden not in mod
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for forbidden in OOT_MODULE_FORBIDDEN:
                assert forbidden not in node.func.id.lower()
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            for forbidden in OOT_MODULE_FORBIDDEN:
                if forbidden in node.value.id.lower():
                    raise AssertionError(
                        f"forbidden OOT attribute access in synthetic.py: {node.value.id}"
                    )


def _inst_freq_std_hz(chunks: np.ndarray, sample_rate_hz: float = 48000.0) -> float:
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    phase = np.unwrap(np.angle(iq), axis=1)
    inst_freq = np.diff(phase, axis=1) * sample_rate_hz / (2 * np.pi)
    return float(inst_freq.std())


def _envelope_variation(chunks: np.ndarray) -> float:
    iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
    envelope = np.abs(iq)
    return float(envelope.std() / max(envelope.mean(), 1e-12))


def test_dmr_inst_freq_range(rng: np.random.Generator) -> None:
    chunks = generate_variant_chunks(
        "DMR", 40, snr_db=20.0, apply_channel=False, use_gnuradio=False, rng=rng
    )
    std = _inst_freq_std_hz(chunks)
    assert 600.0 <= std <= 2500.0


def test_m17_inst_freq_range(rng: np.random.Generator) -> None:
    chunks = generate_variant_chunks(
        "M17", 40, snr_db=20.0, apply_channel=False, use_gnuradio=False, rng=rng
    )
    std = _inst_freq_std_hz(chunks)
    assert 800.0 <= std <= 3000.0


def test_nxdn_inst_freq_range(rng: np.random.Generator) -> None:
    chunks = generate_variant_chunks(
        "NXDN", 40, snr_db=20.0, apply_channel=False, use_gnuradio=False, rng=rng
    )
    std = _inst_freq_std_hz(chunks)
    assert 300.0 <= std <= 1500.0


@pytest.mark.parametrize("class_name", sorted(PROTOCOL_4FSK_ORDERS))
def test_4fsk_constant_envelope(class_name: str, rng: np.random.Generator) -> None:
    chunks = generate_variant_chunks(
        class_name, 30, snr_db=20.0, apply_channel=False, use_gnuradio=False, rng=rng
    )
    assert _envelope_variation(chunks) < 0.10


def test_dmr_vs_nxdn_different_symbol_rate(rng: np.random.Generator) -> None:
    """4800 baud DMR has wider inst-freq modulation bandwidth than 2400 baud NXDN."""
    dmr = generate_variant_chunks(
        "DMR", 60, snr_db=20.0, apply_channel=False, use_gnuradio=False, rng=rng
    )
    nxdn = generate_variant_chunks(
        "NXDN", 60, snr_db=20.0, apply_channel=False, use_gnuradio=False, rng=rng
    )

    def inst_freq_modulation_bw_hz(chunks: np.ndarray, sample_rate_hz: float = 48000.0) -> float:
        iq = chunks[:, 0, :] + 1j * chunks[:, 1, :]
        phase = np.unwrap(np.angle(iq), axis=1)
        inst = np.diff(phase, axis=1).reshape(-1) * sample_rate_hz / (2 * np.pi)
        inst = inst - np.mean(inst)
        spec = np.abs(np.fft.rfft(inst)) ** 2
        freqs = np.fft.rfftfreq(len(inst), 1.0 / sample_rate_hz)
        cum = np.cumsum(spec) / max(float(np.sum(spec)), 1e-12)
        idx = int(np.searchsorted(cum, 0.9))
        return float(freqs[min(idx, len(freqs) - 1)])

    dmr_bw = inst_freq_modulation_bw_hz(dmr)
    nxdn_bw = inst_freq_modulation_bw_hz(nxdn)
    assert dmr_bw > nxdn_bw * 1.4
    assert dmr_bw > 1500.0
    assert nxdn_bw < 1300.0


def test_generate_synthetic_class_names() -> None:
    ds = generate_synthetic(
        chunks_per_snr=1,
        snr_levels=[10.0],
        verify=False,
        use_gnuradio=False,
        seed=0,
    )
    assert set(ds.class_names) == set(MODE_TO_CLASS.values())
    assert isinstance(ds, IQDataset)
    assert len(ds.samples) == len(MODE_TO_CLASS) * 1 * 1
