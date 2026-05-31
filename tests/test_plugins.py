"""Tests for custom-mode plugins."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.signal import lfilter
from typer.testing import CliRunner

from rmv.cli import app
from rmv.plugins.registry import get, list_plugins
from rmv.plugins.sleipnir_8qpsk import (
    EXPECTED_CARRIER_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    Sleipnir8QPSKValidator,
    compute_confidence_scores,
)
from rmv.validate import (
    build_validation_result_from_custom,
    is_custom_mode_sidecar,
    run_validate_file,
)
from rmv.types import IQSidecar

CHUNK_SAMPLES = 1024


def _rrc_pulse(sps: int, alpha: float = 0.35, num_taps: int | None = None) -> np.ndarray:
    n = num_taps or (8 * sps + 1)
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


def _qpsk_baseband(n_samples: int, fs: float, baud: float, rng: np.random.Generator) -> np.ndarray:
    sps = max(4, int(round(fs / baud)))
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2)
    n_sym = n_samples // sps + 2
    syms = constellation[rng.integers(0, 4, size=n_sym)]
    upsampled = np.zeros(n_sym * sps, dtype=np.complex64)
    upsampled[::sps] = syms
    shaped = lfilter(_rrc_pulse(sps), 1.0, upsampled)
    return shaped[:n_samples].astype(np.complex64)


def synth_multicarrier_chunks(
    carrier_hz: list[float],
    *,
    n_chunks: int = 24,
    fs: float = DEFAULT_SAMPLE_RATE_HZ,
    baud: float = 900.0,
    seed: int = 0,
) -> np.ndarray:
    """Build (N, 2, 1024) float32 composite multi-carrier QPSK."""
    n = n_chunks * CHUNK_SAMPLES
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(seed)
    composite = np.zeros(n, dtype=np.complex64)
    for fc in carrier_hz:
        bb = _qpsk_baseband(n, fs, baud, rng)
        composite += bb * np.exp(2j * np.pi * fc * t).astype(np.complex64)
    peak = float(np.max(np.abs(composite))) or 1.0
    composite = (composite / peak * 0.8).astype(np.complex64)
    chunks = np.zeros((n_chunks, 2, CHUNK_SAMPLES), dtype=np.float32)
    for i in range(n_chunks):
        seg = composite[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        chunks[i, 0] = seg.real.astype(np.float32)
        chunks[i, 1] = seg.imag.astype(np.float32)
    return chunks


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_plugin_registry_lists_sleipnir() -> None:
    assert "sleipnir_8qpsk" in list_plugins()
    plugin = get("sleipnir_8qpsk")
    assert plugin is not None
    assert plugin.mode_id == "sleipnir_8qpsk"


def test_sleipnir_confidence_above_threshold() -> None:
    samples = synth_multicarrier_chunks(EXPECTED_CARRIER_HZ, n_chunks=24, seed=1)
    plugin = Sleipnir8QPSKValidator()
    result = plugin.validate(samples, DEFAULT_SAMPLE_RATE_HZ, {})
    assert result.mode_id == "sleipnir_8qpsk"
    assert result.metrics["carrier_count"] == 8
    assert result.pass_overall is True
    assert result.confidence >= 0.70


def test_sleipnir_scan_reference_iq_passes(tmp_path: Path) -> None:
    """Regression: scan-generated sleipnir reference IQ must pass the plugin."""
    ref = Path(".scan_iq/gr-sleipnir/mod_sleipnir_8qpsk.iq")
    if not ref.is_file():
        pytest.skip("scan IQ not present (run rmv scan first)")
    from rmv.iq_io import load_iq_chunks_from_path

    chunks = load_iq_chunks_from_path(ref)
    result = Sleipnir8QPSKValidator().validate(chunks, DEFAULT_SAMPLE_RATE_HZ, {})
    assert result.pass_overall is True
    assert result.confidence >= 0.70


def test_sleipnir_single_carrier_fails() -> None:
    samples = synth_multicarrier_chunks([0.0], n_chunks=24, seed=2)
    plugin = Sleipnir8QPSKValidator()
    result = plugin.validate(samples, DEFAULT_SAMPLE_RATE_HZ, {})
    assert result.metrics["carrier_count"] != 8
    assert result.pass_overall is False


def test_confidence_no_zero_collapse() -> None:
    """One marginal metric must not drive confidence to near zero."""
    scores = compute_confidence_scores(
        {
            "carrier_count": 8,
            "carrier_spacing_mean_hz": 1300.0,
            "carrier_spacing_std_hz": 90.0,
            "symbol_rate_pass": [True] * 4 + [False] * 4,
            "qpsk_pass": [True] * 8,
            "total_bandwidth_hz": 10500.0,
        }
    )
    assert scores["symbol_rate"] == 0.5
    assert scores["confidence"] >= 0.65


def test_sleipnir_plugin_fails_on_wrong_spacing() -> None:
    carriers = [
        -8750.0,
        -6250.0,
        -3750.0,
        -1250.0,
        1250.0,
        3750.0,
        6250.0,
        8750.0,
    ]
    samples = synth_multicarrier_chunks(carriers, n_chunks=12, seed=3)
    plugin = Sleipnir8QPSKValidator()
    result = plugin.validate(samples, DEFAULT_SAMPLE_RATE_HZ, {})
    assert result.pass_overall is False
    assert result.metrics["carrier_count"] <= 8


def test_custom_mode_routed_correctly(tmp_path: Path, mocker: pytest.MockFixture) -> None:
    iq = tmp_path / "tx.iq"
    sidecar = tmp_path / "tx.json"
    samples = synth_multicarrier_chunks(EXPECTED_CARRIER_HZ, n_chunks=8, seed=4)
    iq.write_bytes(samples.astype(np.float32).tobytes())

    sidecar.write_text(
        json.dumps(
            {
                "source": "gr-sleipnir",
                "block_name": "SleipnirTxHier",
                "expected_family": "custom",
                "expected_order": "sleipnir_8qpsk",
                "sample_rate_hz": 48000,
                "center_freq_hz": 0,
            }
        ),
        encoding="utf-8",
    )

    mock_classifier = MagicMock()
    mock_classifier.classify_aggregate = MagicMock(side_effect=AssertionError("CNN should not run"))
    mocker.patch("rmv.validate.load_iq_chunks", return_value=samples)

    result = run_validate_file(iq, mock_classifier, threshold=0.5)
    assert result.custom_mode is not None
    assert result.custom_mode["mode_id"] == "sleipnir_8qpsk"
    assert result.predicted_family == "custom"
    mock_classifier.classify_aggregate.assert_not_called()


def test_is_custom_mode_sidecar() -> None:
    sc = IQSidecar(
        source="gr-sleipnir",
        block_name="x",
        expected_family="custom",
        expected_order="sleipnir_8qpsk",
        sample_rate_hz=48000,
    )
    assert is_custom_mode_sidecar(sc) is True


def test_build_validation_result_from_custom() -> None:
    from rmv.plugins.base import CustomModeResult

    sc = IQSidecar(
        source="gr-sleipnir",
        block_name="x",
        expected_family="custom",
        expected_order="sleipnir_8qpsk",
        sample_rate_hz=48000,
    )
    custom = CustomModeResult(
        mode_id="sleipnir_8qpsk",
        pass_overall=True,
        confidence=0.87,
        metrics={"carrier_count": 8},
    )
    vr = build_validation_result_from_custom(Path("f.iq"), sc, custom, threshold=0.7)
    assert vr.family_pass and vr.order_pass
    assert vr.custom_mode is not None


def test_cli_plugins_list(runner: CliRunner) -> None:
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    assert "sleipnir_8qpsk" in (result.stdout or result.output)
