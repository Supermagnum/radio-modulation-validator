"""CLI tests using typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from rmv.cli import app
from rmv.types import ClassifierResult, ValidationResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout


def test_checksum_verify_missing(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["checksum", "verify", "--models-dir", str(tmp_path / "models")],
    )
    assert result.exit_code == 1


def test_classify_json(runner: CliRunner, tmp_path: Path, mocker: pytest.MockFixture) -> None:
    from tests.fixtures.synthetic_iq import generate_bpsk, write_test_iq_file

    iq = tmp_path / "test.iq"
    write_test_iq_file(
        iq,
        {"source": "other", "block_name": "t", "expected_family": "PSK",
         "expected_order": "BPSK", "sample_rate_hz": 48000},
        [generate_bpsk()],
    )
    mock_v = MagicMock()
    mock_v.classify_file.return_value = ClassifierResult(
        "PSK", 0.9, "BPSK", 0.85, np.zeros(6), np.zeros(20)
    )
    mocker.patch("rmv.cli.RadioModulationValidator", return_value=mock_v)
    result = runner.invoke(app, ["classify", str(iq), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["family"] == "PSK"


def test_validate_command(runner: CliRunner, tmp_path: Path, mocker: pytest.MockFixture) -> None:
    mocker.patch(
        "rmv.cli.run_validate_cli",
        return_value=0,
    )
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 0


def test_report_markdown(runner: CliRunner, tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "gr-qradiolink"
    results_dir.mkdir(parents=True)
    payload = {
        "schema_version": "1.0",
        "iq_file": "iq_samples/x.iq",
        "block_name": "mod_nbfm",
        "source_repo": "gr-qradiolink",
        "expected_family": "FM",
        "expected_order": "NBFM",
        "predicted_family": "FM",
        "predicted_order": "NBFM",
        "family_confidence": 0.94,
        "order_confidence": 0.71,
        "family_pass": True,
        "order_pass": True,
        "snr_db": None,
        "timestamp": "2026-05-30T12:00:00Z",
        "notes": "",
    }
    (results_dir / "mod_nbfm.json").write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(app, ["report", str(tmp_path / "results")])
    assert result.exit_code == 0
    assert "mod_nbfm" in result.stdout
