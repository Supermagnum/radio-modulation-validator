"""Tests for RadioModulationValidator API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rmv.api import RadioModulationValidator, format_summary_markdown
from rmv.types import ClassifierResult, ValidationResult


@pytest.fixture
def mock_classifier(mocker: pytest.MockFixture) -> MagicMock:
    clf = MagicMock()
    clf.classify.return_value = [
        ClassifierResult("FM", 0.95, "NBFM", 0.80, np.zeros(6), np.zeros(20))
    ]
    clf.classify_aggregate.return_value = ClassifierResult(
        "FM", 0.95, "NBFM", 0.80, np.zeros(6), np.zeros(20)
    )
    mocker.patch("rmv.api.ModulationClassifier", return_value=clf)
    return clf


def test_classify_raw(mock_classifier: MagicMock) -> None:
    v = RadioModulationValidator(verify_checksums=False)
    samples = np.random.randn(1, 2, 1024).astype(np.float32)
    results = v.classify(samples)
    assert len(results) == 1
    mock_classifier.classify.assert_called_once()


def test_validate_file(tmp_path: Path, mock_classifier: MagicMock) -> None:
    from tests.fixtures.synthetic_iq import generate_fm, write_test_iq_file

    iq = tmp_path / "mod_nbfm.iq"
    write_test_iq_file(
        iq,
        {
            "source": "gr-qradiolink",
            "block_name": "mod_nbfm",
            "expected_family": "FM",
            "expected_order": "NBFM",
            "sample_rate_hz": 48000,
        },
        [generate_fm()],
    )
    mock_classifier.classify_aggregate.return_value = ClassifierResult(
        "FM", 0.94, "NBFM", 0.71, np.zeros(6), np.zeros(20)
    )
    v = RadioModulationValidator(verify_checksums=False)
    result = v.validate_file(iq)
    assert isinstance(result, ValidationResult)
    assert result.block_name == "mod_nbfm"
    assert result.family_pass is True


def test_summary_report() -> None:
    results = [
        ValidationResult(
            iq_file="a.iq",
            block_name="mod_nbfm",
            source_repo="gr-qradiolink",
            expected_family="FM",
            expected_order="NBFM",
            predicted_family="FM",
            predicted_order="NBFM",
            family_confidence=0.94,
            order_confidence=0.71,
            family_pass=True,
            order_pass=True,
            snr_db=None,
            timestamp="2026-05-30T12:00:00Z",
            notes="",
        )
    ]
    v = RadioModulationValidator(verify_checksums=False)
    summary = v.summary_report(results)
    assert summary["schema_version"] == "1.0"
    assert summary["passed"] == 1
    md = format_summary_markdown(results)
    assert "mod_nbfm" in md
