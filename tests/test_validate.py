"""Tests for validation pass/fail logic."""

from __future__ import annotations

import numpy as np

from rmv.types import ClassifierResult, IQSidecar
from rmv.validate import (
    build_validation_result,
    evaluate_validation,
    order_matches,
)


def _sidecar() -> IQSidecar:
    return IQSidecar(
        source="test",
        block_name="mod_am",
        expected_family="AM",
        expected_order="AM-DSB",
        sample_rate_hz=48000,
        center_freq_hz=0,
        snr_db=None,
        notes="",
    )


def test_evaluate_correct_low_confidence_passes() -> None:
    """Matching labels must pass even when confidence is below threshold."""
    sidecar = _sidecar()
    prediction = ClassifierResult(
        family="AM",
        family_confidence=0.55,
        order="AM-DSB",
        order_confidence=0.50,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    family_pass, order_pass, hard_fail, reason = evaluate_validation(
        sidecar, prediction, threshold=0.70
    )
    assert family_pass is True
    assert order_pass is True
    assert hard_fail is False
    assert reason is None


def test_evaluate_wrong_order_fails_not_hard() -> None:
    sidecar = _sidecar()
    prediction = ClassifierResult(
        family="AM",
        family_confidence=0.95,
        order="WBFM",
        order_confidence=0.90,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    family_pass, order_pass, hard_fail, _ = evaluate_validation(
        sidecar, prediction, threshold=0.70
    )
    assert family_pass is True
    assert order_pass is False
    assert hard_fail is False


def test_order_matches_gmsk_msk() -> None:
    assert order_matches("GMSK", "MSK")
    assert order_matches("GMSK", "GMSK")
    assert not order_matches("GMSK", "QPSK")


def test_order_matches_cpfsk_gfsk() -> None:
    assert order_matches("CPFSK", "GFSK")
    assert order_matches("CPFSK", "CPFSK")
    assert not order_matches("CPFSK", "QPSK")


def test_evaluate_gmsk_predicted_msk_passes() -> None:
    sidecar = IQSidecar(
        source="test",
        block_name="mod_gmsk",
        expected_family="FSK",
        expected_order="GMSK",
        sample_rate_hz=48000,
        center_freq_hz=0,
        snr_db=None,
        notes="",
    )
    prediction = ClassifierResult(
        family="FSK",
        family_confidence=0.75,
        order="MSK",
        order_confidence=0.63,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    family_pass, order_pass, hard_fail, _ = evaluate_validation(
        sidecar, prediction, threshold=0.70
    )
    assert family_pass is True
    assert order_pass is True
    assert hard_fail is False


def test_evaluate_dpmr_correct_passes() -> None:
    sidecar = IQSidecar(
        source="test",
        block_name="mod_dpmr",
        expected_family="FSK",
        expected_order="dPMR",
        sample_rate_hz=48000,
        center_freq_hz=0,
        snr_db=None,
        notes="",
    )
    prediction = ClassifierResult(
        family="FSK",
        family_confidence=0.54,
        order="dPMR",
        order_confidence=0.54,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    family_pass, order_pass, hard_fail, _ = evaluate_validation(
        sidecar, prediction, threshold=0.70
    )
    assert family_pass is True
    assert order_pass is True
    assert hard_fail is False


def test_build_validation_result_correct_low_confidence() -> None:
    from pathlib import Path

    sidecar = _sidecar()
    prediction = ClassifierResult(
        family="AM",
        family_confidence=0.50,
        order="AM-DSB",
        order_confidence=0.50,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    result = build_validation_result(
        Path("mod_am.iq"), sidecar, prediction, threshold=0.70
    )
    assert result.family_pass is True
    assert result.order_pass is True
    assert result.hard_fail is False
