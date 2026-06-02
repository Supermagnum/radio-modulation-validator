"""Tests for validation pass/fail logic."""

from __future__ import annotations

import numpy as np

from rmv.types import ClassifierResult, IQSidecar
from rmv.validate import (
    build_validation_result,
    evaluate_validation,
    family_matches,
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
    assert order_matches("GMSK", "GMSK_BT05")
    assert order_matches("GMSK_BT03", "GMSK")
    assert order_matches("GMSK_BT05", "GMSK_BT03")
    assert order_matches("GMSK", "dPMR")
    assert order_matches("GMSK", "NXDN")
    assert order_matches("GMSK_BT05", "dPMR")
    assert not order_matches("GMSK", "QPSK")


def test_order_matches_cpfsk_gfsk() -> None:
    assert order_matches("CPFSK", "GFSK")
    assert order_matches("CPFSK", "CPFSK")
    assert not order_matches("CPFSK", "QPSK")


def test_order_matches_am_dsb_aviation() -> None:
    assert order_matches("AM-DSB", "AM_AIR_833")
    assert order_matches("AM_AIR_25K", "AM-DSB")
    assert order_matches("AM_AIR_833", "AM_AIR_25K")
    assert not order_matches("AM-DSB", "WBFM")


def test_order_matches_nxdn_dpmr_symmetric() -> None:
    assert order_matches("NXDN", "dPMR")
    assert order_matches("dPMR", "NXDN")
    assert order_matches("NXDN", "NXDN")
    assert order_matches("GMSK", "NXDN")
    assert order_matches("GMSK", "dPMR")
    assert not order_matches("NXDN", "GMSK")


def test_family_matches_am_ssb_accepts_fm_and_qam() -> None:
    assert family_matches("AM", "AM-SSB", "AM")
    assert family_matches("AM", "AM-SSB", "FM")
    assert family_matches("AM", "AM-SSB", "QAM")
    assert family_matches("AM", "USB", "QAM")
    assert family_matches("AM", "LSB", "FM")
    assert not family_matches("AM", "AM-SSB", "PSK")
    assert not family_matches("AM", "AM-DSB", "FM")
    assert not family_matches("AM", "AM-DSB", "QAM")
    assert not family_matches("FM", "WBFM", "AM")


def test_order_matches_am_ssb_scan_ambiguity() -> None:
    assert order_matches("AM-SSB", "WBFM")
    assert order_matches("AM-SSB", "NFM_DCS")
    assert order_matches("AM-SSB", "AM-SSB")


def test_evaluate_gmsk_predicted_dpmr_passes() -> None:
    sidecar = IQSidecar(
        source="test",
        block_name="mod_d_star",
        expected_family="FSK",
        expected_order="GMSK",
        sample_rate_hz=48000,
        center_freq_hz=0,
        snr_db=None,
        notes="",
    )
    prediction = ClassifierResult(
        family="FSK",
        family_confidence=0.90,
        order="dPMR",
        order_confidence=0.55,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    family_pass, order_pass, hard_fail, _ = evaluate_validation(
        sidecar, prediction, threshold=0.70
    )
    assert family_pass is True
    assert order_pass is True
    assert hard_fail is False


def test_evaluate_am_ssb_fm_family_passes() -> None:
    sidecar = IQSidecar(
        source="test",
        block_name="mod_ssb",
        expected_family="AM",
        expected_order="AM-SSB",
        sample_rate_hz=48000,
        center_freq_hz=0,
        snr_db=None,
        notes="",
    )
    prediction = ClassifierResult(
        family="FM",
        family_confidence=0.96,
        order="NFM_DCS",
        order_confidence=0.92,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    family_pass, order_pass, hard_fail, _ = evaluate_validation(
        sidecar, prediction, threshold=0.70
    )
    assert family_pass is True
    assert order_pass is True
    assert hard_fail is False


def test_evaluate_am_ssb_qam_family_passes() -> None:
    sidecar = IQSidecar(
        source="test",
        block_name="mod_ssb",
        expected_family="AM",
        expected_order="AM-SSB",
        sample_rate_hz=48000,
        center_freq_hz=0,
        snr_db=None,
        notes="",
    )
    prediction = ClassifierResult(
        family="QAM",
        family_confidence=0.88,
        order="QAM16",
        order_confidence=0.75,
        family_logits=np.zeros(1),
        order_logits=np.zeros(1),
    )
    family_pass, order_pass, hard_fail, _ = evaluate_validation(
        sidecar, prediction, threshold=0.70
    )
    assert family_pass is True
    assert order_pass is False
    assert hard_fail is False


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
