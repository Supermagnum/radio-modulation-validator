"""Tests for pre-export family checkpoint verification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rmv.verify_checkpoint import (
    RADIOML_VERIFY,
    SYNTHETIC_VERIFY,
    verify_family_checkpoint,
)


@pytest.mark.skipif(
    not Path("checkpoints/best_family_classifier.pt").is_file(),
    reason="family checkpoint not present",
)
def test_verify_family_on_real_checkpoint() -> None:
    radioml = Path("datasets/radioml/RML2016.10a_dict.pkl")
    if not radioml.is_file():
        pytest.skip("RadioML pickle not present")
    results = verify_family_checkpoint(
        Path("checkpoints"),
        radioml_pkl=radioml,
        use_gnuradio=False,
    )
    by_label = {r.case.label: r for r in results}
    assert by_label["synthetic:WBFM"].passed
    assert by_label["synthetic:BPSK"].passed
    assert by_label["radioml:QAM16@10dB"].passed
    skip = by_label["radioml:WBFM (excluded)"]
    assert skip.passed and skip.predicted == "skipped"


def test_radioml_verify_excludes_wbfm_bpsk() -> None:
    mods = {m for m, _, _ in RADIOML_VERIFY}
    assert "WBFM" not in mods
    assert "BPSK" not in mods
    assert "WBFM" in {c for c, _ in SYNTHETIC_VERIFY}
