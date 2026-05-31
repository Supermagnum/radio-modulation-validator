"""Tests for label constants and family mappings."""

from __future__ import annotations

from rmv.constants import RADIOML_CLASSES, RADIOML_TO_FAMILY
from rmv.train import _order_to_family


def test_radioml_qam_pickle_names_map_to_qam_family() -> None:
    assert "QAM16" in RADIOML_CLASSES
    assert "QAM64" in RADIOML_CLASSES
    for name in ("QAM16", "QAM64", "16QAM", "64QAM"):
        assert RADIOML_TO_FAMILY[name] == "QAM"
    assert _order_to_family("QAM16", "radioml2016") == "QAM"
    assert _order_to_family("QAM64", "radioml2016") == "QAM"


def test_radioml_bpsk_maps_to_psk_family() -> None:
    assert _order_to_family("BPSK", "radioml2016") == "PSK"


def test_unknown_order_returns_none() -> None:
    assert _order_to_family("NOT_A_MODE", "radioml2016") is None
    assert _order_to_family("NOT_A_MODE", "cspb") is None
