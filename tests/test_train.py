"""Tests for training label mapping and family balance."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from rmv.constants import (
    CSPB_CLASSES,
    CSPB_TO_FAMILY,
    FAMILY_CLASSES,
    HISARMOD_CLASSES,
    HISARMOD_TO_FAMILY,
    RADIOML_CLASSES,
    RADIOML_TO_FAMILY,
    SYNTHETIC_CLASSES,
    SYNTHETIC_TO_FAMILY,
)
from rmv.dataset.synthetic import generate_variant_chunks
from rmv.train import _build_label_arrays, _order_to_family
from rmv.types import IQDataset


def _tiny_dataset(
    class_names: list[str],
    source: str,
    n_per_class: int = 2,
) -> IQDataset:
    chunks = np.random.randn(
        len(class_names) * n_per_class,
        2,
        1024,
    ).astype(np.float32)
    labels = np.array(
        [i for i in range(len(class_names)) for _ in range(n_per_class)],
        dtype=np.int32,
    )
    snr = np.zeros(len(labels), dtype=np.float32)
    return IQDataset(
        samples=chunks,
        labels=labels,
        snr_db=snr,
        class_names=class_names,
        source=source,
    )


def test_family_mapping_no_unknown_defaults() -> None:
    """Every order in standard class lists must map to a family (not None)."""
    datasets = [
        _tiny_dataset(RADIOML_CLASSES, "radioml2016"),
        _tiny_dataset(HISARMOD_CLASSES, "hisarmod"),
        _tiny_dataset(CSPB_CLASSES, "cspb"),
        _tiny_dataset(SYNTHETIC_CLASSES, "synthetic"),
    ]
    for ds in datasets:
        for idx in ds.labels:
            order = ds.class_names[int(idx)]
            fam = _order_to_family(order, ds.source)
            assert fam is not None, f"{order} from {ds.source}"
            assert fam in FAMILY_CLASSES

    samples, labels, snr, _ = _build_label_arrays(datasets, "family")
    assert len(labels) == len(snr) == samples.shape[0]
    assert len(labels) > 0


def test_cspb_am_count_reasonable() -> None:
    """CSPB order-to-family map must not inflate AM (no silent AM default)."""
    by_family = Counter(CSPB_TO_FAMILY.values())
    am_count = by_family.get("AM", 0)
    fsk_count = by_family.get("FSK", 0)
    assert am_count == 0
    assert fsk_count >= 1
    assert am_count < 3 * max(fsk_count, 1)


@pytest.mark.skipif(
    not Path("datasets/cspb").is_dir(),
    reason="CSPB dataset not present",
)
def test_cspb_loaded_samples_have_family_mapping() -> None:
    from rmv.dataset.loader import load_cspb

    ds = load_cspb(Path("datasets/cspb"), cache_dir=Path(".cache"))
    for idx in ds.labels[:500]:
        order = ds.class_names[int(idx)]
        assert _order_to_family(order, ds.source) is not None


def test_radioml_wbfm_skipped_for_family_training() -> None:
    only_wbfm = _tiny_dataset(["WBFM"], "radioml2016", n_per_class=4)
    with pytest.raises(ValueError, match="No training samples"):
        _build_label_arrays([only_wbfm], "family")

    mixed = _tiny_dataset(["WBFM", "AM-DSB"], "radioml2016", n_per_class=4)
    samples, labels, _, _ = _build_label_arrays([mixed], "family")
    assert samples.shape[0] == 4
    assert all(labels[i] == FAMILY_CLASSES.index("AM") for i in range(4))


def test_synthetic_wbfm_bpsk_chunks_shape() -> None:
    wbfm = generate_variant_chunks(
        "WBFM",
        2,
        snr_db=10.0,
        use_gnuradio=False,
        apply_channel=False,
    )
    bpsk = generate_variant_chunks(
        "BPSK",
        2,
        snr_db=10.0,
        use_gnuradio=False,
        apply_channel=False,
    )
    assert wbfm.shape == (2, 2, 1024)
    assert bpsk.shape == (2, 2, 1024)
