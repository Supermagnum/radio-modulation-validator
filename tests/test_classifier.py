"""Tests for ONNX classifier with mocked runtime."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rmv.classifier import ModulationClassifier, aggregate_results
from rmv.constants import FAMILY_CLASSES, ORDER_CLASSES
from rmv.types import ClassifierResult


@pytest.fixture
def mock_ort(mocker: pytest.MockFixture) -> MagicMock:
    session = MagicMock()

    def run(_outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = feed[list(feed.keys())[0]].shape[0]
        n_cls = 6
        logits = np.zeros((batch, n_cls), dtype=np.float32)
        logits[:, 2] = 5.0
        return [logits]

    session.run.side_effect = run
    session.get_inputs.return_value = [MagicMock(name="iq_samples")]
    mocker.patch("rmv.classifier.ort.InferenceSession", return_value=session)
    return session


def test_classify_output_fields(tmp_path: Path, mock_ort: MagicMock) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "family_classifier.onnx").write_bytes(b"fake")
    (models_dir / "order_classifier.onnx").write_bytes(b"fake")
    clf = ModulationClassifier(models_dir, verify_checksums=False)
    samples = np.random.randn(2, 2, 1024).astype(np.float32)
    results = clf.classify(samples)
    assert len(results) == 2
    r = results[0]
    assert isinstance(r, ClassifierResult)
    assert r.family in FAMILY_CLASSES or isinstance(r.family, str)
    assert 0.0 <= r.family_confidence <= 1.0
    assert isinstance(r.order, str)
    assert r.family_logits.shape[0] >= 1


def test_aggregate_majority_vote() -> None:
    results = [
        ClassifierResult("FM", 0.9, "NBFM", 0.8, np.zeros(6), np.zeros(10)),
        ClassifierResult("FM", 0.85, "WBFM", 0.75, np.zeros(6), np.zeros(10)),
        ClassifierResult("PSK", 0.6, "QPSK", 0.7, np.zeros(6), np.zeros(10)),
    ]
    agg = aggregate_results(results, threshold=0.70)
    assert agg.family == "FM"
    assert agg.order in ("NBFM", "WBFM")
