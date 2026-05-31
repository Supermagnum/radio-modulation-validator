"""ONNX Runtime inference for family and order classifiers."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

from rmv.checksum_util import verify_model_checksum
from rmv.constants import FAMILY_CLASSES, ORDER_CLASSES
from rmv.models_paths import FAMILY_STEM, ORDER_STEM, resolve_onnx_model
from rmv.types import ClassifierResult

logger = logging.getLogger(__name__)


class ModulationClassifier:
    """Load ONNX models and classify IQ chunks."""

    def __init__(
        self,
        models_dir: Path,
        *,
        checksums_path: Path | None = None,
        confidence_threshold: float = 0.70,
        verify_checksums: bool = True,
    ) -> None:
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        checksums = checksums_path or models_dir.parent / "checksums.sha256"

        family_path = resolve_onnx_model(models_dir, FAMILY_STEM)
        order_path = resolve_onnx_model(models_dir, ORDER_STEM)

        if verify_checksums and checksums.is_file():
            from rmv.checksum_util import parse_checksums_file

            entries = parse_checksums_file(checksums)
            if family_path.is_file() and family_path.name in entries:
                verify_model_checksum(family_path, checksums)
            if order_path.is_file() and order_path.name in entries:
                verify_model_checksum(order_path, checksums)

        if family_path.name.endswith("_int8.onnx"):
            logger.info("Using INT8 family classifier: %s", family_path.name)
        if order_path.name.endswith("_int8.onnx"):
            logger.info("Using INT8 order classifier: %s", order_path.name)

        self.family_names = self._load_class_names(family_path, FAMILY_CLASSES)
        self.order_names = self._load_class_names(order_path, ORDER_CLASSES)

        self._family_session = self._create_session(family_path)
        self._order_session = self._create_session(order_path)

    @staticmethod
    def _load_class_names(model_path: Path, default: list[str]) -> list[str]:
        meta = model_path.with_suffix(".meta.json")
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            names = data.get("class_names")
            if isinstance(names, list):
                return [str(n) for n in names]
        return list(default)

    @staticmethod
    def _create_session(path: Path) -> ort.InferenceSession | None:
        if not path.is_file():
            logger.warning("ONNX model not found (will use mock/fail at inference): %s", path)
            return None
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def _run_session(
        self,
        session: ort.InferenceSession | None,
        samples: np.ndarray,
        class_names: list[str],
    ) -> tuple[str, float, np.ndarray]:
        if session is None:
            logits = np.zeros(len(class_names), dtype=np.float32)
            return class_names[0], 0.0, logits

        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        inp_name = session.get_inputs()[0].name
        logits = session.run(None, {inp_name: arr})[0]
        if logits.ndim == 2:
            row = logits[0]
        else:
            row = logits
        n = min(len(class_names), row.shape[0])
        row = row[:n]
        probs = _softmax(row)
        idx = int(np.argmax(probs))
        return class_names[idx], float(probs[idx]), row.astype(np.float32)

    def classify_chunk(self, sample: np.ndarray) -> ClassifierResult:
        """Classify a single (2, 1024) chunk."""
        family, fam_conf, fam_logits = self._run_session(
            self._family_session, sample, self.family_names
        )
        order, ord_conf, ord_logits = self._run_session(
            self._order_session, sample, self.order_names
        )
        return ClassifierResult(
            family=family,
            family_confidence=fam_conf,
            order=order,
            order_confidence=ord_conf,
            family_logits=fam_logits,
            order_logits=ord_logits,
            pass_threshold=self.confidence_threshold,
        )

    def classify(self, samples: np.ndarray) -> list[ClassifierResult]:
        """
        Classify batch of IQ chunks.

        Args:
            samples: (N, 2, 1024) float32
        """
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 2 or arr.shape[2] != 1024:
            msg = f"Expected shape (N, 2, 1024), got {arr.shape}"
            raise ValueError(msg)
        return [self.classify_chunk(arr[i]) for i in range(arr.shape[0])]

    def classify_aggregate(self, samples: np.ndarray) -> ClassifierResult:
        """Classify all chunks and aggregate by majority vote."""
        results = self.classify(samples)
        if not results:
            msg = "No chunks to classify"
            raise ValueError(msg)
        return aggregate_results(results, self.confidence_threshold)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    exp = np.exp(x)
    return exp / np.sum(exp)


def aggregate_results(
    results: list[ClassifierResult],
    threshold: float = 0.70,
) -> ClassifierResult:
    """Majority vote across chunk results; confidence = mean for winning class."""
    families = [r.family for r in results]
    orders = [r.order for r in results]
    fam_winner = Counter(families).most_common(1)[0][0]
    ord_winner = Counter(orders).most_common(1)[0][0]
    fam_conf = float(np.mean([r.family_confidence for r in results if r.family == fam_winner]))
    ord_conf = float(np.mean([r.order_confidence for r in results if r.order == ord_winner]))
    fam_logits = np.mean([r.family_logits for r in results], axis=0)
    ord_logits = np.mean([r.order_logits for r in results], axis=0)
    return ClassifierResult(
        family=fam_winner,
        family_confidence=fam_conf,
        order=ord_winner,
        order_confidence=ord_conf,
        family_logits=fam_logits.astype(np.float32),
        order_logits=ord_logits.astype(np.float32),
        pass_threshold=threshold,
    )
