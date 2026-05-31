"""CPU/NPU classifier backend detection for scan and embedded integrations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rmv.export import INPUT_NAME
from rmv.models_paths import FAMILY_STEM, int8_onnx_available, resolve_onnx_model

logger = logging.getLogger(__name__)


def detect_cpu_classifier(models_dir: Path) -> bool:
    """Initialise ONNX Runtime on CPU; prefer INT8 models when present."""
    model_path = resolve_onnx_model(models_dir, FAMILY_STEM)

    if not model_path.is_file():
        return False

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not available for CPU classifier")
        return False

    try:
        sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        dummy = np.zeros((1, 2, 1024), dtype=np.float32)
        inp_name = sess.get_inputs()[0].name if sess.get_inputs() else INPUT_NAME
        sess.run(None, {inp_name: dummy})
        quant = "INT8" if int8_onnx_available(models_dir, FAMILY_STEM) else "FP32"
        logger.info(
            "CPU classifier initialised: %s (%s)",
            model_path.name,
            quant,
        )
        return True
    except Exception as exc:
        logger.warning("CPU classifier failed: %s", exc)
        return False
