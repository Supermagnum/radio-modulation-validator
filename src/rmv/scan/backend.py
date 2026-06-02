"""CPU/NPU classifier backend detection for scan and embedded integrations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from rmv.export import INPUT_NAME
from rmv.models_paths import (
    ORDER_STEM,
    int8_onnx_available,
    resolve_family_onnx_model,
    resolve_order_onnx_model,
)

logger = logging.getLogger(__name__)


def _probe_session(model_path: Path) -> bool:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    dummy = np.zeros((1, 2, 1024), dtype=np.float32)
    inp_name = sess.get_inputs()[0].name if sess.get_inputs() else INPUT_NAME
    sess.run(None, {inp_name: dummy})
    return True


def detect_cpu_classifier(models_dir: Path) -> bool:
    """
    Initialise ONNX Runtime on CPU for family and order classifiers.

    Family: prefer INT8 when present and verified at export time.
    Order: always FP32 (INT8 order quantisation is not deployed).
    """
    family_path = resolve_family_onnx_model(models_dir)
    order_path = resolve_order_onnx_model(models_dir)

    if not family_path.is_file() or not order_path.is_file():
        return False

    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        logger.warning("onnxruntime not available for CPU classifier")
        return False

    try:
        _probe_session(family_path)
        _probe_session(order_path)
        family_quant = "INT8" if int8_onnx_available(models_dir, "family_classifier") else "FP32"
        order_quant = "FP32"
        if int8_onnx_available(models_dir, ORDER_STEM):
            logger.warning(
                "Ignoring stale %s_int8.onnx; order inference uses FP32 only.",
                ORDER_STEM,
            )
        logger.info(
            "CPU classifier initialised: family=%s (%s), order=%s (%s)",
            family_path.name,
            family_quant,
            order_path.name,
            order_quant,
        )
        return True
    except Exception as exc:
        logger.warning("CPU classifier failed: %s", exc)
        return False
