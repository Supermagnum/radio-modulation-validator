"""Export PyTorch checkpoints to ONNX and update checksums."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from rmv.checksum_util import update_checksums_for_dir

logger = logging.getLogger(__name__)

OPSET_VERSION = 17
INPUT_NAME = "iq_samples"
OUTPUT_NAME = "logits"


def _load_checkpoint(checkpoint: Path) -> tuple[object, list[str], str]:
    import torch

    from rmv.model import ResidualCNN

    if not checkpoint.is_file():
        msg = f"Checkpoint not found: {checkpoint}"
        raise FileNotFoundError(msg)
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names: list[str] = data["class_names"]
    mode: str = data.get("mode", "family")
    model = ResidualCNN(len(class_names))
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, class_names, mode


def export_to_onnx(
    checkpoint: Path,
    output_path: Path,
    *,
    verify: bool = True,
) -> Path:
    """Export checkpoint to ONNX with dynamic batch axis."""
    import torch

    model, class_names, mode = _load_checkpoint(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 2, 1024, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            dynamo=False,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "onnxscript":
            msg = (
                "ONNX export requires onnxscript with this PyTorch version, or use "
                "dynamo=False (install train extras: uv sync --extra train)"
            )
            raise ImportError(msg) from exc
        raise

    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"class_names": class_names, "mode": mode}, indent=2), encoding="utf-8")

    if verify:
        _verify_onnx(output_path)

    logger.info("Exported ONNX model: %s", output_path)
    return output_path


def _verify_onnx(onnx_path: Path) -> None:
    """Run onnxruntime inference smoke test."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    batch = np.random.randn(2, 2, 1024).astype(np.float32)
    outputs = session.run(None, {inp.name: batch})
    if not outputs or outputs[0].shape[0] != 2:
        msg = f"ONNX verification failed for {onnx_path}"
        raise RuntimeError(msg)


def export_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    checksums_path: Path | None = None,
) -> list[Path]:
    """
    Export checkpoint to appropriate ONNX filename and update checksums.

    family -> family_classifier.onnx
    order -> order_classifier.onnx
    """
    _, _, mode = _load_checkpoint(checkpoint)
    if mode == "family":
        out_name = "family_classifier.onnx"
    else:
        out_name = "order_classifier.onnx"
    out_path = output_dir / out_name
    export_to_onnx(checkpoint, out_path)
    if checksums_path is not None:
        update_checksums_for_dir(output_dir, checksums_path)
    return [out_path]


def export_both_from_dir(
    checkpoints_dir: Path,
    output_dir: Path,
    checksums_path: Path,
) -> list[Path]:
    """Export family and order best checkpoints from a directory."""
    exported: list[Path] = []
    family_ckpt = checkpoints_dir / "best_family_classifier.pt"
    order_ckpt = checkpoints_dir / "best_order_classifier.pt"
    if family_ckpt.is_file():
        exported.extend(export_checkpoint(family_ckpt, output_dir, checksums_path=checksums_path))
    if order_ckpt.is_file():
        update_checksums_for_dir(output_dir, checksums_path)
        exported.extend(export_checkpoint(order_ckpt, output_dir, checksums_path=checksums_path))
    if not exported:
        msg = f"No checkpoints found in {checkpoints_dir}"
        raise FileNotFoundError(msg)
    update_checksums_for_dir(output_dir, checksums_path)
    return exported
