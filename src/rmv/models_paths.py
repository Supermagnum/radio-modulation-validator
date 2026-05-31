"""Resolve FP32 vs INT8 ONNX model paths under models/."""

from __future__ import annotations

from pathlib import Path

FAMILY_STEM = "family_classifier"
ORDER_STEM = "order_classifier"
ONNX_INPUT_NAME = "iq_samples"


def resolve_onnx_model(models_dir: Path, stem: str) -> Path:
    """Prefer INT8 ONNX when present, otherwise FP32."""
    int8_path = models_dir / f"{stem}_int8.onnx"
    if int8_path.is_file():
        return int8_path
    return models_dir / f"{stem}.onnx"


def int8_onnx_available(models_dir: Path, stem: str) -> bool:
    return (models_dir / f"{stem}_int8.onnx").is_file()


def int8_output_path(models_dir: Path, stem: str) -> Path:
    return models_dir / f"{stem}_int8.onnx"


def npu_output_path(models_dir: Path, stem: str) -> Path:
    return models_dir / f"{stem}.nb"


def resolve_synthetic_npz(path: Path) -> Path:
    """Return path to synthetic.npz from a file or directory argument."""
    npz = path / "synthetic.npz" if path.is_dir() else path
    if not npz.is_file():
        msg = f"Synthetic dataset not found: {npz}"
        raise FileNotFoundError(msg)
    return npz
