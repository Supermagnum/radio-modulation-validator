"""Resolve FP32 vs INT8 ONNX model paths under models/."""

from __future__ import annotations

from pathlib import Path

FAMILY_STEM = "family_classifier"
ORDER_STEM = "order_classifier"
ONNX_INPUT_NAME = "iq_samples"


def resolve_onnx_model(models_dir: Path, stem: str) -> Path:
    """Resolve ONNX path for a model stem (family: prefer INT8; order: FP32 only)."""
    if stem == ORDER_STEM:
        return resolve_order_onnx_model(models_dir)
    if stem == FAMILY_STEM:
        return resolve_family_onnx_model(models_dir)
    return _resolve_with_int8_preference(models_dir, stem, prefer_int8=True)


def resolve_family_onnx_model(models_dir: Path) -> Path:
    """Prefer verified INT8 family classifier when present, otherwise FP32."""
    return _resolve_with_int8_preference(models_dir, FAMILY_STEM, prefer_int8=True)


def resolve_order_onnx_model(models_dir: Path) -> Path:
    """Always use FP32 order classifier (INT8 order export is not deployed)."""
    return _resolve_with_int8_preference(models_dir, ORDER_STEM, prefer_int8=False)


def _resolve_with_int8_preference(
    models_dir: Path,
    stem: str,
    *,
    prefer_int8: bool,
) -> Path:
    fp32_path = models_dir / f"{stem}.onnx"
    if prefer_int8:
        int8_path = models_dir / f"{stem}_int8.onnx"
        if int8_path.is_file():
            return int8_path
    return fp32_path


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
