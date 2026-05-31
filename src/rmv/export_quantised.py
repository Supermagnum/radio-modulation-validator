"""
Quantise float32 ONNX models to INT8 for NPU deployment.

Uses ONNX Runtime static quantisation with calibration data from the synthetic
dataset. Output is standard INT8 ONNX (CPU via onnxruntime) or input to SpacemiT
NPU conversion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from rmv.checksum_util import update_checksums_for_dir
from rmv.export import INPUT_NAME
from rmv.models_paths import FAMILY_STEM, ORDER_STEM, int8_output_path, resolve_synthetic_npz

logger = logging.getLogger(__name__)

CALIBRATION_CHUNKS = 512


def load_calibration_data(
    synthetic_path: Path,
    n_chunks: int = CALIBRATION_CHUNKS,
    snr_db_min: float = 0.0,
    *,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Load calibration data from synthetic.npz.
    Returns shape (N, 2, 1024) float32.

    High-SNR samples are preferred for calibration.
    """
    npz_path = resolve_synthetic_npz(synthetic_path)
    data = np.load(npz_path, allow_pickle=True)
    samples = data["samples"].astype(np.float32)
    snr_db = data["snr_db"].astype(np.float32)

    mask = snr_db >= snr_db_min
    high_snr = samples[mask]

    if len(high_snr) == 0:
        logger.warning(
            "No samples with snr_db >= %s in %s; using full dataset for calibration.",
            snr_db_min,
            npz_path,
        )
        high_snr = samples

    if len(high_snr) < n_chunks:
        logger.warning(
            "Only %d high-SNR samples available, requested %d. Using all available.",
            len(high_snr),
            n_chunks,
        )
        return high_snr.astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(high_snr), n_chunks, replace=False)
    return high_snr[idx].astype(np.float32)


def quantise_model(
    onnx_path: Path,
    output_path: Path,
    calibration_data: np.ndarray,
    input_name: str = INPUT_NAME,
) -> None:
    """Quantise a float32 ONNX model to INT8 using static calibration."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class IQCalibrationReader(CalibrationDataReader):
        def __init__(self, data: np.ndarray, name: str) -> None:
            self.data = data
            self.input_name = name
            self.idx = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.idx >= len(self.data):
                return None
            batch = self.data[self.idx : self.idx + 1]
            self.idx += 1
            return {self.input_name: batch}

        def rewind(self) -> None:
            self.idx = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = IQCalibrationReader(calibration_data, input_name)

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    logger.info("Quantised model written to %s", output_path)
    if onnx_path.is_file() and output_path.is_file():
        logger.info(
            "Size: %.2f MB -> %.2f MB",
            onnx_path.stat().st_size / 1e6,
            output_path.stat().st_size / 1e6,
        )


def verify_quantised_accuracy(
    fp32_path: Path,
    int8_path: Path,
    calibration_data: np.ndarray,
    class_names: list[str],
    tolerance_pct: float = 3.0,
) -> bool:
    """
    Compare top-1 predictions between FP32 and INT8 on calibration data.
    Returns True if disagreement rate is within tolerance_pct.
    """
    import onnxruntime as ort

    _ = class_names  # reserved for future per-class reporting

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inp_name = sess_fp32.get_inputs()[0].name

    batch_size = 64
    agree = 0
    total = 0

    for i in range(0, len(calibration_data), batch_size):
        batch = calibration_data[i : i + batch_size]
        inp = {inp_name: batch}

        logits_fp32 = sess_fp32.run(None, inp)[0]
        logits_int8 = sess_int8.run(None, inp)[0]

        pred_fp32 = logits_fp32.argmax(axis=1)
        pred_int8 = logits_int8.argmax(axis=1)

        agree += int((pred_fp32 == pred_int8).sum())
        total += len(batch)

    agreement_pct = 100.0 * agree / total
    drop_pct = 100.0 - agreement_pct

    logger.info(
        "FP32 vs INT8 top-1 agreement: %.2f%% (drop: %.2f%%, tolerance: %.1f%%)",
        agreement_pct,
        drop_pct,
        tolerance_pct,
    )

    if drop_pct > tolerance_pct:
        logger.error(
            "Quantisation accuracy drop %.2f%% exceeds tolerance %.1f%%. "
            "Do not deploy this model.",
            drop_pct,
            tolerance_pct,
        )
        return False

    logger.info("Quantisation accuracy within tolerance — safe to deploy.")
    return True


def _load_class_names(meta_path: Path) -> list[str]:
    if not meta_path.is_file():
        return []
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    names = data.get("class_names")
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


def run_export_quantised(
    synthetic_path: Path,
    models_dir: Path,
    *,
    calibration_chunks: int = CALIBRATION_CHUNKS,
    snr_min: float = 0.0,
    tolerance_pct: float = 3.0,
    skip_verify: bool = False,
    npu: bool = False,
    checksums_path: Path | None = None,
    seed: int | None = 42,
) -> list[Path]:
    """Quantise family and order classifiers; optionally run NPU conversion."""
    calibration = load_calibration_data(
        synthetic_path,
        n_chunks=calibration_chunks,
        snr_db_min=snr_min,
        seed=seed,
    )

    written: list[Path] = []
    pairs = [
        (FAMILY_STEM, "family_classifier.meta.json"),
        (ORDER_STEM, "order_classifier.meta.json"),
    ]

    for stem, meta_name in pairs:
        fp32_path = models_dir / f"{stem}.onnx"
        if not fp32_path.is_file():
            msg = f"FP32 model not found: {fp32_path}"
            raise FileNotFoundError(msg)

        int8_path = int8_output_path(models_dir, stem)
        quantise_model(fp32_path, int8_path, calibration)

        if not skip_verify:
            class_names = _load_class_names(models_dir / meta_name)
            ok = verify_quantised_accuracy(
                fp32_path,
                int8_path,
                calibration,
                class_names,
                tolerance_pct=tolerance_pct,
            )
            if not ok:
                msg = f"INT8 verification failed for {stem}"
                raise RuntimeError(msg)

        written.append(int8_path)

    if npu:
        from rmv.export_npu import run_export_npu

        npz = resolve_synthetic_npz(synthetic_path)
        written.extend(
            run_export_npu(
                models_dir,
                output_dir=models_dir,
                calibration_data_path=npz,
            )
        )

    if checksums_path is not None:
        update_checksums_for_dir(models_dir, checksums_path)

    return written
