"""
Convert INT8 ONNX models to SpacemiT NPU binary format (.nb).

Requires spacemit-npu-convert from the SpacemiT NPU SDK (optional).
CPU inference uses INT8 ONNX directly via onnxruntime.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from rmv.models_paths import FAMILY_STEM, ORDER_STEM, int8_output_path, npu_output_path

logger = logging.getLogger(__name__)

NPU_CONVERT_CMD = "spacemit-npu-convert"
_NPU_STEMS = (FAMILY_STEM, ORDER_STEM)


def find_npu_convert() -> Path | None:
    """Find the spacemit-npu-convert tool."""
    found = shutil.which(NPU_CONVERT_CMD)
    if found:
        return Path(found)

    candidates = [
        Path("/opt/spacemit-npu/bin/spacemit-npu-convert"),
        Path("/usr/local/bin/spacemit-npu-convert"),
        Path.home() / ".local/bin/spacemit-npu-convert",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def convert_to_npu(
    int8_onnx_path: Path,
    output_path: Path,
    *,
    precision: str = "int8",
    calibration_data_path: Path | None = None,
) -> bool:
    """
    Convert an INT8 ONNX model to SpacemiT NPU binary (.nb) format.
    Returns True on success, False if the tool is missing or conversion fails.
    """
    converter = find_npu_convert()
    if converter is None:
        logger.warning(
            "%s not found. NPU conversion skipped. "
            "Install the SpacemiT NPU SDK from https://developer.spacemit.com/",
            NPU_CONVERT_CMD,
        )
        return False

    if not int8_onnx_path.is_file():
        logger.error("INT8 ONNX not found: %s", int8_onnx_path)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(converter),
        str(int8_onnx_path),
        "--precision",
        precision,
        "--output",
        str(output_path),
    ]

    if calibration_data_path is not None:
        cmd += ["--calibration-data", str(calibration_data_path)]

    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error(
            "NPU conversion failed (exit %d):\n%s",
            result.returncode,
            result.stderr,
        )
        return False

    logger.info("NPU model written to %s", output_path)
    if int8_onnx_path.is_file() and output_path.is_file():
        logger.info(
            "Size: %.0f KB -> %.0f KB",
            int8_onnx_path.stat().st_size / 1e3,
            output_path.stat().st_size / 1e3,
        )
    return True


def run_export_npu(
    int8_dir: Path,
    *,
    output_dir: Path | None = None,
    calibration_data_path: Path | None = None,
    precision: str = "int8",
) -> list[Path]:
    """Convert all INT8 ONNX models in int8_dir to .nb files."""
    out_dir = output_dir or int8_dir
    written: list[Path] = []

    for stem in _NPU_STEMS:
        int8_path = int8_output_path(int8_dir, stem)
        if not int8_path.is_file():
            logger.warning("Skipping NPU export; missing %s", int8_path)
            continue
        nb_path = npu_output_path(out_dir, stem)
        if convert_to_npu(
            int8_path,
            nb_path,
            precision=precision,
            calibration_data_path=calibration_data_path,
        ):
            written.append(nb_path)

    return written
