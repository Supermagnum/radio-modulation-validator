"""SHA-256 checksum management for ONNX model files."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksums_file(path: Path) -> dict[str, str]:
    """Parse checksums.sha256 into {filename: hex_digest}."""
    if not path.is_file():
        msg = f"Checksum file not found: {path}"
        raise FileNotFoundError(msg)
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            digest, name = parts[0], parts[-1]
            if len(digest) == 64:
                result[Path(name).name] = digest.lower()
    return result


def write_checksums_file(path: Path, entries: dict[str, str]) -> None:
    """Write checksum entries sorted by filename."""
    lines = [
        "# SHA-256 checksums for models/*.onnx (FP32 and INT8).",
        "# Run: rmv checksum update",
        "# Verify: rmv checksum verify",
        "",
    ]
    for name in sorted(entries):
        lines.append(f"{entries[name]}  {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Updated checksum file: %s", path)


def verify_model_checksum(model_path: Path, checksums_path: Path) -> None:
    """Verify a single model file against checksums.sha256."""
    if not model_path.is_file():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)
    expected_all = parse_checksums_file(checksums_path)
    name = model_path.name
    if name not in expected_all:
        msg = f"No checksum entry for {name} in {checksums_path}"
        raise ValueError(msg)
    actual = sha256_file(model_path)
    expected = expected_all[name]
    if actual != expected:
        msg = (
            f"Checksum mismatch for {name}: expected {expected}, got {actual}. "
            "Re-download models or run rmv checksum update after export."
        )
        raise ValueError(msg)


def update_checksums_for_dir(models_dir: Path, checksums_path: Path) -> int:
    """Recompute checksums for every models/*.onnx and update checksums.sha256."""
    entries: dict[str, str] = {}
    if not models_dir.is_dir():
        msg = f"Models directory not found: {models_dir}"
        raise FileNotFoundError(msg)
    onnx_files = sorted(models_dir.glob("*.onnx"))
    if not onnx_files:
        msg = f"No .onnx files found in {models_dir}"
        raise FileNotFoundError(msg)
    for onnx in onnx_files:
        entries[onnx.name] = sha256_file(onnx)
    write_checksums_file(checksums_path, entries)
    return len(entries)


def verify_all_models(models_dir: Path, checksums_path: Path) -> list[str]:
    """Verify all ONNX models; return list of verified filenames."""
    expected = parse_checksums_file(checksums_path)
    if not expected:
        msg = f"No checksum entries in {checksums_path}"
        raise ValueError(msg)
    verified: list[str] = []
    for name, digest in expected.items():
        model_path = models_dir / name
        if not model_path.is_file():
            msg = f"Model file missing: {model_path}"
            raise FileNotFoundError(msg)
        actual = sha256_file(model_path)
        if actual != digest:
            msg = f"Checksum mismatch for {name}"
            raise ValueError(msg)
        verified.append(name)
    return verified
