"""Locate RML2016.10a_dict.pkl after extraction (strict filename)."""

from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path

from rmv.dataset.paths import RADIOML_PKL_NAME

logger = logging.getLogger(__name__)

# Zenodo mirror ships optimized pickle name; data is bit-identical to 2016.10a_dict.pkl.
RADIOML_PKL_ALIASES: tuple[str, ...] = (
    RADIOML_PKL_NAME,
    "RML2016.10a_dict_optimized.pkl",
)


class RadioMLPickleNotFoundError(FileNotFoundError):
    """Raised when the canonical RadioML pickle is missing."""


def find_radioml_pickle(search_dir: Path) -> Path | None:
    """Find canonical or known alias pickle under search_dir."""
    for name in RADIOML_PKL_ALIASES:
        direct = search_dir / name
        if direct.is_file():
            return direct
        for candidate in search_dir.rglob(name):
            if candidate.is_file():
                return candidate
    return None


def ensure_canonical_pickle(search_dir: Path) -> Path:
    """Ensure RML2016.10a_dict.pkl exists (copy from Zenodo optimized name if needed)."""
    canonical = search_dir / RADIOML_PKL_NAME
    if canonical.is_file():
        return canonical
    found = find_radioml_pickle(search_dir)
    if found is None:
        msg = f"No RadioML pickle found under {search_dir}"
        raise RadioMLPickleNotFoundError(msg)
    if found.resolve() != canonical.resolve():
        logger.info("Installing canonical pickle from %s", found.name)
        shutil.copy2(found, canonical)
    return canonical


def extract_radioml_tar_strict(tar_path: Path, dest_dir: Path) -> Path:
    """
    Extract RadioML tar.bz2 and return path to RML2016.10a_dict.pkl.

    Does not fall back to arbitrary .pkl files (archive layouts vary).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting RadioML archive: %s", tar_path)
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(dest_dir, filter="data")

    found = find_radioml_pickle(dest_dir)
    if found is not None:
        return ensure_canonical_pickle(dest_dir)

    other_pkls = sorted(dest_dir.rglob("*.pkl"))
    names = [p.name for p in other_pkls]
    msg = (
        f"Expected {RADIOML_PKL_NAME} after extracting {tar_path}, not found. "
        f"Other .pkl files present: {names or 'none'}. "
        "Verify you have RadioML 2016.10A (RML2016.10a.tar.bz2)."
    )
    raise RadioMLPickleNotFoundError(msg)
