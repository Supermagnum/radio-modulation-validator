"""Default dataset paths and auto-detection."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATASETS_ROOT = Path("datasets")

RADIOML_DIR_NAME = "radioml"
HISARMOD_DIR_NAME = "hisarmod"
CSPB_DIR_NAME = "cspb"

RADIOML_TAR_NAME = "RML2016.10a.tar.bz2"
RADIOML_PKL_NAME = "RML2016.10a_dict.pkl"
HISARMOD_H5_NAME = "HisarMod2019.1.h5"

TRUTH_FILE_NAMES = (
    "signal_record.txt",
    "signal_record_C_2023.txt",
    "signal_record_first_20000.txt",
    "metadata.txt",
)

_ARCHIVE_SUFFIXES = (".zip", ".gz", ".bz2", ".bin", ".7z", ".tar")


def radioml_dir(root: Path = DEFAULT_DATASETS_ROOT) -> Path:
    return root / RADIOML_DIR_NAME


def hisarmod_dir(root: Path = DEFAULT_DATASETS_ROOT) -> Path:
    return root / HISARMOD_DIR_NAME


def cspb_dir(root: Path = DEFAULT_DATASETS_ROOT) -> Path:
    return root / CSPB_DIR_NAME


def radioml_tar_path(root: Path = DEFAULT_DATASETS_ROOT) -> Path:
    return radioml_dir(root) / RADIOML_TAR_NAME


def radioml_pkl_path(root: Path = DEFAULT_DATASETS_ROOT) -> Path:
    return radioml_dir(root) / RADIOML_PKL_NAME


def hisarmod_h5_path(root: Path = DEFAULT_DATASETS_ROOT) -> Path:
    return hisarmod_dir(root) / HISARMOD_H5_NAME


def detect_radioml(root: Path = DEFAULT_DATASETS_ROOT) -> Path | None:
    """Return path usable by rmv train --radioml (pkl preferred, else tar)."""
    from rmv.dataset.radioml_resolve import find_radioml_pickle

    pkl = radioml_pkl_path(root)
    if pkl.is_file():
        return pkl
    found = find_radioml_pickle(radioml_dir(root))
    if found is not None:
        return found
    tar = radioml_tar_path(root)
    if tar.is_file():
        return tar
    return None


def detect_hisarmod(root: Path = DEFAULT_DATASETS_ROOT) -> Path | None:
    """Return HISARMOD HDF5 path if present."""
    h5 = hisarmod_h5_path(root)
    if h5.is_file():
        return h5
    candidates = list(hisarmod_dir(root).glob("**/*.h5")) + list(hisarmod_dir(root).glob("**/*.hdf5"))
    return candidates[0] if candidates else None


def find_cspb_truth_file(cdir: Path) -> Path | None:
    """Locate CSPB truth file without scanning every .tim file in the tree."""
    for name in TRUTH_FILE_NAMES:
        candidate = cdir / name
        if candidate.is_file():
            return candidate
    for batch in sorted(cdir.glob("Batch_Dir*")):
        if not batch.is_dir():
            continue
        for name in TRUTH_FILE_NAMES:
            candidate = batch / name
            if candidate.is_file():
                return candidate
    return None


def list_cspb_tim_files(cdir: Path) -> list[Path]:
    """List CSPB signal_*.tim files (Batch_Dir layout, no full-tree rglob)."""
    files: list[Path] = []
    files.extend(sorted(cdir.glob("signal_*.tim")))
    for batch in sorted(cdir.glob("Batch_Dir*")):
        if batch.is_dir():
            files.extend(sorted(batch.glob("signal_*.tim")))
    if files:
        return files
    return sorted(cdir.rglob("signal_*.tim"))


def cspb_has_tim_files(cdir: Path) -> bool:
    """True if at least one CSPB signal_*.tim file exists (Batch_Dir or flat layout)."""
    if any(cdir.glob("signal_*.tim")):
        return True
    for batch in cdir.iterdir():
        if batch.is_dir() and batch.name.startswith("Batch_Dir"):
            if any(batch.glob("signal_*.tim")):
                return True
    return next(cdir.rglob("signal_*.tim"), None) is not None


def cspb_has_archives(cdir: Path) -> bool:
    """True if top-level CSPB archive files exist."""
    for path in cdir.iterdir():
        if not path.is_file():
            continue
        if path.suffix in _ARCHIVE_SUFFIXES or ".tar." in path.name.lower():
            return True
    return False


def detect_cspb_present(root: Path = DEFAULT_DATASETS_ROOT) -> Path | None:
    """
    Return CSPB directory when R2 signals or archives are present.

    Does not require a truth file (labels may still be unavailable for training).
    """
    from rmv.dataset.cspb_detect import CSPBVariant, analyze_cspb_directory

    cdir = cspb_dir(root)
    if not cdir.is_dir():
        return None
    variant = analyze_cspb_directory(cdir)
    if variant == CSPBVariant.ORIGINAL:
        return None
    if variant == CSPBVariant.EMPTY:
        return None
    if cspb_has_tim_files(cdir) or cspb_has_archives(cdir):
        return cdir
    return None


def detect_cspb(root: Path = DEFAULT_DATASETS_ROOT) -> Path | None:
    """Return CSPB directory when truth file and signal/archive content exist (trainable)."""
    cdir = detect_cspb_present(root)
    if cdir is None:
        return None
    if find_cspb_truth_file(cdir) is None:
        return None
    return cdir


def has_original_cspb_only(root: Path = DEFAULT_DATASETS_ROOT) -> bool:
    """True if CSPB files are original 2018 only (RNG flaw), not R2."""
    from rmv.dataset.cspb_detect import has_original_cspb_only as _detect

    return _detect(cspb_dir(root))
