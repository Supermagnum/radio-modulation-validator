"""Strict CSPB.ML.2018 vs CSPB.ML.2018R2 detection."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

# R2 correction post and typical WordPress asset names
CSPB_R2_URL_PATTERNS = (
    re.compile(r"2018\s*r2", re.I),
    re.compile(r"2018r2", re.I),
    re.compile(r"cspb\.ml\.2018r2", re.I),
    re.compile(r"cspb-ml-2018r2", re.I),
    re.compile(r"cspb_ml_2018r2", re.I),
)

# Original 2018 (no R2) - must not be treated as R2
CSPB_ORIGINAL_URL_PATTERNS = (
    re.compile(r"cspb\.ml\.2018(?!r2)", re.I),
    re.compile(r"dataset.*2018(?!.*r2)", re.I),
    re.compile(r"machine-learning-challenge", re.I),
)


class CSPBVariant(Enum):
    R2 = "r2"
    ORIGINAL = "original"
    UNKNOWN = "unknown"
    EMPTY = "empty"


def classify_cspb_filename(name: str) -> CSPBVariant:
    """
    Classify a CSPB-related filename or URL fragment.

    R2 requires explicit R2 markers; original 2018 without R2 is rejected for training.
    """
    lower = name.lower().replace("_", "-").replace(" ", "")
    if not lower:
        return CSPBVariant.EMPTY

    has_r2 = (
        "2018r2" in lower
        or "cspb.ml.2018r2" in lower
        or "cspb-ml-2018r2" in lower
        or bool(re.search(r"cspb.*2018.*r2", lower))
        or (lower.endswith("r2.zip") or lower.endswith("r2.tar.gz"))
    )
    if has_r2:
        return CSPBVariant.R2

    is_original = (
        ("cspb" in lower and "2018" in lower and "r2" not in lower)
        or ("cspb.ml.2018" in lower and "r2" not in lower)
        or (lower.startswith("batch") and "r2" not in lower and "2018" in lower)
    )
    if is_original:
        return CSPBVariant.ORIGINAL

    if "cspb" in lower:
        return CSPBVariant.UNKNOWN
    return CSPBVariant.UNKNOWN


def is_cspb_r2_download_link(url: str, link_text: str = "") -> bool:
    """Return True only if URL/text indicates CSPB.ML.2018R2, not original 2018."""
    combined = f"{url} {link_text}".lower()
    if classify_cspb_filename(combined) == CSPBVariant.ORIGINAL:
        return False
    if classify_cspb_filename(combined) == CSPBVariant.R2:
        return True
    for pat in CSPB_R2_URL_PATTERNS:
        if pat.search(combined):
            if not any(p.search(combined) for p in CSPB_ORIGINAL_URL_PATTERNS if "r2" not in combined):
                return True
    return False


def analyze_cspb_directory(cdir: Path) -> CSPBVariant:
    """
    Determine whether datasets/cspb/ contains R2, original-only, or unknown files.
    """
    if not cdir.is_dir():
        return CSPBVariant.EMPTY

    names = [p.name for p in cdir.iterdir() if p.is_file() or p.is_dir()]
    if not names:
        return CSPBVariant.EMPTY

    variants = {classify_cspb_filename(n) for n in names}
    if CSPBVariant.R2 in variants:
        return CSPBVariant.R2
    if any(n.startswith("Batch_Dir") for n in names):
        return CSPBVariant.R2
    if CSPBVariant.ORIGINAL in variants and CSPBVariant.R2 not in variants:
        return CSPBVariant.ORIGINAL
    if CSPBVariant.UNKNOWN in variants:
        return CSPBVariant.UNKNOWN
    return CSPBVariant.UNKNOWN


def has_original_cspb_only(cdir: Path) -> bool:
    """True when directory has original 2018 markers and no confirmed R2 markers."""
    return analyze_cspb_directory(cdir) == CSPBVariant.ORIGINAL


def has_confirmed_cspb_r2(cdir: Path) -> bool:
    """True when directory has confirmed R2 batch files or manifest says R2."""
    return analyze_cspb_directory(cdir) == CSPBVariant.R2
