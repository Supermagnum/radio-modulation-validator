"""Checksum helpers; authoritative values live in datasets/.manifest.json."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UNVERIFIED = "UNVERIFIED"

# Static fallbacks for release builds; runtime authority is datasets/.manifest.json
RELEASE_CHECKSUMS: dict[str, str] = {
    "radioml/RML2016.10a.tar.bz2": UNVERIFIED,
    "radioml/RML2016.10a_dict.pkl": UNVERIFIED,
    "hisarmod/HisarMod2019.1.h5": UNVERIFIED,
}

DATASET_CHECKSUMS = RELEASE_CHECKSUMS

CSPB_MANIFEST_NAME = ".rmv_cspb_checksums.json"

# opendata.deepsig.ai often does not resolve; .io redirects to HTTPS and works.
RADIOML_DOWNLOAD_URLS: list[str] = [
    # Zenodo mirror (validated re-pack); primary because deepsig.io cert is often expired.
    "https://zenodo.org/api/records/18397070/files/RML2016.10a.tar.bz2/content",
    "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2",
    "http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2",
]
RADIOML_PRIMARY_URL = RADIOML_DOWNLOAD_URLS[0]
RADIOML_MANUAL_URL = "https://www.deepsig.ai/datasets"
RADIOML_ZENODO_RECORD = "https://zenodo.org/records/18397070"

HISARMOD_GITHUB_RELEASES_API = "https://api.github.com/repos/WestdoorSad/IQFormer/releases"

CSPB_PAGE_URLS = [
    "https://cyclostationary.blog/2023/09/25/cspb-ml-2018r2-correcting-an-rng-flaw-in-cspb-ml-2018/",
    "https://cyclostationary.blog/2019/02/15/data-set-for-the-machine-learning-challenge/",
]

# R2 metadata (labels/parameters); not included inside batch zip files
CSPB_R2_TRUTH_URL = (
    "https://cyclostationary.blog/wp-content/uploads/2023/09/signal_record_C_2023.txt"
)
CSPB_R2_TRUTH_FILENAME = "signal_record_C_2023.txt"

CSPB_ARCHIVE_EXTENSIONS = (".tar.gz", ".zip", ".bin", ".bz2", ".tar.bz2", ".7z")

DEFAULT_DOWNLOAD_TIMEOUT_SEC = 300.0


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_verified_checksum(expected: str) -> bool:
    """Return True if checksum is a real SHA-256 (not UNVERIFIED)."""
    if expected == UNVERIFIED:
        return False
    return len(expected) == 64 and all(c in "0123456789abcdef" for c in expected.lower())


def verify_file_checksum(path: Path, rel_key: str, datasets_root: Path | None = None) -> tuple[bool, str]:
    """
    Verify file against manifest or release fallbacks.

    Returns (ok, message).
    """
    if not path.is_file():
        return False, "missing"

    expected: str | None = None
    if datasets_root is not None:
        from rmv.dataset.manifest import get_expected_checksum

        expected = get_expected_checksum(datasets_root, rel_key)
    if expected is None:
        expected = RELEASE_CHECKSUMS.get(rel_key)

    if expected is None:
        return True, "no checksum registered"

    if not is_verified_checksum(expected):
        logger.warning(
            "Checksum for %s is UNVERIFIED; skipping strict verification. "
            "Run: rmv dataset checksum-update --dataset <name>",
            rel_key,
        )
        return True, "unverified (skipped)"

    actual = sha256_file(path)
    if actual.lower() == expected.lower():
        return True, "ok"
    return False, "corrupt"


def load_cspb_batch_checksums(cspb_dir: Path) -> dict[str, str]:
    """Load per-batch CSPB checksums from manifest datasets.cspb.files."""
    from rmv.dataset.manifest import get_dataset_entry

    entry = get_dataset_entry(cspb_dir.parent, "cspb")
    if entry and isinstance(entry.get("files"), dict):
        return {str(k): str(v) for k, v in entry["files"].items()}

    manifest_path = cspb_dir / CSPB_MANIFEST_NAME
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    return {}


def checksum_prefix(sha256: str) -> str:
    """First 16 hex chars for display tables."""
    if not is_verified_checksum(sha256):
        return "UNVERIFIED"
    return sha256[:16]
