"""datasets/.manifest.json - authoritative checksums and download metadata."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rmv.dataset.checksums import RELEASE_CHECKSUMS, UNVERIFIED, is_verified_checksum, sha256_file

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".manifest.json"
SCHEMA_VERSION = "1.0"


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_FILENAME


def _empty_manifest() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "datasets": {}}


def load_manifest(root: Path) -> dict[str, Any]:
    """Load datasets/.manifest.json or return empty structure."""
    path = manifest_path(root)
    if not path.is_file():
        return _empty_manifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _empty_manifest()
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("datasets", {})
    return data


def save_manifest(root: Path, data: dict[str, Any]) -> None:
    """Write datasets/.manifest.json."""
    root.mkdir(parents=True, exist_ok=True)
    data["schema_version"] = SCHEMA_VERSION
    path = manifest_path(root)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote manifest: %s", path)


def get_dataset_entry(root: Path, name: str) -> dict[str, Any] | None:
    """Return manifest entry for radioml, hisarmod, or cspb."""
    manifest = load_manifest(root)
    entry = manifest.get("datasets", {}).get(name)
    return entry if isinstance(entry, dict) else None


def get_expected_checksum(root: Path, rel_key: str) -> str | None:
    """
    Resolve expected SHA-256: manifest first, then RELEASE_CHECKSUMS.

    rel_key examples: radioml/RML2016.10a_dict.pkl, hisarmod/HisarMod2019.1.h5
    """
    part, _, filename = rel_key.partition("/")
    entry = get_dataset_entry(root, part)
    if entry:
        files = entry.get("files")
        if isinstance(files, dict) and filename in files:
            val = str(files[filename])
            if val and val != UNVERIFIED:
                return val
        if filename in (entry.get("primary_file", ""),) and entry.get("sha256"):
            val = str(entry["sha256"])
            if is_verified_checksum(val):
                return val
    fallback = RELEASE_CHECKSUMS.get(rel_key)
    if fallback and is_verified_checksum(fallback):
        return fallback
    return None


def checksum_prefix_from_manifest(root: Path, dataset_name: str) -> str:
    """Return checksum prefix for status display without re-hashing."""
    entry = get_dataset_entry(root, dataset_name)
    if not entry:
        return "UNVERIFIED"
    sha = entry.get("sha256")
    if isinstance(sha, str) and is_verified_checksum(sha):
        return sha[:16]
    files = entry.get("files")
    if isinstance(files, dict) and files:
        first = next(iter(files.values()))
        if is_verified_checksum(str(first)):
            return str(first)[:16]
    status = entry.get("status", "")
    if status == "verified" and entry.get("sha256"):
        s = str(entry["sha256"])
        return s[:16] if len(s) >= 16 else s
    return "UNVERIFIED"


def update_radioml_manifest(
    root: Path,
    *,
    tar_sha256: str | None = None,
    pkl_sha256: str | None = None,
    status: str = "verified",
) -> None:
    """Record RadioML download metadata in manifest."""
    manifest = load_manifest(root)
    files: dict[str, str] = {}
    if tar_sha256:
        files["RML2016.10a.tar.bz2"] = tar_sha256
    if pkl_sha256:
        files["RML2016.10a_dict.pkl"] = pkl_sha256
    manifest["datasets"]["radioml"] = {
        "version": "2016.10a",
        "downloaded_at": _now_iso(),
        "sha256": pkl_sha256 or tar_sha256 or UNVERIFIED,
        "primary_file": "RML2016.10a_dict.pkl",
        "files": files,
        "status": status,
    }
    save_manifest(root, manifest)


def update_hisarmod_manifest(root: Path, sha256: str, *, status: str = "verified") -> None:
    manifest = load_manifest(root)
    manifest["datasets"]["hisarmod"] = {
        "version": "2019.1",
        "downloaded_at": _now_iso(),
        "sha256": sha256,
        "primary_file": "HisarMod2019.1.h5",
        "status": status,
    }
    save_manifest(root, manifest)


def update_cspb_manifest(
    root: Path,
    *,
    batch_files: int,
    file_checksums: dict[str, str] | None = None,
    status: str = "verified",
    version: str = "R2",
) -> None:
    manifest = load_manifest(root)
    entry: dict[str, Any] = {
        "version": version,
        "downloaded_at": _now_iso(),
        "batch_files": batch_files,
        "status": status,
    }
    if file_checksums:
        entry["files"] = file_checksums
    manifest["datasets"]["cspb"] = entry
    save_manifest(root, manifest)


def refresh_manifest_checksums(root: Path, dataset: str) -> None:
    """Recompute SHA-256 from disk and update manifest (maintainer command)."""
    from rmv.dataset.paths import (
        cspb_dir,
        radioml_pkl_path,
        radioml_tar_path,
        hisarmod_h5_path,
        detect_hisarmod,
    )

    if dataset == "radioml":
        tar = radioml_tar_path(root)
        pkl = radioml_pkl_path(root)
        tar_sha = sha256_file(tar) if tar.is_file() else None
        pkl_sha = sha256_file(pkl) if pkl.is_file() else None
        if not tar_sha and not pkl_sha:
            msg = "No RadioML files found to checksum"
            raise ValueError(msg)
        update_radioml_manifest(
            root,
            tar_sha256=tar_sha,
            pkl_sha256=pkl_sha,
            status="verified",
        )
    elif dataset == "hisarmod":
        path = detect_hisarmod(root) or hisarmod_h5_path(root)
        if not path.is_file():
            msg = f"HISARMOD file not found: {path}"
            raise ValueError(msg)
        update_hisarmod_manifest(root, sha256_file(path), status="verified")
    elif dataset == "cspb":
        cdir = cspb_dir(root)
        if not cdir.is_dir():
            msg = f"CSPB directory not found: {cdir}"
            raise ValueError(msg)
        entries: dict[str, str] = {}
        for fpath in sorted(cdir.rglob("*")):
            if fpath.is_file() and fpath.name not in (MANIFEST_FILENAME, ".rmv_cspb_checksums.json"):
                if fpath.suffix in (".tim", ".zip", ".gz", ".bz2", ".bin", ".txt", ".7z"):
                    entries[fpath.name] = sha256_file(fpath)
        update_cspb_manifest(
            root,
            batch_files=len([n for n in entries if n.endswith(".tim") or "batch" in n.lower()]),
            file_checksums=entries,
            status="verified",
        )
    else:
        msg = f"Unknown dataset: {dataset}"
        raise ValueError(msg)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
