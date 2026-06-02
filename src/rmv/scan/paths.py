"""Paths for rmv scan (database, IQ output, project root)."""

from __future__ import annotations

import os
import re
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"(?:^|[/\\])path[/\\]to[/\\]", re.IGNORECASE)


def find_rmv_project_root(start: Path | None = None) -> Path:
    """Walk up from start (or cwd) to find directory containing pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        if (directory / "pyproject.toml").is_file():
            return directory
    return current


def default_db_path(root: Path | None = None) -> Path:
    return find_rmv_project_root(root) / ".rmv_findings.db"


def default_iq_output(root: Path | None = None) -> Path:
    return find_rmv_project_root(root) / ".scan_iq"


def expand_scan_path(path: Path) -> Path:
    """Expand ~ and environment variables in a scan directory argument."""
    return Path(os.path.expandvars(str(path.expanduser())))


def is_documentation_placeholder(path: Path) -> bool:
    """True for README examples like /path/to/github-projects."""
    return bool(_PLACEHOLDER_RE.search(str(path)))


def _directory_has_gr_oot_children(directory: Path) -> bool:
    try:
        for child in directory.iterdir():
            if child.is_dir() and child.name.startswith("gr-"):
                return True
    except OSError:
        return False
    return False


def infer_default_scan_directory(start: Path | None = None) -> Path | None:
    """
    Guess the OOT parent directory when none is passed on the CLI.

    Prefer [scan].root from .rmv_config.toml, then the parent of the rmv repo
    if it contains gr-* projects (typical github-projects layout).
    """
    rmv_root = find_rmv_project_root(start)
    try:
        from rmv.scan.config import load_scan_config

        cfg = load_scan_config(rmv_root)
        if cfg.scan_root:
            candidate = expand_scan_path(Path(cfg.scan_root))
            if candidate.is_dir() and _directory_has_gr_oot_children(candidate):
                return candidate.resolve()
    except Exception:
        pass

    parent = rmv_root.parent
    if parent.is_dir() and _directory_has_gr_oot_children(parent):
        return parent.resolve()
    return None


def _resolve_existing_directory(candidate: Path) -> Path | None:
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved.is_dir():
        return resolved
    return None


def _candidate_scan_paths(path: Path, *, rmv_root: Path) -> list[Path]:
    expanded = expand_scan_path(path)
    cwd = Path.cwd()
    candidates = [
        expanded,
        cwd / expanded,
        rmv_root / expanded,
        rmv_root.parent / expanded,
    ]
    if not expanded.is_absolute():
        candidates.append((cwd / expanded).resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def resolve_scan_directory(
    path: Path | None,
    *,
    start: Path | None = None,
) -> Path:
    """
    Resolve the scan root to an absolute existing directory.

    - None: infer parent github-projects (or [scan].root in config)
    - Documentation placeholders (/path/to/...): use inferred root when available
    - Relative paths: try cwd, rmv repo, and parent of rmv repo
    """
    rmv_root = find_rmv_project_root(start)

    if path is None:
        inferred = infer_default_scan_directory(rmv_root)
        if inferred is None:
            msg = (
                "No scan directory given and none could be inferred. "
                "Pass the parent folder of your gr-* OOT repos, for example:\n"
                f"  rmv scan run {rmv_root.parent}\n"
                "  rmv scan run ..    (when rmv is inside github-projects)"
            )
            raise ValueError(msg)
        return inferred

    expanded = expand_scan_path(path)

    if is_documentation_placeholder(expanded):
        inferred = infer_default_scan_directory(rmv_root)
        if inferred is not None:
            return inferred
        msg = (
            f"'{expanded}' is a documentation placeholder, not a real directory. "
            f"Pass your OOT parent folder, for example: rmv scan run {rmv_root.parent}"
        )
        raise ValueError(msg)

    for candidate in _candidate_scan_paths(path, rmv_root=rmv_root):
        resolved = _resolve_existing_directory(candidate)
        if resolved is not None:
            return resolved

    msg = f"Path does not exist: {expanded}"
    if str(path) != str(expanded):
        msg += f" (from argument: {path})"
    inferred = infer_default_scan_directory(rmv_root)
    if inferred is not None:
        msg += f"\nHint: rmv scan run {inferred}"
    raise ValueError(msg)
