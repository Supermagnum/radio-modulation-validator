"""Paths for rmv scan (database, IQ output, project root)."""

from __future__ import annotations

from pathlib import Path


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


def resolve_scan_directory(path: Path) -> Path:
    """
    Expand ~ and environment variables, then resolve to an absolute path.

    Raises ValueError with a clear message if the path is missing or not a directory.
    """
    expanded = path.expanduser()
    if not expanded.exists():
        msg = f"Path does not exist: {expanded}"
        if str(path) != str(expanded):
            msg += f" (from argument: {path})"
        raise ValueError(msg)
    if not expanded.is_dir():
        raise ValueError(f"Not a directory: {expanded.resolve()}")
    return expanded.resolve()
