"""Load optional .rmv_config.toml for scan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rmv.scan.paths import find_rmv_project_root


@dataclass
class ScanConfig:
    gr3_prefix: str | None = None
    gr4_prefix: str | None = None
    iq_output: str = ".scan_iq"
    default_yes: bool = False
    exclude_projects: tuple[str, ...] = ()
    include_projects: tuple[str, ...] = ()


def config_path(root: Path | None = None) -> Path:
    return find_rmv_project_root(root) / ".rmv_config.toml"


def load_scan_config(root: Path | None = None) -> ScanConfig:
    path = config_path(root)
    if not path.is_file():
        return ScanConfig()
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section: dict[str, Any] = data.get("scan", {})
    exclude_raw = section.get("exclude_projects", [])
    exclude_projects: tuple[str, ...] = ()
    if isinstance(exclude_raw, list):
        exclude_projects = tuple(str(x) for x in exclude_raw)

    include_raw = section.get("include_projects", [])
    include_projects: tuple[str, ...] = ()
    if isinstance(include_raw, list):
        include_projects = tuple(str(x) for x in include_raw)

    return ScanConfig(
        gr3_prefix=section.get("gr3_prefix"),
        gr4_prefix=section.get("gr4_prefix"),
        iq_output=str(section.get("iq_output", ".scan_iq")),
        default_yes=bool(section.get("default_yes", False)),
        exclude_projects=exclude_projects,
        include_projects=include_projects,
    )


def parse_project_name_list(value: str | None) -> frozenset[str] | None:
    """Parse comma-separated project folder names for --filter."""
    if not value or not value.strip():
        return None
    names = {part.strip() for part in value.split(",") if part.strip()}
    return frozenset(names) if names else None
