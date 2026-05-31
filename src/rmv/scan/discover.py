"""Discover GNU Radio OOT projects under a directory tree."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rmv.scan.exclusions import FRAMEWORK_DIR_NAMES, merged_exclude_projects

SKIP_DIR_NAMES = FRAMEWORK_DIR_NAMES | frozenset({
    ".git",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
})

GR3_CMAKE_MARKERS = (
    "GnuradioConfig",
    "gnuradio-runtime",
    "find_package(gnuradio",
    "find_package(Gnuradio",
    "gr_modtool",
    "gr_register",
)

GR4_CMAKE_MARKERS = (
    "gnuradio4",
    "find_package(gnuradio4",
    "Gnuradio4",
)


@dataclass
class GRProject:
    path: Path
    name: str
    gr_version: str  # "3" | "4" | "both" | "unknown"
    readme_path: Path | None
    cmake_path: Path | None
    grc_blocks: list[Path] = field(default_factory=list)
    gr4_headers: list[Path] = field(default_factory=list)
    python_modules: list[Path] = field(default_factory=list)


def _is_oot_candidate(directory: Path) -> bool:
    if directory.name in FRAMEWORK_DIR_NAMES:
        return False
    cmake = directory / "CMakeLists.txt"
    if cmake.is_file():
        text = cmake.read_text(encoding="utf-8", errors="ignore").lower()
        if "find_package(gnuradio" in text or "find_package(gnuradio4" in text:
            return True
        if "gr_modtool" in text or "gr_register" in text:
            return True
    grc = directory / "grc"
    if grc.is_dir() and list(grc.glob("*.block.yml")):
        return True
    if (directory / "include" / "gnuradio").is_dir():
        return True
    if (directory / "include" / "gnuradio-4.0").is_dir():
        return True
    return False


def _detect_gr_version(project_dir: Path, cmake_text: str, readme_text: str) -> str:
    gr3 = False
    gr4 = False

    lower_cmake = cmake_text.lower()
    if any(m.lower() in lower_cmake for m in GR3_CMAKE_MARKERS):
        gr3 = True
    if any(m.lower() in lower_cmake for m in GR4_CMAKE_MARKERS):
        gr4 = True

    if list((project_dir / "grc").glob("*.block.yml")):
        gr3 = True
    inc_gr = project_dir / "include" / "gnuradio"
    if inc_gr.is_dir() and not (project_dir / "include" / "gnuradio-4.0").is_dir():
        # gr3-style include/gnuradio/<name>/
        for child in inc_gr.iterdir():
            if child.is_dir():
                gr3 = True
                break
    if (project_dir / "include" / "gnuradio-4.0").is_dir():
        gr4 = True

    python_dir = project_dir / "python"
    if python_dir.is_dir():
        for bindings in python_dir.glob("*/bindings"):
            if bindings.is_dir():
                gr3 = True

    readme_lower = readme_text.lower()
    if re.search(r"gnu\s*radio\s*4|gr\s*4|gnuradio4", readme_lower):
        gr4 = True
    if re.search(r"gnu\s*radio\s*3\.?10|gr\s*3", readme_lower):
        gr3 = True

    git_head = project_dir / ".git" / "HEAD"
    if git_head.is_file():
        head = git_head.read_text(encoding="utf-8", errors="ignore")
        if "gnuradio4" in head.lower():
            gr4 = True

    if gr3 and gr4:
        return "both"
    if gr3:
        return "3"
    if gr4:
        return "4"
    return "unknown"


def _apply_project_filters(
    projects: list[GRProject],
    *,
    include_names: frozenset[str] | None,
    exclude_names: frozenset[str] | None,
) -> list[GRProject]:
    """Keep only included names (if set); drop excluded names."""
    out: list[GRProject] = []
    for project in projects:
        if project.name in FRAMEWORK_DIR_NAMES:
            continue
        if exclude_names and project.name in exclude_names:
            continue
        if include_names is not None and project.name not in include_names:
            continue
        out.append(project)
    return out


def discover_gr_projects(
    root: Path,
    *,
    include_names: frozenset[str] | None = None,
    exclude_names: frozenset[str] | None = None,
) -> list[GRProject]:
    """
    Scan root recursively for GNU Radio OOT project directories.

    Does not descend into framework trees (gnuradio, gnuradio4, gnuradio3).
    """
    root = root.resolve()
    if not root.is_dir():
        return []

    found: list[GRProject] = []
    seen_paths: set[Path] = set()

    def walk(directory: Path) -> None:
        if not directory.is_dir():
            return
        resolved = directory.resolve()
        if resolved in seen_paths:
            return
        if _is_oot_candidate(resolved):
            seen_paths.add(resolved)
            _append_project(resolved, found)
            return
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name in SKIP_DIR_NAMES:
                continue
            walk(child)

    walk(root)
    effective_exclude = merged_exclude_projects(exclude_names or ())
    filtered = _apply_project_filters(
        found,
        include_names=include_names,
        exclude_names=effective_exclude,
    )
    return sorted(filtered, key=lambda p: p.path.as_posix())


def _append_project(project_dir: Path, found: list[GRProject]) -> None:
    cmake_path = project_dir / "CMakeLists.txt"
    cmake_text = ""
    if cmake_path.is_file():
        cmake_text = cmake_path.read_text(encoding="utf-8", errors="ignore")

    readme_path = project_dir / "README.md"
    if not readme_path.is_file():
        alt = list(project_dir.glob("README*"))
        readme_path = alt[0] if alt else None
    readme_text = ""
    if readme_path is not None and readme_path.is_file():
        readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")

    grc_blocks = sorted((project_dir / "grc").glob("*.block.yml")) if (project_dir / "grc").is_dir() else []
    gr4_headers = sorted((project_dir / "include" / "gnuradio-4.0").glob("**/*.hpp"))
    python_modules = sorted((project_dir / "python").glob("*/"))

    found.append(
        GRProject(
            path=project_dir,
            name=project_dir.name,
            gr_version=_detect_gr_version(project_dir, cmake_text, readme_text),
            readme_path=readme_path if readme_path and readme_path.is_file() else None,
            cmake_path=cmake_path if cmake_path.is_file() else None,
            grc_blocks=grc_blocks,
            gr4_headers=gr4_headers,
            python_modules=python_modules,
        )
    )
