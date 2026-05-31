"""Detect GNU Radio 3/4 installation prefixes and build subprocess environments."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_gr3_prefix(override: Path | None = None) -> Path | None:
    """Find GNU Radio 3.x installation prefix."""
    if override is not None and override.is_dir():
        return override.resolve()
    env = os.environ.get("GNURADIO_PREFIX")
    if env:
        p = Path(env)
        if p.is_dir():
            return p.resolve()
    exe = shutil.which("gnuradio-config-info")
    if exe:
        try:
            result = subprocess.run(
                [exe, "--prefix"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                p = Path(result.stdout.strip())
                if p.is_dir():
                    return p.resolve()
        except (OSError, subprocess.SubprocessError):
            pass
    for candidate in (Path("/usr/local"), Path("/usr"), Path("/opt/gnuradio")):
        if (candidate / "lib" / "python3.dist-packages" / "gnuradio").exists():
            return candidate.resolve()
        if list(candidate.glob("lib/python*/site-packages/gnuradio")):
            return candidate.resolve()
    return None


def find_gr4_prefix(override: Path | None = None) -> Path | None:
    """Find GNU Radio 4.x installation prefix."""
    if override is not None and override.is_dir():
        return override.resolve()
    env = os.environ.get("GNURADIO4_PREFIX")
    if env:
        p = Path(env)
        if p.is_dir():
            return p.resolve()
    for candidate in (
        Path("/opt/gnuradio4-gcc"),
        Path("/opt/gnuradio4"),
        Path("/usr/local"),
    ):
        if (candidate / "include" / "gnuradio-4.0").is_dir():
            return candidate.resolve()
        configs = list(candidate.glob("lib/cmake/gnuradio4*/gnuradio4Config.cmake"))
        if configs:
            return candidate.resolve()
    cmake_prefix = os.environ.get("CMAKE_PREFIX_PATH", "")
    for part in cmake_prefix.split(os.pathsep):
        if not part:
            continue
        p = Path(part)
        if (p / "include" / "gnuradio-4.0").is_dir():
            return p.resolve()
    return None


def _python_subdirs(prefix: Path) -> list[Path]:
    lib = prefix / "lib"
    if not lib.is_dir():
        return []
    return list(lib.glob("python*/site-packages")) + list(lib.glob("python3/dist-packages"))


def build_gr3_env(prefix: Path) -> dict[str, str]:
    """Environment variables to prefer GR3 from prefix."""
    env = os.environ.copy()
    py_paths = [str(p) for p in _python_subdirs(prefix)]
    if py_paths:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing] if existing else []))
    lib_dir = prefix / "lib"
    if lib_dir.is_dir():
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(lib_dir)] + ([existing_ld] if existing_ld else [])
        )
    env["GNURADIO_PREFIX"] = str(prefix)
    return env


def build_gr4_env(prefix: Path) -> dict[str, str]:
    """Environment variables to prefer GR4 from prefix."""
    env = os.environ.copy()
    py_paths = [str(p) for p in _python_subdirs(prefix)]
    if py_paths:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing] if existing else []))
    lib_dir = prefix / "lib"
    if lib_dir.is_dir():
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(lib_dir)] + ([existing_ld] if existing_ld else [])
        )
    existing_cmake = env.get("CMAKE_PREFIX_PATH", "")
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(
        [str(prefix)] + ([existing_cmake] if existing_cmake else [])
    )
    env["GNURADIO4_PREFIX"] = str(prefix)
    return env
