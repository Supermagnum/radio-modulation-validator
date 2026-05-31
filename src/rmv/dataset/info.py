"""Inspect downloaded datasets: classes, sample counts, SNR ranges."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from rich.console import Console
from rich.table import Table

from rmv.constants import RADIOML_CLASSES
from rmv.dataset.cspb_detect import analyze_cspb_directory
from rmv.dataset.loader import _parse_cspb_truth_line, _parse_snr_key
from rmv.dataset.paths import (
    cspb_dir,
    detect_cspb,
    detect_cspb_present,
    find_cspb_truth_file,
    cspb_has_tim_files,
    detect_hisarmod,
    detect_radioml,
    radioml_pkl_path,
)
from rmv.dataset.radioml_resolve import find_radioml_pickle

console = Console(stderr=True)


def inspect_radioml(root: Path) -> dict[str, Any]:
    """Read RadioML pickle metadata without loading full dataset into RAM."""
    path = detect_radioml(root) or radioml_pkl_path(root)
    if path is None or not path.exists():
        return {"present": False}

    if path.suffix == ".pkl":
        pkl_path = path
    else:
        pkl_path = find_radioml_pickle(path.parent) or find_radioml_pickle(path)
        if pkl_path is None:
            return {"present": False, "error": "pickle not found"}

    with pkl_path.open("rb") as f:
        data: dict[tuple[str, str], np.ndarray] = pickle.load(f)

    mods: set[str] = set()
    snrs: list[float] = []
    n_samples = 0
    for (mod, snr_str), arr in data.items():
        mods.add(mod)
        try:
            snrs.append(float(snr_str))
        except ValueError:
            pass
        n_samples += int(arr.shape[0])

    return {
        "present": True,
        "path": str(pkl_path),
        "classes": sorted(mods),
        "expected_classes": RADIOML_CLASSES,
        "num_keys": len(data),
        "total_windows": n_samples,
        "snr_db_min": min(snrs) if snrs else None,
        "snr_db_max": max(snrs) if snrs else None,
        "window_shape": "(1000, 2, 128) typical",
    }


def inspect_hisarmod(root: Path) -> dict[str, Any]:
    """Inspect HISARMOD HDF5 structure."""
    path = detect_hisarmod(root)
    if path is None or not path.is_file():
        return {"present": False}

    classes: list[str] = []
    snrs: list[float] = []
    n_samples = 0

    with h5py.File(path, "r") as hf:
        for key in sorted(hf.keys()):
            if key.startswith("__"):
                continue
            classes.append(key)
            grp = hf[key]
            if not isinstance(grp, h5py.Group):
                continue
            for snr_key in grp.keys():
                try:
                    snrs.append(_parse_snr_key(str(snr_key)))
                except Exception:
                    pass
                ds = grp[snr_key]
                if hasattr(ds, "shape") and len(ds.shape) >= 1:
                    n_samples += int(ds.shape[0])

    return {
        "present": True,
        "path": str(path),
        "classes": classes,
        "num_classes": len(classes),
        "total_signals": n_samples,
        "snr_db_min": min(snrs) if snrs else None,
        "snr_db_max": max(snrs) if snrs else None,
        "format": "HDF5",
    }


def inspect_cspb(root: Path) -> dict[str, Any]:
    """Inspect CSPB directory: truth file, tim count, variant."""
    cdir = cspb_dir(root)
    if not cdir.is_dir():
        return {"present": False}

    variant = analyze_cspb_directory(cdir).value
    truth_path = find_cspb_truth_file(cdir)

    n_tim = 1 if cspb_has_tim_files(cdir) else 0
    mods: set[str] = set()
    n_truth = 0
    if truth_path and truth_path.is_file():
        for line in truth_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = _parse_cspb_truth_line(line)
            if parsed:
                mods.add(parsed[1])
                n_truth += 1

    return {
        "present": detect_cspb_present(root) is not None,
        "trainable": detect_cspb(root) is not None,
        "path": str(cdir),
        "variant": variant,
        "truth_file": str(truth_path) if truth_path else None,
        "truth_records": n_truth,
        "modulation_types": sorted(mods),
        "tim_files": n_tim,
    }


def print_dataset_info(root: Path) -> dict[str, Any]:
    """Print rich tables and return combined info dict."""
    info = {
        "schema_version": "1.0",
        "radioml": inspect_radioml(root),
        "hisarmod": inspect_hisarmod(root),
        "cspb": inspect_cspb(root),
    }

    for name, data in info.items():
        if name == "schema_version" or not isinstance(data, dict):
            continue
        table = Table(title=f"{name.upper()} dataset")
        table.add_column("Property")
        table.add_column("Value")
        if not data.get("present"):
            table.add_row("Status", "[red]not present[/]")
        else:
            for key, val in data.items():
                if key in ("present",):
                    continue
                if isinstance(val, list):
                    items = val[:20]
                    val = ", ".join(str(v) for v in items)
                    if len(val) > 20:  # noqa: PLR2004
                        val += " ..."
                table.add_row(key, str(val))
        console.print(table)

    return info
