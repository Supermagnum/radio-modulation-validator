"""Dataset status, verification, checksum maintenance, and train-path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rmv.dataset.checksums import is_verified_checksum, sha256_file, verify_file_checksum
from rmv.dataset.cspb_detect import analyze_cspb_directory, CSPBVariant
from rmv.dataset.download import DownloadError, download_datasets
from rmv.dataset.manifest import (
    checksum_prefix_from_manifest,
    get_dataset_entry,
    load_manifest,
    refresh_manifest_checksums,
)
from rmv.dataset.paths import (
    DEFAULT_DATASETS_ROOT,
    cspb_dir,
    detect_cspb,
    detect_cspb_present,
    detect_hisarmod,
    detect_radioml,
    find_cspb_truth_file,
    hisarmod_h5_path,
    radioml_pkl_path,
    radioml_tar_path,
)

console = Console(stderr=True)


@dataclass
class DatasetStatusRow:
    name: str
    status: str
    size: str
    path: str
    checksum: str


def _format_size(path: Path) -> str:
    if not path.exists():
        return "-"
    if path.is_file():
        nbytes = path.stat().st_size
    else:
        nbytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _status_from_manifest(root: Path, dataset_key: str, path: Path | None) -> str:
    if path is None or not path.exists():
        return "missing"
    if dataset_key == "cspb":
        if not path.is_dir():
            return "missing"
        variant = analyze_cspb_directory(path)
        if variant == CSPBVariant.ORIGINAL:
            return "warning (original, not R2)"
        if detect_cspb(root):
            return "present"
        if detect_cspb_present(root):
            return "present (no truth file)"
        return "incomplete"
    if dataset_key == "radioml":
        if detect_radioml(root):
            return "present"
        if path.is_file() and path.stat().st_size > 0:
            return "incomplete"
        return "missing"
    if dataset_key == "hisarmod":
        if detect_hisarmod(root):
            return "present"
        if path.is_file() and path.stat().st_size > 0:
            return "incomplete"
        return "missing"
    if path.is_file() and path.stat().st_size > 0:
        return "present"
    return "incomplete"


def collect_status(root: Path = DEFAULT_DATASETS_ROOT) -> list[DatasetStatusRow]:
    """Collect status from manifest (no full re-hash)."""
    rows: list[DatasetStatusRow] = []

    r_path = detect_radioml(root)
    r_display = r_path or radioml_tar_path(root)
    rows.append(
        DatasetStatusRow(
            name="RadioML 2016.10A",
            status=_status_from_manifest(root, "radioml", r_display if r_display.exists() else None),
            size=_format_size(r_display) if r_display.exists() else "-",
            path=str(r_display) if r_display.exists() else str(radioml_pkl_path(root)),
            checksum=checksum_prefix_from_manifest(root, "radioml"),
        )
    )

    h_path = detect_hisarmod(root) or hisarmod_h5_path(root)
    rows.append(
        DatasetStatusRow(
            name="HISARMOD 2019.1",
            status=_status_from_manifest(root, "hisarmod", h_path if h_path.is_file() else None),
            size=_format_size(h_path),
            path=str(h_path),
            checksum=checksum_prefix_from_manifest(root, "hisarmod"),
        )
    )

    c_path = detect_cspb_present(root) or cspb_dir(root)
    rows.append(
        DatasetStatusRow(
            name="CSPB.ML.2018R2",
            status=_status_from_manifest(root, "cspb", c_path if c_path.exists() else None),
            size=_format_size(c_path) if c_path.exists() else "-",
            path=str(c_path),
            checksum=checksum_prefix_from_manifest(root, "cspb"),
        )
    )
    return rows


def print_status_table(root: Path = DEFAULT_DATASETS_ROOT) -> None:
    table = Table(title="Dataset status")
    table.add_column("Dataset")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Path")
    table.add_column("SHA-256 (prefix)")

    for row in collect_status(root):
        style = (
            "green"
            if row.status == "present"
            else "red"
            if row.status in ("missing", "corrupt")
            else "yellow"
        )
        table.add_row(
            row.name,
            f"[{style}]{row.status}[/]",
            row.size,
            row.path,
            row.checksum,
        )
    console.print(table)


def verify_all_datasets(root: Path = DEFAULT_DATASETS_ROOT) -> bool:
    """Verify datasets by re-hashing against manifest (explicit verify step)."""
    failures = False
    manifest = load_manifest(root)

    r = detect_radioml(root)
    if r is None:
        console.print("[red]RadioML 2016.10A:[/] missing")
        failures = True
    elif r.is_file():
        key = "radioml/RML2016.10a_dict.pkl" if r.suffix == ".pkl" else "radioml/RML2016.10a.tar.bz2"
        ok, msg = verify_file_checksum(r, key, datasets_root=root)
        if not ok:
            console.print(f"[red]RadioML 2016.10A:[/] {msg}")
            failures = True
        else:
            console.print(f"[green]RadioML 2016.10A:[/] {msg}")

    h = detect_hisarmod(root)
    if h is None:
        console.print("[yellow]HISARMOD 2019.1:[/] missing (optional)")
    else:
        ok, msg = verify_file_checksum(h, "hisarmod/HisarMod2019.1.h5", datasets_root=root)
        if not ok:
            console.print(f"[red]HISARMOD 2019.1:[/] {msg}")
            failures = True
        else:
            console.print(f"[green]HISARMOD 2019.1:[/] {msg}")

    cdir = cspb_dir(root)
    c_present = detect_cspb_present(root)
    if c_present is None:
        if cdir.is_dir() and analyze_cspb_directory(cdir) == CSPBVariant.ORIGINAL:
            console.print("[red]CSPB.ML.2018R2:[/] original 2018 detected (RNG flaw); use R2")
            failures = True
        else:
            console.print("[red]CSPB.ML.2018R2:[/] missing or incomplete")
            failures = True
    elif analyze_cspb_directory(c_present) == CSPBVariant.ORIGINAL:
        console.print("[red]CSPB.ML.2018R2:[/] original 2018 detected (RNG flaw); use R2")
        failures = True
    else:
        truth = find_cspb_truth_file(c_present)
        cspb_ok = True
        entry = manifest.get("datasets", {}).get("cspb", {})
        files = entry.get("files", {}) if isinstance(entry, dict) else {}
        if isinstance(files, dict):
            for fname, expected in files.items():
                fpath = c_present / fname
                if fpath.is_file() and is_verified_checksum(str(expected)):
                    if sha256_file(fpath).lower() != str(expected).lower():
                        console.print(f"[red]CSPB.ML.2018R2:[/] corrupt: {fname}")
                        cspb_ok = False
                        failures = True
        if cspb_ok:
            if truth is not None:
                console.print("[green]CSPB.ML.2018R2:[/] ok (signals and truth file)")
            else:
                console.print(
                    "[yellow]CSPB.ML.2018R2:[/] signals present; "
                    "truth file missing (training labels unavailable)"
                )

    return not failures


def checksum_update_dataset(dataset: str, root: Path = DEFAULT_DATASETS_ROOT) -> None:
    """Compute SHA-256 and update datasets/.manifest.json (not package source)."""
    refresh_manifest_checksums(root, dataset.lower())
    console.print(f"[green]Updated manifest for {dataset}[/] at {root / '.manifest.json'}")


def resolve_train_paths(
    root: Path,
    radioml: Path | None,
    hisarmod: Path | None,
    cspb: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    return (
        radioml if radioml is not None else detect_radioml(root),
        hisarmod if hisarmod is not None else detect_hisarmod(root),
        cspb if cspb is not None else detect_cspb(root),
    )


def ensure_datasets_for_training(
    root: Path,
    radioml: Path | None,
    hisarmod: Path | None,
    cspb: Path | None,
    *,
    interactive: bool = True,
    auto_download: bool = False,
) -> tuple[Path | None, Path | None, Path | None]:
    r_path, h_path, c_path = resolve_train_paths(root, radioml, hisarmod, cspb)

    if r_path is not None or h_path is not None or c_path is not None:
        return r_path, h_path, c_path

    missing_specs: list[tuple[str, Path, bool, bool, bool]] = [
        ("RadioML 2016.10A", radioml_pkl_path(root), True, False, False),
        ("HISARMOD 2019.1", hisarmod_h5_path(root), False, True, False),
        ("CSPB.ML.2018R2", cspb_dir(root), False, False, True),
    ]

    for name, expected_path, want_r, want_h, want_c in missing_specs:
        console.print(f"[yellow]{name} not found[/] in {expected_path.parent}/")
        do_download = auto_download
        if interactive and not auto_download:
            try:
                import typer

                do_download = typer.confirm("Download now?", default=True)
            except Exception:
                do_download = False
        if do_download:
            download_datasets(
                root,
                radioml=want_r,
                hisarmod=want_h,
                cspb=want_c,
                force=False,
                verify_only=False,
            )

    return resolve_train_paths(root, radioml, hisarmod, cspb)
