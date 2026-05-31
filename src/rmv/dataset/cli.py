"""CLI handlers for rmv dataset command group."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rmv.dataset.download import (
    DownloadError,
    convert_hisarmod_mat_to_h5,
    download_datasets,
    set_download_timeout,
)
from rmv.dataset.info import print_dataset_info
from rmv.dataset.manage import (
    checksum_update_dataset,
    collect_status,
    print_status_table,
    verify_all_datasets,
)
from rmv.dataset.paths import DEFAULT_DATASETS_ROOT
from rmv.dataset.synthetic import (
    ALL_MODES,
    MODE_TO_CLASS,
    SNR_DB_LEVELS,
    SyntheticMode,
    generate_synthetic,
    save_synthetic_dataset,
)

console = Console(stderr=True)

dataset_app = typer.Typer(
    name="dataset",
    help="Download, verify, and prepare training datasets.",
    no_args_is_help=True,
)


@dataset_app.command("download")
def dataset_download(
    dest: Path = typer.Option(DEFAULT_DATASETS_ROOT, "--dest", help="Root directory for datasets"),
    radioml: bool = typer.Option(False, "--radioml", help="Download RadioML only"),
    hisarmod: bool = typer.Option(False, "--hisarmod", help="Download HISARMOD only"),
    cspb: bool = typer.Option(False, "--cspb", help="Download CSPB.ML.2018R2 only"),
    verify_only: bool = typer.Option(False, "--verify-only", help="Skip download, verify only"),
    force: bool = typer.Option(False, "--force", help="Re-download existing files"),
    timeout: float = typer.Option(
        300.0,
        "--timeout",
        help="HTTP timeout in seconds for large downloads (default 300)",
    ),
) -> None:
    """
    Download and prepare training datasets (RadioML, HISARMOD, CSPB R2).

    Proxy: set HTTP_PROXY / HTTPS_PROXY environment variables (httpx uses them automatically).
    """
    set_download_timeout(timeout)
    flags = (radioml, hisarmod, cspb)
    download_all = not any(flags)

    try:
        ok = download_datasets(
            dest,
            radioml=download_all or radioml,
            hisarmod=download_all or hisarmod,
            cspb=download_all or cspb,
            force=force,
            verify_only=verify_only,
        )
        console.print()
        print_status_table(dest)
        if not ok:
            raise typer.Exit(code=1)
    except DownloadError as exc:
        console.print(f"[red]Download failed:[/] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        console.print(f"[red]Download failed:[/] {exc}")
        raise typer.Exit(code=1) from None


@dataset_app.command("status")
def dataset_status(
    dest: Path = typer.Option(DEFAULT_DATASETS_ROOT, "--dest", help="Datasets root directory"),
) -> None:
    """Show dataset presence, paths, and checksum status."""
    print_status_table(dest)
    rows = collect_status(dest)
    summary = {
        "schema_version": "1.0",
        "datasets": [
            {
                "name": r.name,
                "status": r.status,
                "size": r.size,
                "path": r.path,
                "checksum_prefix": r.checksum,
            }
            for r in rows
        ],
    }
    import json

    print(json.dumps(summary))


@dataset_app.command("verify")
def dataset_verify(
    dest: Path = typer.Option(DEFAULT_DATASETS_ROOT, "--dest", help="Datasets root directory"),
) -> None:
    """Verify checksums of all downloaded dataset files."""
    try:
        ok = verify_all_datasets(dest)
    except DownloadError as exc:
        console.print(f"[red]Verify failed:[/] {exc}")
        raise typer.Exit(code=1) from None
    raise typer.Exit(code=0 if ok else 1)


@dataset_app.command("info")
def dataset_info(
    dest: Path = typer.Option(DEFAULT_DATASETS_ROOT, "--dest", help="Datasets root directory"),
) -> None:
    """Show class names, sample counts, and SNR ranges from downloaded files."""
    import json

    info = print_dataset_info(dest)
    print(json.dumps(info, indent=2))


@dataset_app.command("checksum-update")
def dataset_checksum_update(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Dataset to update: radioml, hisarmod, or cspb",
    ),
    dest: Path = typer.Option(DEFAULT_DATASETS_ROOT, "--dest", help="Datasets root directory"),
) -> None:
    """Update datasets/.manifest.json checksums after verifying a new dataset version."""
    try:
        checksum_update_dataset(dataset.lower(), dest)
    except DownloadError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None


@dataset_app.command("convert-hisarmod")
def dataset_convert_hisarmod(
    input_path: Path = typer.Option(..., "--input", help="HISARMOD .mat file or directory"),
    output: Path = typer.Option(
        DEFAULT_DATASETS_ROOT / "hisarmod" / "HisarMod2019.1.h5",
        "--output",
        help="Output HDF5 path",
    ),
) -> None:
    """Convert HISARMOD .mat files to HDF5 for the loader."""
    try:
        if input_path.is_dir():
            mats = list(input_path.glob("*.mat"))
            if not mats:
                raise DownloadError(f"No .mat files in {input_path}")
            convert_hisarmod_mat_to_h5(mats[0], output)
        else:
            convert_hisarmod_mat_to_h5(input_path, output)
    except DownloadError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None


@dataset_app.command("generate-synthetic")
def dataset_generate_synthetic(
    modes: Optional[str] = typer.Option(
        None,
        "--modes",
        help="Comma-separated: nbfm25,nbfm50,am_air_25k,am_air_833 (default: all)",
    ),
    chunks_per_snr: int = typer.Option(
        1000,
        "--chunks-per-snr",
        help="IQ chunks per SNR level per mode",
    ),
    output: Path = typer.Option(
        DEFAULT_DATASETS_ROOT / "synthetic",
        "--output",
        help="Output directory for synthetic.npz",
    ),
    sample_rate: int = typer.Option(48000, "--sample-rate", help="IQ sample rate in Hz"),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify occupied bandwidth before saving",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed for reproducibility"),
) -> None:
    """
    Generate synthetic NBFM and aviation AM training data (GNU Radio / scipy only).

    Output is written under datasets/synthetic/ by default (gitignored).
    """
    selected: list[SyntheticMode]
    if modes is None:
        selected = list(ALL_MODES)
    else:
        raw = [m.strip().lower() for m in modes.split(",") if m.strip()]
        invalid = [m for m in raw if m not in MODE_TO_CLASS]
        if invalid:
            console.print(f"[red]Unknown mode(s):[/] {', '.join(invalid)}")
            console.print(f"Valid: {', '.join(ALL_MODES)}")
            raise typer.Exit(code=1)
        selected = raw  # type: ignore[assignment]

    try:
        console.print(
            f"Generating synthetic data: {', '.join(MODE_TO_CLASS[m] for m in selected)} "
            f"({chunks_per_snr} chunks x {len(SNR_DB_LEVELS)} SNR levels each)..."
        )
        dataset = generate_synthetic(
            selected,
            chunks_per_snr=chunks_per_snr,
            sample_rate_hz=float(sample_rate),
            verify=verify,
            seed=seed,
        )
        path = save_synthetic_dataset(output, dataset)
        console.print(
            f"[green]Done[/] {len(dataset.samples):,} samples -> {path} "
            f"({', '.join(dataset.class_names)})"
        )
    except ValueError as exc:
        console.print(f"[red]Generation failed:[/] {exc}")
        raise typer.Exit(code=1) from None
    except ImportError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None


def register_dataset_commands(app: typer.Typer) -> None:
    """Attach dataset command group to main rmv app."""
    app.add_typer(dataset_app, name="dataset")
