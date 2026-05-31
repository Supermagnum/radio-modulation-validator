"""Typer-based CLI for radio-modulation-validator."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from rmv.api import RadioModulationValidator, format_summary_markdown
from rmv.checksum_util import update_checksums_for_dir, verify_all_models
from rmv.classifier import ModulationClassifier
from rmv.dataset.cli import register_dataset_commands
from rmv.plugins.cli import plugins_app
from rmv.validate import run_validate_cli

app = typer.Typer(
    name="rmv",
    help="Radio Modulation Validator - classify IQ and validate GNU Radio blocks.",
    no_args_is_help=True,
)
checksum_app = typer.Typer(help="Manage model checksums.")
app.add_typer(checksum_app, name="checksum")
register_dataset_commands(app)
app.add_typer(plugins_app, name="plugins")

console = Console(stderr=True)
err_console = Console(stderr=True)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


@app.command("validate")
def validate_cmd(
    iq_file_or_dir: Path = typer.Argument(..., help="IQ file or directory to validate"),
    threshold: float = typer.Option(0.70, "--threshold", help="Confidence threshold"),
    output: Optional[Path] = typer.Option(None, "--output", help="Write JSON results here"),
    verbose: bool = typer.Option(False, "--verbose", help="Print per-chunk predictions"),
    repo: Optional[str] = typer.Option(None, "--repo", help="Filter by source repo name"),
) -> None:
    """Validate IQ file(s) against sidecar metadata."""
    _setup_logging(verbose)
    models_dir = Path("models")
    classifier = ModulationClassifier(
        models_dir,
        confidence_threshold=threshold,
        verify_checksums=models_dir.joinpath("family_classifier.onnx").is_file(),
    )
    code = run_validate_cli(
        iq_file_or_dir,
        classifier,
        threshold=threshold,
        output=output,
        output_dir=Path("validation_results"),
        verbose=verbose,
        repo_filter=repo,
    )
    raise typer.Exit(code=code)


@app.command("train")
def train_cmd(
    radioml: Optional[Path] = typer.Option(
        None,
        "--radioml",
        help="RadioML pickle or tar (default: datasets/radioml/ if present)",
    ),
    hisarmod: Optional[Path] = typer.Option(
        None,
        "--hisarmod",
        help="HISARMOD HDF5 (default: datasets/hisarmod/HisarMod2019.1.h5)",
    ),
    cspb: Optional[Path] = typer.Option(
        None,
        "--cspb",
        help="CSPB.ML.2018R2 directory (default: datasets/cspb/)",
    ),
    synthetic: Optional[Path] = typer.Option(
        None,
        "--synthetic",
        help="Synthetic dataset (synthetic.npz or directory, default: datasets/synthetic/)",
    ),
    order_only: bool = typer.Option(
        False,
        "--order-only",
        help="Retrain order classifier only (skip family)",
    ),
    datasets_dir: Path = typer.Option(
        Path("datasets"),
        "--datasets-dir",
        help="Root for auto-detected datasets; used when paths above are omitted",
    ),
    cache: Path = typer.Option(Path(".cache"), "--cache", help="Preprocessed cache"),
    output: Path = typer.Option(Path("checkpoints"), "--output", help="Checkpoint directory"),
    epochs: int = typer.Option(50, "--epochs"),
    batch_size: int = typer.Option(512, "--batch-size"),
    device: str = typer.Option("auto", "--device", help="cuda | cpu | auto"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Download missing datasets without prompting",
    ),
) -> None:
    """
    Train family and order classifiers (requires [train] extras).

    If --radioml, --hisarmod, or --cspb are omitted, looks under --datasets-dir.
    When a dataset is missing, offers to download it (or use -y to download automatically).
    """
    _setup_logging(verbose)
    try:
        from rmv.dataset.manage import ensure_datasets_for_training
        from rmv.train import run_training, train_console

        r_path, h_path, c_path = ensure_datasets_for_training(
            datasets_dir,
            radioml,
            hisarmod,
            cspb,
            interactive=not yes,
            auto_download=yes,
        )
        s_path = synthetic
        if s_path is None:
            default_syn = datasets_dir / "synthetic"
            if (default_syn / "synthetic.npz").is_file():
                s_path = default_syn
        if r_path is None and h_path is None and c_path is None and s_path is None:
            err_console.print(
                "No training datasets found. Run: [bold]rmv dataset download[/] "
                "or pass --radioml / --hisarmod / --cspb / --synthetic paths."
            )
            raise typer.Exit(code=1)

        family_ckpt, order_ckpt = run_training(
            radioml=r_path,
            hisarmod=h_path,
            cspb=c_path,
            synthetic=s_path,
            cache=cache,
            output=output,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            order_only=order_only,
        )
        train_console.print("\n[bold green]Training complete[/]")
        train_console.print(f"  family checkpoint: {family_ckpt}")
        train_console.print(f"  order checkpoint: {order_ckpt}")
    except ImportError as exc:
        err_console.print("Training requires torch. Install with: uv sync --extra train")
        raise typer.Exit(code=1) from exc


@app.command("export")
def export_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Best .pt checkpoint"),
    output_dir: Path = typer.Option(Path("models"), "--output-dir"),
) -> None:
    """Export PyTorch checkpoint to ONNX and update checksums."""
    _setup_logging(False)
    try:
        from rmv.export import export_checkpoint
    except ImportError as exc:
        err_console.print(
            "Export requires PyTorch. Install with: [bold]uv sync --extra train[/]"
        )
        raise typer.Exit(code=1) from exc

    checksums = Path("checksums.sha256")
    try:
        paths = export_checkpoint(checkpoint, output_dir, checksums_path=checksums)
    except ImportError as exc:
        err_console.print(f"[red]Export failed:[/] {exc}")
        err_console.print("Install train extras: [bold]uv sync --extra train[/]")
        raise typer.Exit(code=1) from exc
    for p in paths:
        print(json.dumps({"exported": str(p), "schema_version": "1.0"}))


@app.command("classify")
def classify_cmd(
    iq_file: Path = typer.Argument(..., help="IQ file to classify"),
    chunk_size: int = typer.Option(1024, "--chunk-size"),
    threshold: float = typer.Option(0.70, "--threshold"),
    format: str = typer.Option("table", "--format", help="table | json"),
) -> None:
    """Classify IQ file without sidecar validation."""
    _setup_logging(False)
    validator = RadioModulationValidator(confidence_threshold=threshold, verify_checksums=False)
    result = validator.classify_file(iq_file, chunk_size=chunk_size)

    if format == "json":
        print(json.dumps(result.to_dict()))
        return

    table = Table(title=f"Classification: {iq_file.name}")
    table.add_column("Field")
    table.add_column("Value")
    fam_style = _style_for_confidence(result.family_confidence, threshold)
    ord_style = _style_for_confidence(result.order_confidence, threshold)
    table.add_row("Family", f"[{fam_style}]{result.family} ({result.family_confidence:.2f})[/]")
    table.add_row("Order", f"[{ord_style}]{result.order} ({result.order_confidence:.2f})[/]")
    console.print(table)


def _style_for_confidence(conf: float, threshold: float) -> str:
    if conf >= threshold:
        return "green"
    if conf < 0.40:
        return "red"
    return "yellow"


@app.command("report")
def report_cmd(
    results_dir: Path = typer.Argument(..., help="Directory with validation JSON files"),
    output: Optional[Path] = typer.Option(None, "--output"),
    format: str = typer.Option("markdown", "--format", help="json | markdown"),
) -> None:
    """Generate summary report from validation_results."""
    from rmv.types import ValidationResult

    results: list[ValidationResult] = []
    for path in sorted(results_dir.glob("**/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                results.append(_dict_to_validation(item))
        else:
            results.append(_dict_to_validation(data))

    validator = RadioModulationValidator(verify_checksums=False)
    summary = validator.summary_report(results, output_path=None)

    if format == "json":
        text = json.dumps(summary, indent=2)
    else:
        text = format_summary_markdown(results)

    if output is not None:
        output.write_text(text, encoding="utf-8")
    print(text)


def _dict_to_validation(data: dict[str, object]) -> object:
    from rmv.types import ValidationResult

    return ValidationResult(
        iq_file=str(data.get("iq_file", "")),
        block_name=str(data.get("block_name", "")),
        source_repo=str(data.get("source_repo", "")),
        expected_family=str(data.get("expected_family", "")),
        expected_order=str(data.get("expected_order", "")),
        predicted_family=str(data.get("predicted_family", "")),
        predicted_order=str(data.get("predicted_order", "")),
        family_confidence=float(data.get("family_confidence", 0.0)),
        order_confidence=float(data.get("order_confidence", 0.0)),
        family_pass=bool(data.get("family_pass", False)),
        order_pass=bool(data.get("order_pass", False)),
        snr_db=data.get("snr_db"),  # type: ignore[arg-type]
        timestamp=str(data.get("timestamp", "")),
        notes=str(data.get("notes", "")),
        hard_fail=bool(data.get("hard_fail", False)),
        hard_fail_reason=data.get("hard_fail_reason"),  # type: ignore[arg-type]
        custom_mode=data.get("custom_mode"),  # type: ignore[arg-type]
    )


@checksum_app.command("verify")
def checksum_verify(
    models_dir: Path = typer.Option(Path("models"), "--models-dir"),
    checksums: Path = typer.Option(Path("checksums.sha256"), "--checksums"),
) -> None:
    """Verify models/*.onnx against checksums.sha256."""
    _setup_logging(False)
    try:
        verified = verify_all_models(models_dir, checksums)
        print(json.dumps({"verified": verified, "schema_version": "1.0"}))
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"Checksum verify failed: {exc}")
        raise typer.Exit(code=1) from exc


@checksum_app.command("update")
def checksum_update(
    models_dir: Path = typer.Option(Path("models"), "--models-dir"),
    checksums: Path = typer.Option(Path("checksums.sha256"), "--checksums"),
) -> None:
    """Recompute and update checksums.sha256 for all models/*.onnx."""
    _setup_logging(False)
    try:
        count = update_checksums_for_dir(models_dir, checksums)
        print(json.dumps({"updated": count, "schema_version": "1.0"}))
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"Checksum update failed: {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    """Entry point for rmv console script."""
    app()


if __name__ == "__main__":
    main()
