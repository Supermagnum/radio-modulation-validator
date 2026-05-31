"""CLI for rmv scan command group."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rmv.scan.database import FindingsDB, normalize_scan_timestamp
from rmv.scan.paths import default_db_path, resolve_scan_directory
from rmv.scan.config import parse_project_name_list
from rmv.scan.pipeline import ScanRunOptions, run_scan

console = Console(stderr=True)

scan_app = typer.Typer(
    name="scan",
    help="Scan GNU Radio OOT project trees (read-only; prompts before writes).",
    no_args_is_help=True,
)


@scan_app.command("run")
def scan_run(
    directory: Path = typer.Argument(..., help="Root directory to scan for OOT projects"),
    gr3_prefix: Optional[Path] = typer.Option(None, "--gr3-prefix", help="GNU Radio 3 prefix"),
    gr4_prefix: Optional[Path] = typer.Option(None, "--gr4-prefix", help="GNU Radio 4 prefix"),
    iq_output: Optional[Path] = typer.Option(None, "--iq-output", help="Generated IQ output dir"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", help="ONNX models directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve IQ generation prompts"),
    yes_report: bool = typer.Option(
        False,
        "--yes-report",
        help="Auto-approve writing VALIDATION_REPORT.md (IQ still prompted unless -y)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Discover only; no IQ, DB, or reports"),
    filter_names: Optional[str] = typer.Option(
        None,
        "--filter",
        help="Comma-separated project names to include (default: gr-qradiolink, gr-packet-protocols, gr-sleipnir)",
    ),
    all_projects: bool = typer.Option(
        False,
        "--all",
        help="Scan all discovered OOT projects (ignore default include list)",
    ),
    threshold: float = typer.Option(0.70, "--threshold", help="Validation confidence threshold"),
) -> None:
    """
    Discover OOT projects, generate reference IQ (with approval), validate, report.

    Never modifies source files or runs git in scanned projects.
    """
    try:
        scan_root = resolve_scan_directory(directory)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        if "~" in str(directory):
            console.print(
                "[dim]Note: ~ is your home directory "
                f"({Path.home()}), not the current working directory.[/]"
            )
        raise typer.Exit(code=1) from None

    run_scan(
        ScanRunOptions(
            root=scan_root,
            gr3_prefix=gr3_prefix,
            gr4_prefix=gr4_prefix,
            iq_output=iq_output,
            models_dir=models_dir,
            yes=yes,
            yes_report=yes_report,
            dry_run=dry_run,
            include_names=parse_project_name_list(filter_names),
            scan_all=all_projects,
            threshold=threshold,
        )
    )


@scan_app.command("issues")
def scan_issues(
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project name"),
    severity: Optional[str] = typer.Option(None, "--severity", help="hard_fail|soft_fail|warning|info"),
    unresolved: bool = typer.Option(True, "--unresolved/--all", help="Show unresolved only"),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Only issues detected at or after this UTC timestamp (e.g. 2026-05-31T15:00:00)",
    ),
    format: str = typer.Option("table", "--format", help="table | json"),
) -> None:
    """List issues from the local findings database."""
    detected_since: str | None = None
    if since is not None:
        try:
            detected_since = normalize_scan_timestamp(since)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from None

    db = FindingsDB(default_db_path())
    try:
        rows = db.list_issues(
            project_name=project,
            severity=severity,
            unresolved_only=unresolved,
            detected_since=detected_since,
        )
    finally:
        db.close()

    if format == "json":
        print(json.dumps(rows, indent=2))
        return

    table = Table(title="Open issues" if unresolved else "All issues")
    table.add_column("ID")
    table.add_column("Project")
    table.add_column("Block")
    table.add_column("Severity")
    table.add_column("Description")
    for row in rows:
        table.add_row(
            str(row["id"]),
            str(row.get("project_name", "")),
            str(row.get("block_name", "")),
            str(row["severity"]),
            str(row["description"])[:80],
        )
    console.print(table)


@scan_app.command("purge")
def scan_purge(
    keep_latest: bool = typer.Option(
        False,
        "--keep-latest",
        help="Keep only the latest validation per block; delete older validations and issues.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """
    Remove stale rows from previous scan runs in .rmv_findings.db.

    Project and block rows are not deleted.
    """
    if not keep_latest:
        console.print("[red]Specify --keep-latest to purge stale validations and issues.[/]")
        raise typer.Exit(code=1)

    db_path = default_db_path()
    if not db_path.is_file():
        console.print(f"[red]Database not found:[/] {db_path}")
        raise typer.Exit(code=1)

    db = FindingsDB(db_path)
    try:
        preview = db.preview_purge_keep_latest()
    finally:
        db.close()

    if preview.validations_to_delete == 0 and preview.issues_to_delete == 0:
        console.print("[green]Nothing to purge; database already has only latest run data.[/]")
        return

    console.print(f"Database: {db_path}")
    console.print(
        f"Will delete {preview.validations_to_delete} validation row(s) "
        f"and {preview.issues_to_delete} issue row(s)."
    )
    console.print(
        f"Will keep {preview.validations_to_keep} validation row(s) "
        f"and {preview.issues_to_keep} issue row(s)."
    )
    if not yes and not typer.confirm("Delete stale scan data?", default=False):
        console.print("[yellow]Purge cancelled.[/]")
        raise typer.Exit(code=0)

    db = FindingsDB(db_path)
    try:
        result = db.purge_keep_latest()
    finally:
        db.close()

    console.print(
        f"[green]Purged {result.validations_to_delete} validation(s) "
        f"and {result.issues_to_delete} issue(s).[/]"
    )


@scan_app.command("resolve")
def scan_resolve(
    issue_id: int = typer.Argument(..., help="Issue ID from rmv scan issues"),
    note: Optional[str] = typer.Option(None, "--note", help="Optional resolution note"),
) -> None:
    """Mark an issue as resolved."""
    db = FindingsDB(default_db_path())
    try:
        ok = db.resolve_issue(issue_id, note)
    finally:
        db.close()
    if not ok:
        console.print(f"[red]Issue {issue_id} not found[/]")
        raise typer.Exit(code=1)
    console.print(f"[green]Issue {issue_id} marked resolved[/]")


@scan_app.command("status")
def scan_status() -> None:
    """Show database summary."""
    db = FindingsDB(default_db_path())
    try:
        summary = db.status_summary()
    finally:
        db.close()

    table = Table(title="rmv scan database status")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Projects", str(summary["projects"]))
    table.add_row("Validations", str(summary["validations"]))
    table.add_row("Last scan", str(summary.get("last_scan") or "-"))
    for sev, count in summary.get("open_issues", {}).items():
        table.add_row(f"Open {sev}", str(count))
    console.print(table)


@scan_app.command("list")
def scan_list() -> None:
    """List projects recorded in the database."""
    db = FindingsDB(default_db_path())
    try:
        projects = db.list_projects()
    finally:
        db.close()

    table = Table(title="Scanned projects")
    table.add_column("Name")
    table.add_column("GR")
    table.add_column("Last scanned")
    table.add_column("Status")
    for p in projects:
        table.add_row(
            str(p["name"]),
            str(p.get("gr_version", "")),
            str(p.get("last_scanned", "")),
            str(p.get("scan_status", "")),
        )
    console.print(table)


@scan_app.command("report")
def scan_report(
    project_name: str = typer.Argument(..., help="Project name in database"),
) -> None:
    """Re-print validation summary from database (no re-validation)."""
    db = FindingsDB(default_db_path())
    try:
        validations = db.latest_validations_for_project(project_name)
        projects = db.list_projects()
    finally:
        db.close()

    proj = next((p for p in projects if p["name"] == project_name), None)
    if proj is None:
        console.print(f"[red]Project not in database:[/] {project_name}")
        raise typer.Exit(code=1)

    table = Table(title=f"Last validations — {project_name}")
    table.add_column("Block")
    table.add_column("Expected")
    table.add_column("Predicted")
    table.add_column("F-pass")
    table.add_column("O-pass")
    for v in validations:
        table.add_row(
            str(v["block_name"]),
            f"{v['expected_family']}/{v['expected_order']}",
            f"{v['predicted_family']}/{v['predicted_order']}",
            "Y" if v["family_pass"] else "N",
            "Y" if v["order_pass"] else "N",
        )
    console.print(table)


def register_scan_commands(app: typer.Typer) -> None:
    """Attach scan command group to main rmv app."""
    app.add_typer(scan_app, name="scan")
