"""Orchestrate rmv scan run across discovered projects."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rmv.classifier import ModulationClassifier
from rmv.scan.config import ScanConfig, load_scan_config
from rmv.scan.database import FindingsDB
from rmv.scan.discover import GRProject, discover_gr_projects
from rmv.scan.exclusions import merged_exclude_projects, resolve_include_names
from rmv.scan.gr_env import build_gr3_env, build_gr4_env, find_gr3_prefix, find_gr4_prefix
from rmv.scan.class_vocab import load_classifier_vocab, print_mode_label_warnings
from rmv.scan.iq_generator import generate_iq_for_project
from rmv.scan.paths import default_db_path, default_iq_output, find_rmv_project_root
from rmv.scan.prompts import (
    ask_generate_iq,
    ask_write_report,
    reset_prompt_state,
    set_auto_yes_iq,
    set_auto_yes_report,
)
from rmv.scan.readme_parser import ReadmeSummary, parse_readme
from rmv.scan.report_writer import write_project_report
from rmv.scan.runner import ProjectValidationRun, run_validation

logger = logging.getLogger(__name__)
console = Console(stderr=True)


@dataclass
class ScanRunOptions:
    root: Path
    gr3_prefix: Path | None = None
    gr4_prefix: Path | None = None
    iq_output: Path | None = None
    models_dir: Path = Path("models")
    yes: bool = False
    yes_report: bool = False
    dry_run: bool = False
    include_names: frozenset[str] | None = None
    exclude_names: frozenset[str] | None = None
    scan_all: bool = False
    threshold: float = 0.70


def _print_project_line(run: ProjectValidationRun) -> None:
    name = run.project.name
    if run.hard_failed > 0:
        style = "red"
        symbol = "FAIL"
    elif run.soft_failed > 0:
        style = "yellow"
        symbol = "WARN"
    elif run.passed > 0 and run.hard_failed == 0:
        style = "green"
        symbol = "OK"
    else:
        style = "dim"
        symbol = "SKIP"
    validated = run.passed + run.soft_failed + run.hard_failed
    console.print(
        f"[{style}]{symbol}[/{style}] {name:<24} {run.passed}/{validated} passed  "
        f"{run.soft_failed} soft fail  {run.hard_failed} hard fail  {run.skipped} skipped"
    )


def run_scan(options: ScanRunOptions) -> list[ProjectValidationRun]:
    """Discover, optionally generate IQ, validate, optionally write reports."""
    reset_prompt_state()
    if options.yes:
        set_auto_yes_iq(True)
    if options.yes_report:
        set_auto_yes_report(True)

    cfg = load_scan_config()
    rmv_root = find_rmv_project_root()
    db_path = default_db_path(rmv_root)
    iq_out = options.iq_output or (rmv_root / cfg.iq_output)
    if not options.iq_output:
        iq_out = default_iq_output(rmv_root)

    gr3_p = find_gr3_prefix(options.gr3_prefix or (Path(cfg.gr3_prefix) if cfg.gr3_prefix else None))
    gr4_p = find_gr4_prefix(options.gr4_prefix or (Path(cfg.gr4_prefix) if cfg.gr4_prefix else None))
    gr3_env = build_gr3_env(gr3_p) if gr3_p else None
    gr4_env = build_gr4_env(gr4_p) if gr4_p else None

    extra_exclude = frozenset(cfg.exclude_projects)
    if options.exclude_names:
        extra_exclude = extra_exclude | options.exclude_names
    include_names = resolve_include_names(
        cli_filter=options.include_names,
        config_includes=cfg.include_projects,
        scan_all=options.scan_all,
    )
    projects = discover_gr_projects(
        options.root,
        include_names=include_names,
        exclude_names=extra_exclude if extra_exclude else None,
    )

    if not projects:
        console.print(f"[yellow]No GNU Radio OOT projects found under {options.root}[/]")
        return []

    console.print(f"Discovered {len(projects)} project(s) under {options.root}")

    if options.dry_run:
        table = Table(title="Discovered projects (dry run)")
        table.add_column("Name")
        table.add_column("GR version")
        table.add_column("Path")
        for p in projects:
            table.add_row(p.name, p.gr_version, str(p.path))
        console.print(table)
        if gr4_p is None:
            console.print("[yellow]GR4 prefix not found; GR4-only steps will be skipped.[/]")
        return []

    db = FindingsDB(db_path)
    vocab = load_classifier_vocab(options.models_dir)
    print_mode_label_warnings(vocab)
    classifier = ModulationClassifier(options.models_dir, verify_checksums=False)
    runs: list[ProjectValidationRun] = []

    try:
        for project in projects:
            if project.readme_path and project.readme_path.is_file():
                summary = parse_readme(project.readme_path)
            elif (project.path / "README.md").is_file():
                summary = parse_readme(project.path / "README.md")
            else:
                summary = ReadmeSummary()

            if project.gr_version == "4" and gr4_p is None:
                console.print(
                    f"[yellow]Skipping {project.name}: GR4 prefix not found "
                    f"(set GNURADIO4_PREFIX or --gr4-prefix)[/]"
                )
                db.upsert_project(
                    path=str(project.path),
                    name=project.name,
                    gr_version=project.gr_version,
                    readme_path=str(project.readme_path) if project.readme_path else None,
                    scan_status="skipped",
                )
                continue

            choice = ask_generate_iq(project, summary)
            if choice in ("no", "skip_all"):
                db.upsert_project(
                    path=str(project.path),
                    name=project.name,
                    gr_version=project.gr_version,
                    readme_path=str(project.readme_path) if project.readme_path else None,
                    scan_status="skipped",
                )
                continue

            generated = generate_iq_for_project(
                project,
                summary,
                iq_out,
                gr3_env=gr3_env,
                gr4_env=gr4_env,
                vocab=vocab,
            )
            run = run_validation(project, generated, db, classifier, threshold=options.threshold)
            runs.append(run)
            _print_project_line(run)

            report_path = project.path / "VALIDATION_REPORT.md"
            if ask_write_report(project, report_path):
                write_project_report(project, run, report_path, summary=summary)
                console.print(f"  Report written: {report_path}")
    finally:
        db.close()

    if runs:
        table = Table(title="Scan summary")
        table.add_column("Project")
        table.add_column("Result")
        table.add_column("Passed")
        table.add_column("Soft fail")
        table.add_column("Hard fail")
        table.add_column("Skipped")
        for run in runs:
            table.add_row(
                run.project.name,
                _overall_label(run),
                str(run.passed),
                str(run.soft_failed),
                str(run.hard_failed),
                str(run.skipped),
            )
        console.print(table)

    db_summary = FindingsDB(db_path).status_summary()
    FindingsDB(db_path).close()
    open_issues = sum(db_summary.get("open_issues", {}).values())
    console.print(f"Open issues in database: {open_issues}")
    console.print("Run [bold]rmv scan issues[/] to see all open issues.")

    return runs


def _overall_label(run: ProjectValidationRun) -> str:
    if run.hard_failed > 0:
        return "HARD FAIL"
    if run.soft_failed > 0:
        return "SOFT FAIL"
    if run.passed > 0:
        return "PASS"
    return "SKIPPED"
