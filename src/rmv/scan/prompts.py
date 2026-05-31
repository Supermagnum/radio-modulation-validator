"""User prompts for rmv scan (Rich + Typer)."""

from __future__ import annotations

from typing import Literal

import typer
from rich.console import Console
from rich.panel import Panel

from rmv.scan.discover import GRProject
from rmv.scan.readme_parser import ReadmeSummary

console = Console(stderr=True)

_skip_all_iq: bool = False
_auto_yes_iq: bool = False
_auto_yes_report: bool = False


def reset_prompt_state() -> None:
    global _skip_all_iq, _auto_yes_iq, _auto_yes_report
    _skip_all_iq = False
    _auto_yes_iq = False
    _auto_yes_report = False


def set_auto_yes_iq(value: bool) -> None:
    global _auto_yes_iq
    _auto_yes_iq = value


def set_auto_yes_report(value: bool) -> None:
    global _auto_yes_report
    _auto_yes_report = value


def set_skip_all_iq(value: bool) -> None:
    global _skip_all_iq
    _skip_all_iq = value


def ask_generate_iq(
    project: GRProject,
    summary: ReadmeSummary,
) -> Literal["yes", "no", "skip_all"]:
    """Prompt before generating IQ for one project."""
    global _skip_all_iq
    if _skip_all_iq:
        return "no"
    if _auto_yes_iq:
        return "yes"

    modes = ", ".join(summary.modulation_modes[:12])
    if len(summary.modulation_modes) > 12:
        modes += ", ..."

    body = (
        f"[bold]Found GNU Radio project:[/] {project.name}\n"
        f"Path: {project.path}\n"
        f"GR version: {project.gr_version}\n"
        f"AI-generated: {'yes' if summary.is_ai_generated else 'no'}\n"
        f"Modes found in README: {modes or '(none)'}\n\n"
        "Generate IQ and validate this project? [Y/n/s]\n"
        "(Y=yes, n=no, s=skip all remaining prompts)"
    )
    console.print(Panel(body, title="rmv scan", border_style="cyan"))
    choice = typer.prompt("Choice", default="Y").strip().lower()
    if choice in ("s", "skip"):
        _skip_all_iq = True
        return "skip_all"
    if choice in ("n", "no"):
        return "no"
    return "yes"


def ask_write_report(project: GRProject, report_path: Path) -> bool:
    """Prompt before writing VALIDATION_REPORT.md into a scanned project."""
    if _auto_yes_report:
        return True
    msg = (
        f"Write VALIDATION_REPORT.md to\n  {report_path}\n"
        "[Y/n]:"
    )
    choice = typer.prompt(msg, default="Y").strip().lower()
    return choice not in ("n", "no")
