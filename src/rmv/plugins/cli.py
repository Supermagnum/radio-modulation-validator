"""CLI for custom-mode plugin discovery."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from rmv.plugins.registry import get, list_plugins

plugins_app = typer.Typer(help="Custom-mode validation plugins.")
console = Console(stderr=True)


@plugins_app.command("list")
def plugins_list() -> None:
    """List registered custom mode plugins."""
    table = Table(title="Custom mode plugins")
    table.add_column("mode_id")
    table.add_column("description")
    for mode_id in list_plugins():
        plugin = get(mode_id)
        desc = plugin.description if plugin is not None else ""
        table.add_row(mode_id, desc)
    console.print(table)


@plugins_app.command("describe")
def plugins_describe(
    mode_id: str = typer.Argument(..., help="Plugin mode_id (e.g. sleipnir_8qpsk)"),
) -> None:
    """Print measurements and pass criteria for a plugin."""
    plugin = get(mode_id)
    if plugin is None:
        console.print(f"[red]Unknown plugin:[/] {mode_id}")
        raise typer.Exit(code=1)
    console.print(json.dumps(plugin.describe(), indent=2))
