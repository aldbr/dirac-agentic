#!/usr/bin/env python
"""CLI for dataset generation - Presentation Layer.

This module provides the command-line interface for dataset generation,
handling user input, progress display, and result presentation using Rich UI components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel

from dirac_dataset.business_logic.gen_dataset import (
    generate_dataset as generate_dataset_impl,
)

console = Console()
app = typer.Typer(add_completion=False)


def _create_progress() -> Progress:
    """Create a Rich progress bar with custom styling.

    Returns:
        Configured Progress instance with spinner, bar, and timing columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


@app.command("gen-dataset")
def generate_dataset(
    repos_file: Path = typer.Option(
        ..., "--repos-file", "-r", exists=True, help="TXT/JSON with repo URLs"
    ),
    pdfs_file: Path = typer.Option(
        ..., "--pdfs-file", "-p", exists=True, help="TXT/JSON with PDF URLs"
    ),
    out: Path = typer.Option(
        Path("data"), "--out", "-o", help="Destination for .txt payloads"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Rich logging at DEBUG level"
    ),
):
    """Generate dataset from GitHub repositories and PDF files.

    Downloads PDFs from provided URLs, processes documentation from GitHub
    repositories, and creates a HuggingFace Dataset with separate splits for
    papers and documentation.

    Args:
        repos_file: Path to JSON file containing repository configurations.
        pdfs_file: Path to JSON file containing PDF URLs to download.
        out: Output directory for the generated HuggingFace Dataset.
        verbose: Enable debug-level logging for detailed progress information.

    Raises:
        typer.Exit: With code 1 if generation fails due to validation or processing errors.
    """
    console.print(
        Panel.fit(
            "[bold blue]Dataset Generation[/bold blue]\n"
            f"📁 Repos: {repos_file}\n"
            f"📄 PDFs: {pdfs_file}\n"
            f"💾 Output: {out}",
            title="🚀 Starting Dataset Generation",
        )
    )

    progress_tasks: Dict[str, Any] = {}

    def progress_callback(task_name: str, current: int, total: int):
        """Handle progress updates from the dataset generation process.

        Args:
            task_name: Name of the current task being processed.
            current: Current progress count.
            total: Total expected count for the task.
        """
        if task_name not in progress_tasks:
            if task_name == "pdf_download":
                progress_tasks[task_name] = progress.add_task(
                    "[magenta]📥 Downloading PDFs", total=total
                )
            elif task_name == "repo_processing":
                progress_tasks[task_name] = progress.add_task(
                    "[cyan]📚 Processing repositories", total=total
                )

        progress.update(progress_tasks[task_name], completed=current)

    try:
        with _create_progress() as progress:
            result = generate_dataset_impl(
                repos_file, pdfs_file, out, verbose, progress_callback
            )

        # Display results in a nice table
        table = Table(title="📊 Dataset Generation Results")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Documents", style="magenta", justify="right")

        table.add_row("📄 PDF Papers", str(result["pdf_docs"]))
        table.add_row("📝 Documentation", str(result["md_docs"]))
        table.add_row(
            "[bold]Total Records[/bold]", f"[bold]{result['total_records']}[/bold]"
        )

        console.print(table)
        console.print(
            f"\n[bold green]✅ Dataset saved successfully to[/bold green] [blue]{result['output_path']}[/blue]"
        )

    except ValueError as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]❌ Unexpected error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
