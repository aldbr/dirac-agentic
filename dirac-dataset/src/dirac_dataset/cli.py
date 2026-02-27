"""CLI commands for dirac-dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from datasets import DatasetDict
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from dirac_dataset.generator import generate_dataset as generate_dataset_impl

console = Console()


def _create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# gen-dataset
# ---------------------------------------------------------------------------


def gen_dataset(
    repos_file: Path = typer.Option(
        ..., "--repos-file", "-r", exists=True, help="JSON with repo URLs"
    ),
    pdfs_file: Path = typer.Option(
        ..., "--pdfs-file", "-p", exists=True, help="JSON with PDF URLs"
    ),
    out: Path = typer.Option(Path("data"), "--out", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """Generate a HuggingFace Dataset from GitHub repos and PDF papers."""
    console.print(
        Panel.fit(
            "[bold blue]Dataset Generation[/bold blue]\n"
            f"Repos: {repos_file}\n"
            f"PDFs: {pdfs_file}\n"
            f"Output: {out}",
            title="Starting",
        )
    )

    progress_tasks: dict[str, Any] = {}

    def progress_callback(task_name: str, current: int, total: int):
        if task_name not in progress_tasks:
            if task_name == "pdf_download":
                progress_tasks[task_name] = progress.add_task(
                    "[magenta]Downloading PDFs", total=total
                )
            elif task_name == "repo_processing":
                progress_tasks[task_name] = progress.add_task(
                    "[cyan]Processing repositories", total=total
                )
        progress.update(progress_tasks[task_name], completed=current)

    try:
        with _create_progress() as progress:
            result = generate_dataset_impl(repos_file, pdfs_file, out, verbose, progress_callback)

        table = Table(title="Results")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Documents", style="magenta", justify="right")
        table.add_row("PDF Papers", str(result["pdf_docs"]))
        table.add_row("Documentation", str(result["md_docs"]))
        table.add_row("[bold]Total[/bold]", f"[bold]{result['total_records']}[/bold]")

        console.print(table)
        console.print(
            f"\n[bold green]Dataset saved to[/bold green] [blue]{result['output_path']}[/blue]"
        )

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# push-to-hub
# ---------------------------------------------------------------------------


def push_to_hub(
    dataset_path: Path = typer.Argument(
        ..., help="Path to the HuggingFace dataset directory", exists=True
    ),
    repo_id: str = typer.Option(
        ..., "--repo-id", "-r", help="HuggingFace Hub repo ID (e.g., myorg/dirac-docs)"
    ),
    private: bool = typer.Option(
        False, "--private", help="Make the dataset private on HuggingFace Hub"
    ),
):
    """Push a local HuggingFace Dataset to the HuggingFace Hub."""
    console.print(
        Panel.fit(
            "[bold blue]Push to HuggingFace Hub[/bold blue]\n"
            f"Dataset: {dataset_path}\n"
            f"Repo: [green]{repo_id}[/green]\n"
            f"Private: {private}",
            title="Starting Upload",
        )
    )

    try:
        ds = DatasetDict.load_from_disk(str(dataset_path))

        for split_name, split_ds in ds.items():
            console.print(f"  {split_name}: {len(split_ds)} records")

        with console.status("Uploading to HuggingFace Hub..."):
            ds.push_to_hub(repo_id, private=private)

        console.print(
            f"\n[bold green]Dataset pushed to[/bold green] "
            f"[link=https://huggingface.co/datasets/{repo_id}]https://huggingface.co/datasets/{repo_id}[/link]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
