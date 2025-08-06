#!/usr/bin/env python
"""
CLI for database loading - Presentation Layer
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

from dirac_dataset.business_logic.db_loader import load_dataset_to_database

console = Console()
app = typer.Typer(add_completion=False)


def _create_progress() -> Progress:
    """Create a Rich progress bar with custom styling"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


@app.command("load-db")
def load_db(
    dataset_path: Path = typer.Argument(
        ..., help="Path to the HuggingFace dataset directory", exists=True
    ),
    db_type: str = typer.Option(
        "milvus", "--db-type", help="Database type: milvus, chroma, huggingface"
    ),
    db_path: Path = typer.Option(
        Path("./dirac_vector.db"), "--db-path", "-d", help="Path to vector database"
    ),
    collection_name: str = typer.Option(
        "doc_embeddings", "--collection", "-c", help="Collection name"
    ),
    embedding_model: str = typer.Option(
        "BAAI/bge-small-en-v1.5",
        "--embedding",
        help="Embedding model (e.g., BAAI/bge-small-en-v1.5, text-embedding-3-small)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Rich logging at DEBUG level"
    ),
):
    """
    Load HuggingFace dataset, generate embeddings, and store in vector database.
    """
    console.print(
        Panel.fit(
            f"[bold blue]Vector Database Loading[/bold blue]\n"
            f"📂 Dataset: {dataset_path}\n"
            f"🧠 Embedding: [green]{embedding_model}[/green]\n"
            f"🗄️ Database: [yellow]{db_type}[/yellow]\n"
            f"💾 Path: {db_path}",
            title="🚀 Starting Database Loading",
        )
    )

    progress_tasks: Dict[str, Any] = {}

    def progress_callback(task_name: str, current: int, total: int):
        if task_name not in progress_tasks:
            if task_name == "dataset_loading":
                progress_tasks[task_name] = progress.add_task(
                    "[cyan]📂 Loading HuggingFace dataset", total=total
                )
            elif task_name == "papers_embedding":
                progress_tasks[task_name] = progress.add_task(
                    "[blue]🧠 Embedding PDF documents", total=total
                )
            elif task_name == "docs_embedding":
                progress_tasks[task_name] = progress.add_task(
                    "[green]🧠 Embedding markdown docs", total=total
                )
            elif task_name == "issues_embedding":
                progress_tasks[task_name] = progress.add_task(
                    "[yellow]🧠 Embedding issues/PRs", total=total
                )

        progress.update(progress_tasks[task_name], completed=current)

    try:
        with _create_progress() as progress:
            result = load_dataset_to_database(
                dataset_path=dataset_path,
                db_type=db_type,
                db_path=db_path,
                collection_name=collection_name,
                embedding_model=embedding_model,
                verbose=verbose,
                progress_callback=progress_callback,
            )

        # Display dataset overview first
        dataset_table = Table(title="📊 Dataset Overview")
        dataset_table.add_column("Split", style="cyan", no_wrap=True)
        dataset_table.add_column("Records", style="magenta", justify="right")
        dataset_table.add_column("Sample Text", style="green", max_width=50)

        for split, info in result["dataset_info"].items():
            sample_text = ""
            if info["sample"]:
                sample_text = (
                    info["sample"].get("text", "")[:100] + "..."
                    if len(info["sample"].get("text", "")) > 100
                    else info["sample"].get("text", "")
                )

            dataset_table.add_row(
                f"📑 {split.title()}", str(info["count"]), sample_text
            )

        console.print(dataset_table)

        # Display embedding results
        embed_table = Table(title="🧠 Embedding Results")
        embed_table.add_column("Source", style="cyan", no_wrap=True)
        embed_table.add_column("Embeddings", style="green", justify="right")

        embed_table.add_row("📄 PDF Papers", str(result["pdf_embeddings"]))
        embed_table.add_row("📝 Documentation", str(result["md_embeddings"]))
        embed_table.add_row(
            "[bold]Total[/bold]", f"[bold]{result['total_embeddings']}[/bold]"
        )

        console.print(embed_table)

        # Display configuration summary
        config_table = Table(title="⚙️ Configuration Used")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("🧠 Embedding Model", result["embedding_model"])
        config_table.add_row("🗄️ Database Type", result["db_type"])
        config_table.add_row("📂 Database Path", result["db_path"])

        console.print(config_table)
        console.print(
            "\n[bold green]✅ Database loading completed successfully![/bold green]"
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
