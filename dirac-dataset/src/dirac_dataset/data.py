#!/usr/bin/env python
"""
Harvest documentation & papers and convert them into plain-text files for RAG.

Examples
--------
gen-txt-data --repos-file repos.json --pdfs-file pdfs.txt
gen-txt-data -r repos.txt -p pdfs.txt -o data/text --keep-tmp
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List

from datasets import Dataset, DatasetDict
from pydantic import BaseModel
import typer 
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

from dirac_dataset import downloader, loader
from dirac_dataset.log import logger

console = Console()
app = typer.Typer(add_completion=False)

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


class Repo(BaseModel):
    """Model for repository URLs."""

    url: str
    branch: str = "main"


def _load_repos(path: Path) -> List[Repo]:
    """Accept *.json (dict or list) or *.txt (one URL per line). Returns a list of Repo objects."""
    data = json.loads(path.read_text())
    return [Repo(**repo) for repo in data.values()]


def _load_pdfs(path: Path) -> List[str]:
    """Accept *.json (list of URLs) or *.txt (one URL per line). Returns a list of PDF URLs."""
    return json.loads(path.read_text())


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        BarColumn(bar_width=None),
        TextColumn("[progress.description]{task.description}"),
        TimeRemainingColumn(compact=True),
    )


# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------


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
    """
    Download PDFs, loads docs/issues/PRs/PDFs into a Vector DB.
    """

    # -------------------------------------------------------------------------
    # configure logging
    # -------------------------------------------------------------------------
    logger.setLevel("DEBUG" if verbose else "INFO")

    # -------------------------------------------------------------------------
    # check GITHUB_PERSONAL_ACCESS_TOKEN
    # -------------------------------------------------------------------------
    if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        console.print(
            "[red]GITHUB_PERSONAL_ACCESS_TOKEN environment variable is not set! Exiting."
        )
        raise typer.Exit(1)

    pdf_tmp = out.parent / "tmp_pdfs"
    out.mkdir(parents=True, exist_ok=True)

    repos = _load_repos(repos_file)
    pdf_urls = _load_pdfs(pdfs_file)

    # -------------------------------------------------------------------------
    # download PDFs
    # -------------------------------------------------------------------------
    with _progress() as p:
        pdf_task = p.add_task("[magenta]Downloading PDFs", total=len(pdf_urls))
        pdf_id = 0
        for url in pdf_urls:
            try:
                downloader.download_pdf(
                    url, pdf_tmp, local_name=f"article_{pdf_id}.pdf"
                )
                pdf_id += 1
            except Exception as e:
                console.print(f"[red]Error downloading or copying PDF:[/]\n{e}")
                continue
            p.advance(pdf_task)

    # -------------------------------------------------------------------------
    # Load PDFs, documentation, and issues/PRs using loader.py
    # -------------------------------------------------------------------------
    console.print("\n[bold blue]Loading PDFs and documentation, inserting into Milvus...[/]")

    # PDFs
    pdf_insert_count = loader.pdf_loader(pdf_tmp)

    # Repos
    doc_insert_total = 0
    issue_insert_total = 0

    with _progress() as p:
        doc_task = p.add_task("[cyan]Loading docs/issues/PRs", total=len(repos))
        for repo in repos:
            try:
                doc_insert_total += loader.doc_loader(repo.url, branch=repo.branch)
                issue_insert_total += loader.git_metadata_loader(repo.url)
            except Exception as e:
                console.print(f"[red]Error loading for {repo.url}:[/]\n{e}")
                continue
            p.advance(doc_task)

    # Cleanup
    shutil.rmtree(pdf_tmp, ignore_errors=True)

    # Display inserted stats
    console.print("\n[bold green]✓ All done! Inserted into Milvus:[/]")
    console.print(f"[yellow]PDF chunks inserted:[/] {pdf_insert_count}")
    console.print(f"[yellow]Documentation chunks inserted:[/] {doc_insert_total}")
    console.print(f"[yellow]Issues/PRs inserted:[/] {issue_insert_total}")

@app.command("load-dataset")
def load_dataset(
    dataset_path: Path = typer.Argument(
        ..., help="Path to the saved HuggingFace dataset directory"
    ),
):
    """
    Load and display basic info about a previously saved HuggingFace dataset.
    """
    console.print(f"[bold blue]Loading dataset from {dataset_path}...[/]")
    ds_splits = DatasetDict.load_from_disk(dataset_path)
    for split, ds in ds_splits.items():
        console.print(f"[green]{split}[/]: {len(ds)} records")
        # Show a sample
        if len(ds) > 0:
            console.print(f"[yellow]Sample from {split}:[/]")
            console.print(ds[0])


if __name__ == "__main__":
    app()
