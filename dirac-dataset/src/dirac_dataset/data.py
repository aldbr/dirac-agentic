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


def _load_urls(path: Path) -> List[str]:
    """Accept *.json (list) or *.txt (one URL per line)."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        BarColumn(bar_width=None),
        TextColumn("[progress.description]{task.description}"),
        TimeRemainingColumn(compact=True),
    )


# -----------------------------------------------------------------------------
# single command
# -----------------------------------------------------------------------------


@app.command("load-data")
def load_data(
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

    repo_urls = _load_urls(repos_file)
    pdf_urls = _load_urls(pdfs_file)

    # -------------------------------------------------------------------------
    # download PDFs
    # -------------------------------------------------------------------------
    with _progress() as p:
        pdf_task = p.add_task("[magenta]Downloading PDFs", total=len(pdf_urls))
        for url in pdf_urls:
            try:
                downloader.download_pdf(url, pdf_tmp)
            except Exception as e:
                console.print(f"[red]Error downloading or copying PDF:[/]\n{e}")
                continue
            p.advance(pdf_task)

    # -------------------------------------------------------------------------
    # load PDFs, documentation, and issues/PRs using loader.py
    # -------------------------------------------------------------------------
    console.print("\n[bold blue]Loading PDFs and documentation...[/]")
    pdf_docs = loader.pdf_loader(pdf_tmp)
    md_docs = []
    issue_and_pr_docs = []
    with _progress() as p:
        doc_task = p.add_task("[cyan]Loading docs/issues/PRs", total=len(repo_urls))
        for url in repo_urls:
            try:
                md_docs.extend(loader.doc_loader(url))
                issue_and_pr_docs.extend(loader.git_metadata_loader(url))
            except Exception as e:
                console.print(f"[red]Error loading docs/issues/PRs for {url}:[/]\n{e}")
                continue
            p.advance(doc_task)

    # -------------------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------------------
    shutil.rmtree(pdf_tmp, ignore_errors=True)

    # -------------------------------------------------------------------------
    # display result
    # -------------------------------------------------------------------------
    console.print("\n[bold green]✓ All done! Loaded documents:[/]")
    console.print(f"[yellow]PDF docs:[/] {len(pdf_docs)}")
    console.print(f"[yellow]Documentation chunks:[/] {len(md_docs)}")
    console.print(f"[yellow]Issues/PRs:[/] {len(issue_and_pr_docs)}")
    console.print("[bold blue]Sample output (first 1 of each):[/]")
    if pdf_docs:
        console.print(f"[green]PDF sample:[/] {pdf_docs[0]}")
    if md_docs:
        console.print(f"[green]Doc sample:[/] {md_docs[0]}")
    if issue_and_pr_docs:
        console.print(f"[green]Issue/PR sample:[/] {issue_and_pr_docs[0]}")


if __name__ == "__main__":
    app()
