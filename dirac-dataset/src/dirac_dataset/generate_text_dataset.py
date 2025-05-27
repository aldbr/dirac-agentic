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
import shutil
from pathlib import Path
from typing import Iterable, List

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

from . import converter, downloader
from .log import logger

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


@app.command("run")
def run(
    repos_file: Path = typer.Option(
        ..., "--repos-file", "-r", exists=True, help="TXT/JSON with repo URLs"
    ),
    pdfs_file: Path = typer.Option(
        ..., "--pdfs-file", "-p", exists=True, help="TXT/JSON with PDF URLs"
    ),
    out: Path = typer.Option(
        Path("data/text"), "--out", "-o", help="Destination for .txt payloads"
    ),
    keep_tmp: bool = typer.Option(
        False,
        "--keep-tmp",
        help="Do not delete temporary cloned repos / downloaded PDFs",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Rich logging at DEBUG level"
    ),
):
    """
    Clones repos, converts RST/MD, downloads PDFs, extracts text.
    """

    # -------------------------------------------------------------------------
    # configure logging
    # -------------------------------------------------------------------------
    logger.setLevel("DEBUG" if verbose else "INFO")

    repo_tmp = out.parent / "tmp_repos"
    pdf_tmp = out.parent / "tmp_pdfs"
    out.mkdir(parents=True, exist_ok=True)

    repo_urls = _load_urls(repos_file)
    pdf_urls = _load_urls(pdfs_file)

    # -------------------------------------------------------------------------
    # clone & convert documentation
    # -------------------------------------------------------------------------
    with _progress() as p:
        clone_task = p.add_task("[cyan]Cloning + converting docs", total=len(repo_urls))
        for url in repo_urls:
            repo_path = downloader.clone_repo(url, repo_tmp)
            rst_md_files: Iterable[Path] = list(repo_path.rglob("*.rst")) + list(
                repo_path.rglob("*.md")
            )
            for doc in rst_md_files:
                converter.rst_or_md_to_txt(doc, out)
            p.advance(clone_task)

    # -------------------------------------------------------------------------
    # download & convert PDFs
    # -------------------------------------------------------------------------
    with _progress() as p:
        pdf_task = p.add_task(
            "[magenta]Downloading + extracting PDFs", total=len(pdf_urls)
        )
        for url in pdf_urls:
            try:
                pdf_path = downloader.download_pdf(url, pdf_tmp)
                converter.pdf_to_txt(pdf_path, out)
            except Exception as e:
                console.print(f"[red]Error downloading or converting PDF:[/]\n{e}")
                continue
            p.advance(pdf_task)

    # -------------------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------------------
    if keep_tmp:
        console.print(f"[yellow]Keeping temporary dirs[/]: {repo_tmp}  {pdf_tmp}")
    else:
        shutil.rmtree(repo_tmp, ignore_errors=True)
        shutil.rmtree(pdf_tmp, ignore_errors=True)

    console.print(f"\n[bold green]✓ All done! Text files in:[/] {out}")


if __name__ == "__main__":
    app()
