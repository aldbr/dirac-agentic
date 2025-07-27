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
import numpy as np

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
    # load PDFs, documentation, and issues/PRs using loader.py
    # -------------------------------------------------------------------------
    console.print("\n[bold blue]Loading PDFs and documentation...[/]")
    pdf_docs = loader.pdf_loader(pdf_tmp)
    md_docs = []
    issue_and_pr_docs = []
    with _progress() as p:
        doc_task = p.add_task("[cyan]Loading docs/issues/PRs", total=len(repos))
        for repo in repos:
            try:
                md_docs.extend(loader.doc_loader(repo.url, branch=repo.branch))
                issue_and_pr_docs.extend(loader.git_metadata_loader(repo.url))
            except Exception as e:
                console.print(
                    f"[red]Error loading docs/issues/PRs for {repo.url}:[/]\n{e}"
                )
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

    # -------------------------------------------------------------------------
    # serialize to a local HF Dataset
    # -------------------------------------------------------------------------
    console.print("\n[bold blue]Saving to HuggingFace Dataset…[/]")

    records = []
    for doc in pdf_docs:
        records.append({"text": doc.page_content, **doc.metadata, "source": "paper"})
    for doc in md_docs:
        records.append({"text": doc.page_content, **doc.metadata, "source": "doc"})
    for doc in issue_and_pr_docs:
        records.append({"text": doc.page_content, **doc.metadata, "source": "issue"})

    ds = Dataset.from_list(records)
    ds_splits = DatasetDict(
        {
            "papers": ds.filter(lambda x: x["source"] == "paper"),
            "docs": ds.filter(lambda x: x["source"] == "doc"),
            "issues": ds.filter(lambda x: x["source"] == "issue"),
        }
    )
    ds_splits.save_to_disk(out)

    console.print(f"[bold green]✓ Dataset saved to {out}[/]")


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


@app.command("load-db")
def load_db(
    repos_file: Path = typer.Option(
        ..., "--repos-file", "-r", exists=True, help="TXT/JSON with repo URLs"
    ),
    pdfs_file: Path = typer.Option(
        ..., "--pdfs-file", "-p", exists=True, help="TXT/JSON with PDF URLs"
    ),
    db_path: Path = typer.Option(
        Path("./milvus_demo.db"), "--db-path", "-d", help="Path to Milvus database"
    ),
    collection_name: str = typer.Option(
        "doc_embeddings", "--collection", "-c", help="Milvus collection name"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Rich logging at DEBUG level"
    ),
):
    """
    Load documents from repos and PDFs, generate embeddings, and store in Milvus database.
    """
    from dirac_dataset.milvus import store_embeddings_in_milvus
    
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

    pdf_tmp = Path("tmp_pdfs")
    pdf_tmp.mkdir(parents=True, exist_ok=True)

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
    # load PDFs, documentation, and issues/PRs using loader.py
    # -------------------------------------------------------------------------
    console.print("\n[bold blue]Loading PDFs and documentation...[/]")
    pdf_docs = loader.pdf_loader(pdf_tmp)
    md_docs = []
    issue_and_pr_docs = []
    with _progress() as p:
        doc_task = p.add_task("[cyan]Loading docs/issues/PRs", total=len(repos))
        for repo in repos:
            try:
                md_docs.extend(loader.doc_loader(repo.url, branch=repo.branch))
                issue_and_pr_docs.extend(loader.git_metadata_loader(repo.url))
            except Exception as e:
                console.print(
                    f"[red]Error loading docs/issues/PRs for {repo.url}:[/]\n{e}"
                )
                continue
            p.advance(doc_task)

    # -------------------------------------------------------------------------
    # Generate embeddings and store in Milvus
    # -------------------------------------------------------------------------
    console.print("\n[bold blue]Generating embeddings and storing in Milvus...[/]")
    
    # Process PDF documents
    with _progress() as p:
        pdf_embed_task = p.add_task("[magenta]Processing PDF documents", total=len(pdf_docs))
        texts = [doc.page_content for doc in pdf_docs]
        if texts:
            embeddings = loader.embed_model.get_text_embedding_batch(texts)
            embeddings_np = np.array(embeddings, dtype=np.float32)
            embeddings_list = embeddings_np.tolist()
            inserted_count = store_embeddings_in_milvus(
                embeddings_list, texts, source="paper", 
                collection_name=collection_name, db_path=str(db_path)
            )
            console.print(f"[green]Stored {inserted_count} PDF document embeddings[/]")
        p.update(pdf_embed_task, completed=len(pdf_docs))
    
    # Process markdown documents
    with _progress() as p:
        md_embed_task = p.add_task("[cyan]Processing markdown documents", total=len(md_docs))
        texts = [node.text for node in md_docs]
        if texts:
            embeddings = loader.embed_model.get_text_embedding_batch(texts)
            embeddings_np = np.array(embeddings, dtype=np.float32)
            embeddings_list = embeddings_np.tolist()
            inserted_count = store_embeddings_in_milvus(
                embeddings_list, texts, source="doc", 
                collection_name=collection_name, db_path=str(db_path)
            )
            console.print(f"[green]Stored {inserted_count} markdown document embeddings[/]")
        p.update(md_embed_task, completed=len(md_docs))
    
    # Process issues and PRs
    with _progress() as p:
        issue_embed_task = p.add_task("[yellow]Processing issues and PRs", total=len(issue_and_pr_docs))
        texts = [doc.page_content for doc in issue_and_pr_docs]
        if texts:
            embeddings = loader.embed_model.get_text_embedding_batch(texts)
            embeddings_np = np.array(embeddings, dtype=np.float32)
            embeddings_list = embeddings_np.tolist()
            inserted_count = store_embeddings_in_milvus(
                embeddings_list, texts, source="issue", 
                collection_name=collection_name, db_path=str(db_path)
            )
            console.print(f"[green]Stored {inserted_count} issue/PR embeddings[/]")
        p.update(issue_embed_task, completed=len(issue_and_pr_docs))
    
    # -------------------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------------------
    shutil.rmtree(pdf_tmp, ignore_errors=True)
    
    # -------------------------------------------------------------------------
    # display result
    # -------------------------------------------------------------------------
    console.print("\n[bold green]✓ All done! Documents stored in Milvus database:[/]")
    console.print(f"[yellow]PDF docs:[/] {len(pdf_docs)}")
    console.print(f"[yellow]Documentation chunks:[/] {len(md_docs)}")
    console.print(f"[yellow]Issues/PRs:[/] {len(issue_and_pr_docs)}")
    console.print(f"[bold green]✓ Database saved to {db_path}[/]")


if __name__ == "__main__":
    app()
