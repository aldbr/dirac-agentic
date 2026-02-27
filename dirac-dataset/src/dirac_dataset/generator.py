"""Dataset generation from GitHub repositories and PDF files.

Loads documentation from GitHub repos and PDF papers, chunks them,
and saves as a HuggingFace Dataset.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.readers.file import PDFReader
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from pydantic import BaseModel

from dirac_dataset.log import logger

logger = logger.getChild(__name__)


class Repo(BaseModel):
    """GitHub repository reference."""

    url: str
    branch: str = "main"


def _load_repos_file(path: Path) -> list[Repo]:
    """Load repository configurations from a JSON file."""
    logger.debug(f"Loading repositories from {path}")
    data = json.loads(path.read_text())
    repos = [Repo(**repo) for repo in data.values()]
    logger.info(f"Loaded {len(repos)} repositories")
    return repos


def _split_markdown_by_heading(documents, max_length=512, overlap=50):
    """Split LlamaIndex documents by markdown headings with sliding window."""
    nodes = []
    for doc in documents:
        text = doc.text if hasattr(doc, "text") else str(doc)
        metadata = doc.metadata if hasattr(doc, "metadata") else {}

        segments = re.split(r"\n(?=#)", text)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            for i in range(0, len(segment), max_length - overlap):
                chunk = segment[i : i + max_length]
                if chunk:
                    nodes.append(TextNode(text=chunk, metadata=metadata))
    return nodes


def _load_doc(repo_url: str, branch: str = "main"):
    """Load and chunk markdown/RST files from a GitHub repository."""
    repo_parts = repo_url.rstrip("/").rstrip(".git").split("/")[-2:]
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid GitHub URL format: {repo_url}")

    owner, repo = repo_parts

    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = GithubClient(github_token=github_token)

    github_reader = GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        use_parser=False,
        verbose=False,
        filter_directories=None,
        filter_file_extensions=(
            [".md", ".rst"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    )

    raw_docs = github_reader.load_data(branch=branch)
    return _split_markdown_by_heading(raw_docs, max_length=512, overlap=50)


def _load_pdfs_file(path: Path) -> list[str]:
    """Load PDF URLs from a JSON file."""
    logger.debug(f"Loading PDF URLs from {path}")
    pdf_urls = json.loads(path.read_text())
    logger.info(f"Loaded {len(pdf_urls)} PDF URLs")
    return pdf_urls


def _download_pdf(url: str, out_dir: Path, local_name: str) -> Path | None:
    """Download a PDF from URL with content-type and magic number validation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / local_name

    logger.debug("GET %s", url)

    r = requests.get(
        url,
        stream=True,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        },
    )
    r.raise_for_status()

    content_type = r.headers.get("content-type", "").lower()
    if "pdf" not in content_type:
        logger.warning(
            "URL %s does not appear to be a PDF (content-type: %s), skipping.",
            url,
            content_type,
        )
        return None

    with open(file, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)

    try:
        with open(file, "rb") as fh:
            magic = fh.read(4)
        if magic != b"%PDF":
            logger.warning(
                "File %s does not start with %%PDF magic number, deleting. (URL: %s)",
                file,
                url,
            )
            file.unlink(missing_ok=True)
            return None
    except Exception as e:
        logger.warning("Error checking PDF magic number for %s: %s", file, e)
        file.unlink(missing_ok=True)
        return None

    logger.info("Saved %s", file)
    return file


def _load_pdfs(pdfs_path: Path):
    """Load and chunk PDF documents using LlamaIndex."""
    pdf_reader = PDFReader()
    text_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

    documents = []
    pdf_files = list(pdfs_path.rglob("*.pdf"))

    for pdf_file in pdf_files:
        try:
            pdf_docs = pdf_reader.load_data(pdf_file)
            for doc in pdf_docs:
                nodes = text_splitter.get_nodes_from_documents([doc])
                documents.extend(nodes)
        except Exception as e:
            logger.warning(f"Failed to load PDF {pdf_file}: {e}")
            continue

    return documents


def generate_dataset(
    repos_file: Path,
    pdfs_file: Path,
    out: Path,
    verbose: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Generate a HuggingFace Dataset from GitHub repos and PDF papers.

    Downloads PDFs, loads documentation from GitHub repositories, chunks
    everything, and saves as a DatasetDict with "papers" and "docs" splits.

    Args:
        repos_file: JSON file with repository configurations.
        pdfs_file: JSON file with PDF URLs.
        out: Output directory for the HuggingFace Dataset.
        verbose: Enable debug logging.
        progress_callback: Optional (task_name, current, total) callback.

    Returns:
        Dict with pdf_docs, md_docs, total_records, output_path.

    Raises:
        ValueError: If GITHUB_TOKEN is not set.
    """
    logger.setLevel("DEBUG" if verbose else "INFO")
    logger.info("Starting dataset generation")

    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError(
            "GITHUB_TOKEN environment variable is required for accessing GitHub repositories. "
            "Please set it with your GitHub personal access token."
        )

    pdf_tmp = out.parent / "tmp_pdfs"
    out.mkdir(parents=True, exist_ok=True)

    repos = _load_repos_file(repos_file)
    pdf_urls = _load_pdfs_file(pdfs_file)

    # Download PDFs
    logger.info(f"Starting PDF download for {len(pdf_urls)} URLs")
    if progress_callback:
        progress_callback("pdf_download", 0, len(pdf_urls))

    pdf_id = 0
    for i, url in enumerate(pdf_urls):
        try:
            _download_pdf(url, pdf_tmp, local_name=f"article_{pdf_id}.pdf")
            pdf_id += 1
        except Exception as e:
            logger.warning(f"Failed to download PDF from {url}: {e}")
            continue

        if progress_callback:
            progress_callback("pdf_download", i + 1, len(pdf_urls))

    # Load and chunk documents
    pdf_docs = _load_pdfs(pdf_tmp)
    logger.info(f"Loaded {len(pdf_docs)} PDF chunks")

    md_docs = []
    logger.info(f"Processing {len(repos)} repositories")
    if progress_callback:
        progress_callback("repo_processing", 0, len(repos))

    for i, repo in enumerate(repos):
        try:
            repo_md_docs = _load_doc(repo.url, branch=repo.branch)
            md_docs.extend(repo_md_docs)
        except Exception as e:
            logger.error(f"Failed to load docs for {repo.url}: {e}")
            continue

        if progress_callback:
            progress_callback("repo_processing", i + 1, len(repos))

    # Cleanup temp PDFs
    shutil.rmtree(pdf_tmp, ignore_errors=True)

    logger.info(f"Document loading complete - PDFs: {len(pdf_docs)}, Docs: {len(md_docs)}")

    # Build HuggingFace Dataset
    records = []
    for doc in pdf_docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        records.append({"text": doc.text, **metadata, "source": "paper"})
    for doc in md_docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        records.append({"text": doc.text, **metadata, "source": "doc"})

    ds = Dataset.from_list(records)
    ds_splits = DatasetDict(
        {
            "papers": ds.filter(lambda x: x["source"] == "paper"),
            "docs": ds.filter(lambda x: x["source"] == "doc"),
        }
    )
    ds_splits.save_to_disk(out)

    logger.info(f"Dataset saved to {out}")
    return {
        "pdf_docs": len(pdf_docs),
        "md_docs": len(md_docs),
        "total_records": len(records),
        "output_path": str(out),
    }
