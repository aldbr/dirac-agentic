"""Dataset generation business logic.

This module provides functions and classes for generating datasets from
GitHub repositories and PDF files. It handles document loading, processing,
and conversion to HuggingFace Dataset format.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Callable, Optional

from datasets import Dataset, DatasetDict
from pydantic import BaseModel

from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.readers.github import GithubRepositoryReader, GithubClient

from dirac_dataset.log import logger
import requests


logger = logger.getChild(__name__)


class Repo(BaseModel):
    """Model for repository URLs.

    Attributes:
        url: The GitHub repository URL.
        branch: The git branch to use, defaults to "main".
    """

    url: str
    branch: str = "main"


# -------------------------------------------------------------------------------------------------


def _load_repos_file(path: Path) -> List[Repo]:
    """Load repository configurations from a JSON file.

    Args:
        path: Path to JSON file containing repository configurations.

    Returns:
        List of Repo objects parsed from the file.
    """
    logger.debug(f"Loading repositories from {path}")
    data = json.loads(path.read_text())
    repos = [Repo(**repo) for repo in data.values()]
    logger.info(f"Loaded {len(repos)} repositories")
    return repos


def _split_markdown_by_heading(documents, max_length=512, overlap=50):
    nodes = []
    for doc in documents:
        # LlamaIndex Document objects use .text attribute
        text = doc.text if hasattr(doc, "text") else str(doc)
        metadata = doc.metadata if hasattr(doc, "metadata") else {}

        # Split by headers
        segments = re.split(r"\n(?=#)", text)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Sliding window chunking
            for i in range(0, len(segment), max_length - overlap):
                chunk = segment[i : i + max_length]
                if chunk:
                    nodes.append(TextNode(text=chunk, metadata=metadata))
    return nodes


def _load_doc(repo_url: str, branch: str = "main"):
    """Load and process documents from a GitHub repository.

    Uses LlamaIndex GitHub reader to load markdown and RST files from the
    specified repository, then splits them using heading-based chunking.

    Args:
        repo_url: GitHub repository URL.
        branch: Git branch to load from, defaults to "main".

    Returns:
        List of TextNode objects containing processed document chunks.

    Raises:
        ValueError: If the GitHub URL format is invalid.
    """
    # Extract owner and repo name from URL
    # e.g., https://github.com/owner/repo -> owner, repo
    repo_parts = repo_url.rstrip("/").rstrip(".git").split("/")[-2:]
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid GitHub URL format: {repo_url}")

    owner, repo = repo_parts

    # Create a LlamaIndex GitHub client with token
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = GithubClient(github_token=github_token)

    # Use GithubRepositoryReader to load documents
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

    # Convert Documents to TextNodes and use block logic by title
    md_docs = _split_markdown_by_heading(raw_docs, max_length=512, overlap=50)
    return md_docs


# -------------------------------------------------------------------------------------------------


def _load_pdfs_file(path: Path) -> List[str]:
    """Load PDF URLs from a JSON file.

    Args:
        path: Path to JSON file containing list of PDF URLs.

    Returns:
        List of PDF URL strings.
    """
    logger.debug(f"Loading PDF URLs from {path}")
    pdf_urls = json.loads(path.read_text())
    logger.info(f"Loaded {len(pdf_urls)} PDF URLs")
    return pdf_urls


def _download_pdf(url: str, out_dir: Path, local_name: str) -> Path | None:
    """Download a PDF from URL with validation.

    Downloads PDF file and validates it has proper content-type header
    and PDF magic number. Only saves valid PDF files.

    Args:
        url: URL to download PDF from.
        out_dir: Directory to save the PDF file.
        local_name: Local filename for the downloaded PDF.

    Returns:
        Path to downloaded file if successful, None if validation failed.
    """
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

    # Check content-type header
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
    # Check PDF magic number
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
    """Load and process PDF documents into chunks.

    Uses LlamaIndex PDFReader to load PDF files and splits them into
    manageable chunks using sentence-based splitting.

    Args:
        pdfs_path: Directory path containing PDF files.

    Returns:
        List of TextNode objects containing processed PDF chunks.
    """
    pdf_reader = PDFReader()
    text_splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

    documents = []
    pdf_files = list(pdfs_path.rglob("*.pdf"))

    for pdf_file in pdf_files:
        try:
            # Load PDF as Document objects
            pdf_docs = pdf_reader.load_data(pdf_file)

            # Split documents into nodes
            for doc in pdf_docs:
                nodes = text_splitter.get_nodes_from_documents([doc])
                documents.extend(nodes)

        except Exception as e:
            logger.warning(f"Failed to load PDF {pdf_file}: {e}")
            continue

    return documents


# -------------------------------------------------------------------------------------------------


def generate_dataset(
    repos_file: Path,
    pdfs_file: Path,
    out: Path,
    verbose: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """Generate HuggingFace dataset from GitHub repositories and PDF files.

    Downloads PDFs, loads documentation from GitHub repositories, processes
    all documents into chunks, and saves as a HuggingFace Dataset with
    separate splits for papers and documentation.

    Args:
        repos_file: Path to JSON file containing repository configurations.
        pdfs_file: Path to JSON file containing PDF URLs.
        out: Output directory for the generated dataset.
        verbose: Whether to enable debug logging.
        progress_callback: Optional callback function for progress updates.

    Returns:
        Dictionary containing generation statistics including document counts
        and output path.

    Raises:
        ValueError: If GITHUB_TOKEN environment variable is not set.
    """
    # Configure logging
    logger.setLevel("DEBUG" if verbose else "INFO")
    logger.info("Starting dataset generation")

    # Check for required GitHub token
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError(
            "GITHUB_TOKEN environment variable is required for accessing GitHub repositories. "
            "Please set it with your GitHub personal access token."
        )

    pdf_tmp = out.parent / "tmp_pdfs"
    out.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {out}")

    repos = _load_repos_file(repos_file)
    pdf_urls = _load_pdfs_file(pdfs_file)

    # Download PDFs
    logger.info(f"Starting PDF download for {len(pdf_urls)} URLs")
    if progress_callback:
        progress_callback("pdf_download", 0, len(pdf_urls))

    pdf_id = 0
    for i, url in enumerate(pdf_urls):
        try:
            logger.debug(f"Downloading PDF {i+1}/{len(pdf_urls)}: {url}")
            _download_pdf(url, pdf_tmp, local_name=f"article_{pdf_id}.pdf")
            pdf_id += 1
            logger.debug(f"Successfully downloaded PDF {pdf_id}")
        except Exception as e:
            logger.warning(f"Failed to download PDF from {url}: {e}")
            continue

        if progress_callback:
            progress_callback("pdf_download", i + 1, len(pdf_urls))

    # Load documents
    logger.info("Loading PDFs and documentation")
    pdf_docs = _load_pdfs(pdf_tmp)
    logger.info(f"Loaded {len(pdf_docs)} PDF documents")

    md_docs = []

    logger.info(f"Processing {len(repos)} repositories")
    if progress_callback:
        progress_callback("repo_processing", 0, len(repos))

    for i, repo in enumerate(repos):
        try:
            logger.debug(f"Processing repository {i+1}/{len(repos)}: {repo.url}")
            repo_md_docs = _load_doc(repo.url, branch=repo.branch)

            md_docs.extend(repo_md_docs)

            logger.debug(f"Repository {repo.url}: {len(repo_md_docs)} docs")
        except Exception as e:
            logger.error(f"Failed to load docs for {repo.url}: {e}")
            continue

        if progress_callback:
            progress_callback("repo_processing", i + 1, len(repos))

    # Cleanup
    shutil.rmtree(pdf_tmp, ignore_errors=True)
    logger.debug("Cleaned up temporary PDF directory")

    # Log results
    logger.info(
        f"Document loading complete - PDF docs: {len(pdf_docs)}, "
        f"Documentation chunks: {len(md_docs)}"
    )

    # Create HuggingFace Dataset
    logger.info("Creating HuggingFace Dataset")

    records = []
    # All docs are now LlamaIndex TextNode objects with .text attribute
    for doc in pdf_docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        records.append({"text": doc.text, **metadata, "source": "paper"})
    for doc in md_docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        records.append({"text": doc.text, **metadata, "source": "doc"})
    logger.debug(f"Created {len(records)} total records")

    ds = Dataset.from_list(records)
    ds_splits = DatasetDict(
        {
            "papers": ds.filter(lambda x: x["source"] == "paper"),
            "docs": ds.filter(lambda x: x["source"] == "doc"),
        }
    )
    ds_splits.save_to_disk(out)

    logger.info(f"Dataset saved successfully to {out}")
    return {
        "pdf_docs": len(pdf_docs),
        "md_docs": len(md_docs),
        "total_records": len(records),
        "output_path": str(out),
    }
