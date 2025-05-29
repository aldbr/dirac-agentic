import os
import shutil
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.git import GitLoader
from langchain_community.document_loaders.github import GitHubIssuesLoader


def pdf_loader(pdfs_path: Path):
    """
    Load PDF documents from a directory and split them into chunks.
    """
    pdf_loader = PyPDFDirectoryLoader(
        pdfs_path,
        recursive=True,
    )
    pdf_docs = pdf_loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    )
    return pdf_docs


def doc_loader(repo_url: str, branch: str = "main"):
    """
    Load text documents from a git repo and split them into chunks. Cleans up repo after loading.
    """
    repo_path = repo_url.split("/")[-1]
    git_loader = GitLoader(
        repo_path=repo_path,
        clone_url=repo_url,
        branch=branch,
        file_filter=lambda path: path.endswith((".md", ".rst")),
    )
    raw_docs = git_loader.load()
    # Clean up the cloned repo directory after loading
    shutil.rmtree(repo_path, ignore_errors=True)
    md_docs = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(raw_docs)

    return md_docs


def git_metadata_loader(repo_url: str):
    """
    Load issues and PRs from a git repo and split them into chunks.
    """
    # Extract owner/repo from repo_url
    repo_name = repo_url.rstrip("/").replace(".git", "").split("github.com/")[-1]
    issues_loader = GitHubIssuesLoader(
        repo=repo_name,
        access_token=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
        include_prs=True,
        state="all",
    )
    issue_and_pr_docs = issues_loader.load()
    return issue_and_pr_docs
