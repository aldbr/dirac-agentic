import os
import shutil
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.git import GitLoader
from langchain_community.document_loaders.github import GitHubIssuesLoader
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
def split_markdown_by_heading(documents, max_length=512, overlap=50):
    from llama_index.core.schema import TextNode
    import re

    nodes = []
    for doc in documents:
        text = doc.page_content if hasattr(doc, "page_content") else doc.text
        metadata = doc.metadata if hasattr(doc, "metadata") else {}

        # 按标题分段
        segments = re.split(r'\n(?=#)', text)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # 滑动窗口分块
            for i in range(0, len(segment), max_length - overlap):
                chunk = segment[i:i + max_length]
                if chunk:
                    nodes.append(TextNode(text=chunk, metadata=metadata))
    return nodes


def doc_loader(repo_url: str, branch: str = "main"):
    """
    Load text documents from a git repo, split using markdown heading-based window splitting,
    vectorize, and insert into Milvus.
    """
    repo_path = repo_url.split("/")[-1]
    git_loader = GitLoader(
        repo_path=repo_path,
        clone_url=repo_url,
        branch=branch,
        file_filter=lambda path: path.endswith((".md", ".rst")),
    )
    raw_docs = git_loader.load()
    shutil.rmtree(repo_path, ignore_errors=True)

    # Use block logic by title
    md_docs = split_markdown_by_heading(raw_docs, max_length=512, overlap=50)
    # texts = [node.text for node in nodes]
    # # 嵌入生成
    # embeddings = embed_model.get_text_embedding_batch(texts)
    # embeddings_np = np.array(embeddings, dtype=np.float32)
    # embeddings_list = embeddings_np.tolist()
    #
    # # 存入 Milvus
    # inserted_count = store_embeddings_in_milvus(embeddings_list, texts, source="doc")
    return md_docs

def pdf_loader(pdfs_path: Path):
    """
    Load PDF documents, split into chunks, embed, and store in Milvus.
    """
    pdf_loader = PyPDFDirectoryLoader(pdfs_path, recursive=True)
    pdf_docs = pdf_loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    )

    # texts = [doc.page_content for doc in pdf_docs]
    # embeddings = embed_model.get_text_embedding_batch(texts)
    # embeddings_np = np.array(embeddings, dtype=np.float32)
    # embeddings_list = embeddings_np.tolist()
    #
    # inserted_count = store_embeddings_in_milvus(embeddings_list, texts, source="paper")
    return pdf_docs

def git_metadata_loader(repo_url: str):
    """
    Load GitHub issues and PRs, embed, and store in Milvus.
    """
    repo_name = repo_url.rstrip("/").replace(".git", "").split("github.com/")[-1]
    issues_loader = GitHubIssuesLoader(
        repo=repo_name,
        access_token=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
        include_prs=True,
        state="all",
    )
    issue_and_pr_docs = issues_loader.load()
    #
    # texts = [doc.page_content for doc in issue_and_pr_docs]
    # embeddings = embed_model.get_text_embedding_batch(texts)
    # embeddings_np = np.array(embeddings, dtype=np.float32)
    # embeddings_list = embeddings_np.tolist()
    #
    # inserted_count = store_embeddings_in_milvus(embeddings_list, texts, source="issue")
    return issue_and_pr_docs
