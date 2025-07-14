import os
import shutil
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.git import GitLoader
from langchain_community.document_loaders.github import GitHubIssuesLoader
from llama_index.embeddings.openai import OpenAIEmbedding
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client import IndexParams
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

def store_embeddings_in_milvus(
    embeddings: list[list[float]],
    texts: list[str],
    source: str = "generic",
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db"
):
    """
    Store embeddings and texts into Milvus with indexing for retrieval.

    Parameters:
        embeddings: List[List[float]] - The embeddings to store.
        texts: List[str] - Corresponding texts.
        source: str - Source label ("paper", "doc", "issue", etc.) for tracking.
        collection_name: str - Milvus collection name.
        db_path: str - Local Milvus DB storage path.
    """

    # Initialize Milvus client
    client = MilvusClient(uri="./milvus_demo.db")

    # Create collection schema if not exists
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
    ]
    schema = CollectionSchema(fields, description="Document embeddings with raw text and source info")

    if not client.has_collection(collection_name):
        client.create_collection(collection_name=collection_name, schema=schema)

    # Prepare data for insertion
    insert_data = [
        {"embedding": emb, "text": txt, "source": source}
        for emb, txt in zip(embeddings, texts)
    ]

    # Insert and flush
    client.insert(collection_name=collection_name, data=insert_data)
    client.flush(collection_name=collection_name)

    # Create index if not already created
    index_params = IndexParams()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        index_name="ivf_index",
        metric_type="COSINE",
        nlist=128
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.flush(collection_name=collection_name)
    return len(insert_data)


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

    # 使用按标题分块逻辑
    nodes = split_markdown_by_heading(raw_docs, max_length=512, overlap=50)
    texts = [node.text for node in nodes]
    # 嵌入生成
    embeddings = embed_model.get_text_embedding_batch(texts)
    embeddings_np = np.array(embeddings, dtype=np.float32)
    embeddings_list = embeddings_np.tolist()

    # 存入 Milvus
    inserted_count = store_embeddings_in_milvus(embeddings_list, texts, source="doc")
    return inserted_count

def pdf_loader(pdfs_path: Path):
    """
    Load PDF documents, split into chunks, embed, and store in Milvus.
    """
    pdf_loader = PyPDFDirectoryLoader(pdfs_path, recursive=True)
    pdf_docs = pdf_loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    )

    texts = [doc.page_content for doc in pdf_docs]
    embeddings = embed_model.get_text_embedding_batch(texts)
    embeddings_np = np.array(embeddings, dtype=np.float32)
    embeddings_list = embeddings_np.tolist()

    inserted_count = store_embeddings_in_milvus(embeddings_list, texts, source="paper")
    return inserted_count

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

    texts = [doc.page_content for doc in issue_and_pr_docs]
    embeddings = embed_model.get_text_embedding_batch(texts)
    embeddings_np = np.array(embeddings, dtype=np.float32)
    embeddings_list = embeddings_np.tolist()

    inserted_count = store_embeddings_in_milvus(embeddings_list, texts, source="issue")
    return inserted_count
