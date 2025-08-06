"""
Milvus database operations - moved from original milvus.py
"""

from typing import Optional
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client import IndexParams


def store_embeddings_in_milvus(
    embeddings: list[list[float]],
    texts: list[str],
    source: str = "generic",
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db",
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
    client = MilvusClient(db_path)

    # Handle empty list case
    if not embeddings or len(embeddings) == 0:
        return 0

    # Create collection schema if not exists
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
    ]
    schema = CollectionSchema(
        fields, description="Document embeddings with raw text and source info"
    )

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

    # Create index (AUTOINDEX)
    index_params = IndexParams()
    index_params.add_index(
        field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    # Load collection for querying
    client.load_collection(collection_name=collection_name)

    return len(insert_data)


def search_embeddings_in_milvus(
    query_embedding: list[float],
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db",
    limit: int = 5,
    output_fields: Optional[list[str]] = None,
) -> list[dict]:
    """
    Search for similar embeddings in Milvus.

    Parameters:
        query_embedding: List[float] - The query embedding vector.
        collection_name: str - Milvus collection name.
        db_path: str - Local Milvus DB storage path.
        limit: int - Number of results to return.
        output_fields: List[str] - Fields to return in results.

    Returns:
        List of dictionaries containing search results.
    """
    if output_fields is None:
        output_fields = ["text", "source"]

    # Initialize Milvus client
    client = MilvusClient(db_path)

    if not client.has_collection(collection_name):
        return []

    # Perform search
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=limit,
        search_params={"metric_type": "COSINE"},
        output_fields=output_fields,
    )

    return results[0] if results else []
