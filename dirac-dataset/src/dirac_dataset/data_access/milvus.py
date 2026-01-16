"""Milvus database operations.

This module provides functions for storing and searching embeddings in Milvus
vector database. It handles collection creation, indexing, and retrieval operations.
"""

from typing import Optional, List, Dict, Any, Tuple
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client import IndexParams


def store_embeddings_in_milvus(
    embeddings: list[list[float]],
    texts: list[str],
    source: str = "generic",
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db",
):
    """Store embeddings and texts into Milvus with indexing for retrieval.

    Creates a Milvus collection if it doesn't exist, inserts the embeddings
    with their corresponding texts and source labels, and creates an index
    for efficient similarity search.

    Args:
        embeddings: The embedding vectors to store.
        texts: Corresponding text documents for the embeddings.
        source: Source label for tracking document origin.
        collection_name: Name of the Milvus collection.
        db_path: Path to the local Milvus database file.

    Returns:
        Number of successfully inserted embedding records.
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
    """Search for embeddings similar to the query vector in Milvus.

    Performs cosine similarity search in the specified Milvus collection
    and returns the most similar embeddings with their metadata.

    Args:
        query_embedding: The query embedding vector to search for.
        collection_name: Name of the Milvus collection to search.
        db_path: Path to the local Milvus database file.
        limit: Maximum number of results to return.
        output_fields: List of fields to include in results, defaults to ["text", "source"].

    Returns:
        List of dictionaries containing search results with similarity scores and metadata.
        Returns empty list if collection doesn't exist.
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


# Document class for consistency
class MilvusDocument:
    """Simple document representation for Milvus results."""

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata


def search_embeddings_with_scores_in_milvus(
    query_embedding: List[float],
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db",
    limit: int = 5,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Tuple[MilvusDocument, float]]:
    """Search for embeddings with similarity scores and optional filtering.

    Args:
        query_embedding: The query embedding vector to search for.
        collection_name: Name of the Milvus collection to search.
        db_path: Path to the local Milvus database file.
        limit: Maximum number of results to return.
        filter: Optional filter criteria (e.g., {"source": "papers"}).

    Returns:
        List of tuples (MilvusDocument, score) sorted by similarity.
    """
    client = MilvusClient(db_path)

    if not client.has_collection(collection_name):
        return []

    # Build search parameters
    search_params = {"metric_type": "COSINE"}
    if filter:
        # Milvus filter expression (e.g., 'source == "papers"')
        filter_expr = " and ".join([f'{k} == "{v}"' for k, v in filter.items()])
        search_params["filter"] = filter_expr

    # Perform search
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=limit,
        search_params=search_params,
        output_fields=["text", "source"],
    )

    # Convert to (document, score) tuples
    if not results or not results[0]:
        return []

    docs_with_scores = []
    for result in results[0]:
        entity = result.get("entity", {})
        doc = MilvusDocument(
            text=entity.get("text", ""),
            metadata={
                "source": entity.get("source", "unknown"),
                "id": entity.get("id", ""),
            },
        )
        score = result.get("distance", 0.0)
        docs_with_scores.append((doc, score))

    return docs_with_scores


def get_document_by_id_in_milvus(
    doc_id: str,
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db",
) -> Optional[MilvusDocument]:
    """Retrieve a specific document by ID from Milvus.

    Args:
        doc_id: The document ID to retrieve.
        collection_name: Name of the Milvus collection.
        db_path: Path to the local Milvus database file.

    Returns:
        MilvusDocument if found, None otherwise.
    """
    client = MilvusClient(db_path)

    if not client.has_collection(collection_name):
        return None

    try:
        # Query by ID
        results = client.query(
            collection_name=collection_name,
            filter=f"id == {doc_id}",
            output_fields=["text", "source"],
        )

        if not results:
            return None

        entity = results[0]
        return MilvusDocument(
            text=entity.get("text", ""),
            metadata={
                "source": entity.get("source", "unknown"),
                "id": str(entity.get("id", "")),
            },
        )

    except Exception:
        return None


def browse_documents_in_milvus(
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db",
    filter: Optional[Dict[str, Any]] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[MilvusDocument]:
    """Browse documents with optional filtering and pagination.

    Args:
        collection_name: Name of the Milvus collection.
        db_path: Path to the local Milvus database file.
        filter: Optional filter criteria.
        limit: Maximum number of documents to return.
        offset: Number of documents to skip.

    Returns:
        List of MilvusDocument objects.
    """
    client = MilvusClient(db_path)

    if not client.has_collection(collection_name):
        return []

    try:
        # Build filter expression
        filter_expr = ""
        if filter:
            filter_expr = " and ".join([f'{k} == "{v}"' for k, v in filter.items()])

        # Query with limit and offset
        results = client.query(
            collection_name=collection_name,
            filter=filter_expr if filter_expr else "",
            output_fields=["text", "source"],
            limit=limit,
            offset=offset,
        )

        documents = []
        for entity in results:
            doc = MilvusDocument(
                text=entity.get("text", ""),
                metadata={
                    "source": entity.get("source", "unknown"),
                    "id": str(entity.get("id", "")),
                },
            )
            documents.append(doc)

        return documents

    except Exception:
        return []


def get_milvus_stats(
    collection_name: str = "doc_embeddings",
    db_path: str = "./milvus_demo.db",
) -> Dict[str, Any]:
    """Get statistics about the Milvus collection.

    Args:
        collection_name: Name of the Milvus collection.
        db_path: Path to the local Milvus database file.

    Returns:
        Dictionary containing collection statistics.
    """
    client = MilvusClient(db_path)

    if not client.has_collection(collection_name):
        return {"total_documents": 0, "sources": []}

    try:
        # Get total count
        stats = client.get_collection_stats(collection_name)
        total_count = stats.get("row_count", 0)

        # Get unique sources
        source_results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["source"],
        )

        sources = list(
            set(result.get("source", "unknown") for result in source_results)
        )

        return {
            "total_documents": total_count,
            "sources": sources,
            "collection_name": collection_name,
            "db_path": db_path,
        }

    except Exception:
        return {"total_documents": 0, "sources": []}
