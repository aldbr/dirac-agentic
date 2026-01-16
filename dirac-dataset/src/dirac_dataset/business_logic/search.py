"""Document search functionality - Business Logic Layer.

Provides high-level search operations that coordinate between embedding models
and vector database implementations to deliver relevant document results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
import logging

from dirac_dataset.data_access.database_factory import DatabaseFactory
from dirac_dataset.business_logic.embedding_factory import create_embedding_service

logger = logging.getLogger(__name__)

# Type definitions
SourceType = Literal["papers", "docs", "issues"]


class SearchResult:
    """Container for search result with metadata."""

    def __init__(
        self,
        doc_id: str,
        text: str,
        source: SourceType,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
    ):
        self.doc_id = doc_id
        self.text = text
        self.source = source
        self.metadata = metadata or {}
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
            "score": self.score,
        }


def search_documents(
    query: str,
    db_path: Path,
    top_k: int = 5,
    source_filter: Optional[SourceType] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    db_type: str = "milvus",
) -> List[SearchResult]:
    """Search documents using semantic similarity.

    Args:
        query: Natural language search query.
        db_path: Path to vector database.
        top_k: Maximum number of results to return.
        source_filter: Filter by source type (papers, docs, issues).
        embedding_model: Model for query embedding.
        db_type: Vector database type (milvus, chroma, huggingface).

    Returns:
        List of SearchResult objects ranked by similarity.

    Raises:
        ValueError: If database not found or invalid parameters.
        RuntimeError: If search operation fails.
    """
    if not db_path.exists():
        raise ValueError(f"Database not found at {db_path}")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    logger.info(f"Searching for '{query}' in {db_path}")

    try:
        # Create embedding service
        embedding_service = create_embedding_service(embedding_model)
        query_embedding = embedding_service.get_text_embedding(query)

        # Create database service
        db_service = DatabaseFactory.create_database_service(
            db_type=db_type,
            db_path=str(db_path),
            collection_name="doc_embeddings",
        )

        # Perform search using the new interface with scores
        raw_results = db_service.similarity_search_with_score(
            query_embedding,
            top_k,
            filter={"source": source_filter} if source_filter else None,
        )

        # Convert to SearchResult objects
        # raw_results is now a list of tuples (document, score)
        results = []
        for i, (doc, score) in enumerate(raw_results):
            # Extract metadata from document
            metadata = getattr(doc, "metadata", {})
            source = metadata.get("source", "unknown")
            doc_id = metadata.get("id", f"doc_{i}")

            result = SearchResult(
                doc_id=doc_id,
                text=getattr(doc, "text", ""),
                source=source,
                metadata=metadata,
                score=float(score) if score is not None else None,
            )
            results.append(result)

        logger.info(f"Found {len(results)} results for query")
        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise RuntimeError(f"Document search failed: {e}") from e


def get_document(
    doc_id: str,
    db_path: Path,
    db_type: str = "milvus",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Optional[SearchResult]:
    """Retrieve a specific document by ID.

    Note: Current implementation is limited - would need database schema extensions.

    Args:
        doc_id: Unique document identifier.
        db_path: Path to vector database.
        db_type: Vector database type.
        embedding_model: Embedding model (for DB service creation).

    Returns:
        SearchResult if found, None otherwise.

    Raises:
        ValueError: If database not found.
        RuntimeError: If retrieval operation fails.
    """
    if not db_path.exists():
        raise ValueError(f"Database not found at {db_path}")

    logger.debug(f"Retrieving document {doc_id} from {db_path}")

    try:
        # Create database service
        db_service = DatabaseFactory.create_database_service(
            db_type=db_type,
            db_path=str(db_path),
            collection_name="doc_embeddings",
        )

        # Get document by ID
        doc = db_service.get_document(doc_id)

        if doc is None:
            logger.debug(f"Document {doc_id} not found")
            return None

        # Convert to SearchResult
        metadata = getattr(doc, "metadata", {})
        result = SearchResult(
            doc_id=doc_id,
            text=getattr(doc, "text", ""),
            source=metadata.get("source", "unknown"),
            metadata=metadata,
        )

        logger.debug(f"Retrieved document {doc_id}")
        return result

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise RuntimeError(f"Document retrieval failed: {e}") from e


def get_similar_documents(
    doc_id: str,
    db_path: Path,
    top_k: int = 5,
    db_type: str = "milvus",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[SearchResult]:
    """Find documents similar to a given document.

    Args:
        doc_id: ID of the reference document.
        db_path: Path to vector database.
        top_k: Maximum number of similar documents to return.
        db_type: Vector database type.
        embedding_model: Embedding model.

    Returns:
        List of similar SearchResult objects.

    Raises:
        ValueError: If reference document not found.
        RuntimeError: If similarity search fails.
    """
    # First get the reference document
    ref_doc = get_document(doc_id, db_path, db_type, embedding_model)
    if ref_doc is None:
        raise ValueError(f"Reference document {doc_id} not found")

    # Search for similar documents using the reference text
    # Exclude the reference document itself from results
    all_results = search_documents(
        query=ref_doc.text,
        db_path=db_path,
        top_k=top_k + 1,  # Get one extra to account for self-match
        db_type=db_type,
        embedding_model=embedding_model,
    )

    # Filter out the reference document
    similar_docs = [r for r in all_results if r.doc_id != doc_id][:top_k]

    logger.info(f"Found {len(similar_docs)} similar documents to {doc_id}")
    return similar_docs


def browse_documents(
    source_type: SourceType,
    db_path: Path,
    limit: int = 20,
    offset: int = 0,
    db_type: str = "milvus",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[SearchResult]:
    """Browse documents by source type.

    Note: Limited implementation - would need database schema extensions.

    Args:
        source_type: Type of documents to browse.
        db_path: Path to vector database.
        limit: Maximum number of documents to return.
        offset: Number of documents to skip.
        db_type: Vector database type.
        embedding_model: Embedding model.

    Returns:
        Empty list (not implemented yet).
    """
    logger.info(f"Browsing {source_type} documents (limit={limit}, offset={offset})")

    try:
        # Create database service
        db_service = DatabaseFactory.create_database_service(
            db_type=db_type,
            db_path=str(db_path),
            collection_name="doc_embeddings",
        )

        # Browse with filter
        docs = db_service.browse_documents(
            filter={"source": source_type},
            limit=limit,
            offset=offset,
        )

        # Convert to SearchResult objects
        results = []
        for i, doc in enumerate(docs):
            metadata = getattr(doc, "metadata", {})
            result = SearchResult(
                doc_id=metadata.get("id", f"doc_{offset + i}"),
                text=getattr(doc, "text", ""),
                source=source_type,
                metadata=metadata,
            )
            results.append(result)

        logger.info(f"Retrieved {len(results)} {source_type} documents")
        return results

    except Exception as e:
        logger.error(f"Browse operation failed: {e}")
        raise RuntimeError(f"Browse operation failed: {e}") from e


def get_collection_stats(
    db_path: Path,
    db_type: str = "milvus",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Get statistics about the document collections.

    Note: Limited implementation - would need database schema extensions.

    Args:
        db_path: Path to vector database.
        db_type: Vector database type.
        embedding_model: Embedding model.

    Returns:
        Basic stats dictionary.
    """
    logger.debug(f"Getting collection stats for {db_path}")

    try:
        # Create database service
        db_service = DatabaseFactory.create_database_service(
            db_type=db_type,
            db_path=str(db_path),
            collection_name="doc_embeddings",
        )

        # Get stats
        stats = db_service.get_stats()

        logger.debug("Retrieved collection statistics")
        return stats

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise RuntimeError(f"Stats retrieval failed: {e}") from e
