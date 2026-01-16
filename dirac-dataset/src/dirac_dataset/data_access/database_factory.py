"""Factory for creating database services.

This module provides a factory pattern implementation for creating different
types of database services (Milvus, Chroma, HuggingFace) with a unified interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple


class DatabaseService(ABC):
    """Abstract base class for database services.

    Defines the interface that all database service implementations must follow.
    """

    @abstractmethod
    def store_embeddings(
        self, embeddings: List[List[float]], texts: List[str], source: str = "generic"
    ) -> int:
        """Store embeddings with their corresponding texts.

        Args:
            embeddings: List of embedding vectors to store.
            texts: List of text documents corresponding to embeddings.
            source: Source label for the embeddings.

        Returns:
            Number of successfully inserted records.
        """
        pass

    @abstractmethod
    def search_embeddings(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """Search for embeddings similar to the query vector.

        Args:
            query_embedding: Query embedding vector to search for.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries containing similar embeddings and metadata.
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for similar documents with similarity scores.

        Args:
            query_embedding: Query embedding vector to search for.
            top_k: Number of top results to return.
            filter: Optional filter criteria (e.g., {"source": "papers"}).

        Returns:
            List of tuples (document, score) sorted by similarity.
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Any]:
        """Retrieve a specific document by ID.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Document object if found, None otherwise.
        """
        pass

    @abstractmethod
    def browse_documents(
        self, filter: Optional[Dict[str, Any]] = None, limit: int = 20, offset: int = 0
    ) -> List[Any]:
        """Browse documents with optional filtering and pagination.

        Args:
            filter: Optional filter criteria.
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.

        Returns:
            List of document objects.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary containing database statistics.
        """
        pass


class MilvusService(DatabaseService):
    """Milvus vector database service implementation.

    Provides database operations using Milvus as the backend storage.
    """

    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.collection_name = collection_name
        try:
            from pymilvus import MilvusClient

            self.client = MilvusClient(db_path)
        except ImportError:
            raise ImportError("pymilvus is required for Milvus database")

    def store_embeddings(
        self, embeddings: List[List[float]], texts: List[str], source: str = "generic"
    ) -> int:
        """Store embeddings in Milvus database.

        Args:
            embeddings: List of embedding vectors to store.
            texts: List of text documents corresponding to embeddings.
            source: Source label for the embeddings.

        Returns:
            Number of successfully inserted records.
        """
        from dirac_dataset.data_access.milvus import store_embeddings_in_milvus

        return store_embeddings_in_milvus(
            embeddings, texts, source, self.collection_name, self.db_path
        )

    def search_embeddings(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """Search for similar embeddings in Milvus database.

        Args:
            query_embedding: Query embedding vector to search for.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries containing similar embeddings and metadata.
        """
        from .milvus import search_embeddings_in_milvus

        return search_embeddings_in_milvus(
            query_embedding, self.collection_name, self.db_path, top_k
        )

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for similar documents with similarity scores."""
        from .milvus import search_embeddings_with_scores_in_milvus

        return search_embeddings_with_scores_in_milvus(
            query_embedding, self.collection_name, self.db_path, top_k, filter
        )

    def get_document(self, doc_id: str) -> Optional[Any]:
        """Retrieve a specific document by ID."""
        from .milvus import get_document_by_id_in_milvus

        return get_document_by_id_in_milvus(doc_id, self.collection_name, self.db_path)

    def browse_documents(
        self, filter: Optional[Dict[str, Any]] = None, limit: int = 20, offset: int = 0
    ) -> List[Any]:
        """Browse documents with optional filtering and pagination."""
        from .milvus import browse_documents_in_milvus

        return browse_documents_in_milvus(
            self.collection_name, self.db_path, filter, limit, offset
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        from .milvus import get_milvus_stats

        return get_milvus_stats(self.collection_name, self.db_path)


class ChromaService(DatabaseService):
    """Chroma vector database service implementation.

    Provides database operations using ChromaDB as the backend storage.
    """

    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.collection_name = collection_name
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except ImportError:
            raise ImportError("chromadb is required for Chroma database")

    def store_embeddings(
        self, embeddings: List[List[float]], texts: List[str], source: str = "generic"
    ) -> int:
        """Store embeddings in Chroma database.

        Args:
            embeddings: List of embedding vectors to store.
            texts: List of text documents corresponding to embeddings.
            source: Source label for the embeddings.

        Returns:
            Number of successfully inserted records.
        """
        if not embeddings or not texts:
            return 0

        ids = [f"{source}_{i}" for i in range(len(texts))]
        metadatas = [{"source": source} for _ in texts]

        self.collection.add(
            embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
        )
        return len(texts)

    def search_embeddings(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """Search for similar embeddings in Chroma database.

        Args:
            query_embedding: Query embedding vector to search for.
            top_k: Number of top results to return.

        Returns:
            Dictionary containing search results from ChromaDB.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        return results

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for similar documents with similarity scores."""
        # TODO: Implement proper Chroma search with scores and filtering
        return []

    def get_document(self, doc_id: str) -> Optional[Any]:
        """Retrieve a specific document by ID."""
        # TODO: Implement Chroma document retrieval by ID
        return None

    def browse_documents(
        self, filter: Optional[Dict[str, Any]] = None, limit: int = 20, offset: int = 0
    ) -> List[Any]:
        """Browse documents with optional filtering and pagination."""
        # TODO: Implement Chroma document browsing
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        # TODO: Implement Chroma statistics
        return {"total_documents": 0, "sources": []}


class HuggingFaceService(DatabaseService):
    """HuggingFace Dataset service for storing embeddings.

    Provides database operations using HuggingFace Datasets as the backend storage.
    Note: Search functionality is limited without FAISS integration.
    """

    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.dataset_name = collection_name
        try:
            from datasets import Dataset

            self.Dataset = Dataset
        except ImportError:
            raise ImportError("datasets is required for HuggingFace storage")

    def store_embeddings(
        self, embeddings: List[List[float]], texts: List[str], source: str = "generic"
    ) -> int:
        """Store embeddings as HuggingFace Dataset.

        Args:
            embeddings: List of embedding vectors to store.
            texts: List of text documents corresponding to embeddings.
            source: Source label for the embeddings.

        Returns:
            Number of successfully stored records.
        """
        if not embeddings or not texts:
            return 0

        data = {"text": texts, "embedding": embeddings, "source": [source] * len(texts)}

        dataset = self.Dataset.from_dict(data)
        dataset.save_to_disk(f"{self.db_path}/{self.dataset_name}_{source}")
        return len(texts)

    def search_embeddings(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """Search for similar embeddings in HuggingFace Dataset.

        Note: Currently returns empty list as FAISS integration is not implemented.

        Args:
            query_embedding: Query embedding vector to search for.
            top_k: Number of top results to return.

        Returns:
            Empty list (search not implemented without FAISS).
        """
        # For now, return empty list - would need FAISS implementation
        return []

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for similar documents with similarity scores."""
        # TODO: Implement HuggingFace FAISS search with scores
        return []

    def get_document(self, doc_id: str) -> Optional[Any]:
        """Retrieve a specific document by ID."""
        # TODO: Implement HuggingFace document retrieval by ID
        return None

    def browse_documents(
        self, filter: Optional[Dict[str, Any]] = None, limit: int = 20, offset: int = 0
    ) -> List[Any]:
        """Browse documents with optional filtering and pagination."""
        # TODO: Implement HuggingFace document browsing
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        # TODO: Implement HuggingFace statistics
        return {"total_documents": 0, "sources": []}


class DatabaseFactory:
    """Factory for creating database services.

    Provides static methods to create appropriate database service instances
    based on the specified database type.
    """

    @staticmethod
    def create_database_service(
        db_type: str, db_path: str, collection_name: str
    ) -> DatabaseService:
        """Create a database service instance based on the specified type.

        Args:
            db_type: Type of database service ("milvus", "chroma", "huggingface").
            db_path: Path to the database storage location.
            collection_name: Name of the collection/table to use.

        Returns:
            DatabaseService instance for the specified type.

        Raises:
            ValueError: If the database type is not supported.
        """
        if db_type.lower() == "milvus":
            return MilvusService(db_path, collection_name)
        elif db_type.lower() == "chroma":
            return ChromaService(db_path, collection_name)
        elif db_type.lower() == "huggingface":
            return HuggingFaceService(db_path, collection_name)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
