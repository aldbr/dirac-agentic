"""Factory for creating database services.

This module provides a factory pattern implementation for creating different
types of database services (Milvus, Chroma, HuggingFace) with a unified interface.
"""

from abc import ABC, abstractmethod
from typing import List


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
