"""
Factory for creating database services
"""

from abc import ABC, abstractmethod
from typing import List


class DatabaseService(ABC):
    """Abstract base class for database services"""

    @abstractmethod
    def store_embeddings(
        self, embeddings: List[List[float]], texts: List[str], source: str = "generic"
    ) -> int:
        """Store embeddings and return number of inserted records"""
        pass

    @abstractmethod
    def search_embeddings(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """Search for similar embeddings"""
        pass


class MilvusService(DatabaseService):
    """Milvus database service"""

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
        """Store embeddings in Milvus"""
        from dirac_dataset.data_access.milvus import store_embeddings_in_milvus

        return store_embeddings_in_milvus(
            embeddings, texts, source, self.collection_name, self.db_path
        )

    def search_embeddings(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """Search for similar embeddings in Milvus"""
        from .milvus import search_embeddings_in_milvus

        return search_embeddings_in_milvus(
            query_embedding, self.collection_name, self.db_path, top_k
        )


class ChromaService(DatabaseService):
    """Chroma database service"""

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
        """Store embeddings in Chroma"""
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
        """Search for similar embeddings in Chroma"""
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        return results


class HuggingFaceService(DatabaseService):
    """HuggingFace Dataset service for storing embeddings"""

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
        """Store embeddings as HuggingFace Dataset"""
        if not embeddings or not texts:
            return 0

        data = {"text": texts, "embedding": embeddings, "source": [source] * len(texts)}

        dataset = self.Dataset.from_dict(data)
        dataset.save_to_disk(f"{self.db_path}/{self.dataset_name}_{source}")
        return len(texts)

    def search_embeddings(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """Search for similar embeddings in HuggingFace Dataset"""
        # For now, return empty list - would need FAISS implementation
        return []


class DatabaseFactory:
    """Factory for creating database services"""

    @staticmethod
    def create_database_service(
        db_type: str, db_path: str, collection_name: str
    ) -> DatabaseService:
        """Create a database service based on type"""
        if db_type.lower() == "milvus":
            return MilvusService(db_path, collection_name)
        elif db_type.lower() == "chroma":
            return ChromaService(db_path, collection_name)
        elif db_type.lower() == "huggingface":
            return HuggingFaceService(db_path, collection_name)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
