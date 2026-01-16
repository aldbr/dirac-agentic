"""
Chroma database operations
"""

import chromadb
from typing import List, Dict, Any, Optional


class ChromaDatabase:
    """Chroma vector database implementation"""

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

    def get_or_create_collection(self, collection_name: str):
        """Get or create a collection"""
        return self.client.get_or_create_collection(name=collection_name)

    def store_embeddings(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        texts: List[str],
        source: str = "generic",
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Store embeddings in Chroma"""
        if not embeddings or not texts:
            return 0

        collection = self.get_or_create_collection(collection_name)

        # Generate IDs
        ids = [f"{source}_{i}" for i in range(len(texts))]

        # Prepare metadata
        if metadatas is None:
            metadatas = [{"source": source} for _ in texts]
        else:
            # Ensure source is in metadata
            for metadata in metadatas:
                metadata["source"] = source

        collection.add(
            embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
        )

        return len(texts)

    def search_embeddings(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search for similar embeddings"""
        collection = self.get_or_create_collection(collection_name)

        results = collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )

        return results

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of items in a collection"""
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        self.client.delete_collection(name=collection_name)
