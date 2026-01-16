"""
HuggingFace dataset operations
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np


class HuggingFaceDatasetStorage:
    """HuggingFace Dataset storage implementation"""

    def __init__(self, base_path: str = "./hf_datasets"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def store_embeddings(
        self,
        dataset_name: str,
        embeddings: List[List[float]],
        texts: List[str],
        source: str = "generic",
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Store embeddings as HuggingFace Dataset"""
        if not embeddings or not texts:
            return 0

        # Prepare data dictionary
        data = {"text": texts, "embedding": embeddings, "source": [source] * len(texts)}

        # Add metadata if provided
        if metadata:
            # Flatten metadata into individual columns
            metadata_keys: Set[str] = set()
            for meta in metadata:
                metadata_keys.update(meta.keys())

            for key in metadata_keys:
                data[key] = [meta.get(key, None) for meta in metadata]

        # Create dataset
        dataset = Dataset.from_dict(data)

        # Save to disk with source-specific naming
        save_path = self.base_path / f"{dataset_name}_{source}"
        dataset.save_to_disk(save_path)

        return len(texts)

    def load_dataset(self, dataset_name: str, source: Optional[str] = None) -> Dataset:
        """Load dataset from disk"""
        if source:
            load_path = self.base_path / f"{dataset_name}_{source}"
        else:
            load_path = self.base_path / dataset_name

        return load_from_disk(load_path)

    def combine_datasets(self, dataset_name: str, sources: List[str]) -> DatasetDict:
        """Combine multiple source datasets into a DatasetDict"""
        datasets = {}

        for source in sources:
            try:
                dataset = self.load_dataset(dataset_name, source)
                datasets[source] = dataset
            except Exception as e:
                print(f"Warning: Could not load dataset for source '{source}': {e}")

        return DatasetDict(datasets)

    def search_embeddings(
        self,
        dataset_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity"""
        dataset = self.load_dataset(dataset_name, source)

        if "embedding" not in dataset.column_names:
            raise ValueError("Dataset does not contain embeddings")

        # Convert query embedding to numpy array
        query_emb = np.array(query_embedding)

        # Calculate similarities
        similarities = []
        for i, row in enumerate(dataset):
            emb = np.array(row["embedding"])
            # Cosine similarity
            similarity = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append((i, similarity))

        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]

        # Return top results
        results = []
        for idx in top_indices:
            row = dataset[idx]
            results.append(
                {
                    "text": row["text"],
                    "source": row["source"],
                    "similarity": similarities[idx][1],
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in ["text", "source", "embedding"]
                    },
                }
            )

        return results

    def get_dataset_info(
        self, dataset_name: str, source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get information about a dataset"""
        dataset = self.load_dataset(dataset_name, source)

        info = {
            "num_rows": len(dataset),
            "columns": dataset.column_names,
            "features": dataset.features,
        }

        # Get source distribution if no specific source
        if source is None and "source" in dataset.column_names:
            source_counts: Dict[str, int] = {}
            for row in dataset:
                src = row["source"]
                source_counts[src] = source_counts.get(src, 0) + 1
            info["source_distribution"] = source_counts

        return info
