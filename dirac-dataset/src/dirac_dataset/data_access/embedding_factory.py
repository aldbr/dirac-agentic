"""
Factory for creating embedding services using LlamaIndex's unified API
"""

from llama_index.core.base.embeddings.base import BaseEmbedding


def create_embedding_model(model_name: str) -> BaseEmbedding:
    """
    Create an embedding model using LlamaIndex's unified API.

    Examples:
        # HuggingFace models (free, local)
        create_embedding_model("BAAI/bge-small-en-v1.5")
        create_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

        # OpenAI models (requires API key)
        create_embedding_model("text-embedding-3-small")
        create_embedding_model("text-embedding-ada-002")

        # Local models
        create_embedding_model("local:BAAI/bge-small-en-v1.5")
    """
    model_name = model_name.lower()

    # OpenAI models
    if model_name.startswith("text-embedding"):
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=model_name)

    # HuggingFace models (default, free)
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_name)


class EmbeddingFactory:
    """Factory for creating embedding services with model flexibility"""

    @staticmethod
    def create_embedding_service(
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> BaseEmbedding:
        """
        Create an embedding service that can handle any model.

        Args:
            model_name: Model identifier. Examples:
                - "BAAI/bge-small-en-v1.5" (HuggingFace, free, high quality)
                - "sentence-transformers/all-MiniLM-L6-v2" (HuggingFace, free)
                - "text-embedding-3-small" (OpenAI, requires API key)

        Returns:
            BaseEmbedding: Unified embedding interface
        """
        return create_embedding_model(model_name)

    @staticmethod
    def get_recommended_models():
        """Get list of recommended embedding models"""
        return {
            "free_high_quality": "BAAI/bge-small-en-v1.5",
            "free_fast": "sentence-transformers/all-MiniLM-L6-v2",
            "free_multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "openai_small": "text-embedding-3-small",
            "openai_large": "text-embedding-3-large",
        }
