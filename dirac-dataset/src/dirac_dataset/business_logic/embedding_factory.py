"""Embedding model factory - Business Logic Layer.

Provides free embedding models ranging from CPU-friendly to high-end GPU models.
"""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from typing import Literal, Dict, Any, Optional, List
import requests
import time
import json
import torch
from pathlib import Path
import logging

from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)

ModelTier = Literal["cpu", "gpu", "multi-gpu"]

# Supported model prefixes for validation
SUPPORTED_MODEL_PREFIXES = ["sentence-transformers/", "BAAI/", "intfloat/"]


class EmbeddingModelInfo(BaseModel):
    """Pydantic model for embedding model metadata.

    Provides validation and type safety for model configuration.
    """

    model_config = ConfigDict(
        extra="forbid",  # Don't allow extra fields
        validate_assignment=True,  # Validate on assignment
        str_strip_whitespace=True,  # Automatically strip whitespace
    )

    id: str = Field(..., description="Full HuggingFace model identifier")
    tier: ModelTier = Field(..., description="Hardware tier requirement")
    description: str = Field(..., description="Human-readable model description")
    library_name: str = Field(
        default="sentence-transformers", description="Framework library"
    )
    pipeline_tag: str = Field(
        default="sentence-similarity", description="HuggingFace pipeline tag"
    )
    downloads: int = Field(default=0, ge=0, description="Total download count")
    default: bool = Field(
        default=False, description="Whether this is the default model"
    )

    @field_validator("id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Validate model ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("Model ID must be a non-empty string")
        if not any(prefix in v for prefix in SUPPORTED_MODEL_PREFIXES):
            raise ValueError("Model ID must be from a supported organization")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description format."""
        if not v or not isinstance(v, str):
            raise ValueError("Description must be a non-empty string")
        if len(v) > 200:
            raise ValueError("Description must be 200 characters or less")
        return v.strip()

    def get_display_name(self) -> str:
        """Get user-friendly model name.

        Returns:
            Short name without organization prefix.
        """
        return self.id.split("/")[-1]

    def is_cpu_friendly(self) -> bool:
        """Check if model is suitable for CPU inference.

        Returns:
            True if model can run efficiently on CPU.
        """
        return self.tier == "cpu"

    def requires_gpu(self) -> bool:
        """Check if model requires GPU acceleration.

        Returns:
            True if model needs GPU for efficient inference.
        """
        return self.tier in {"gpu", "multi-gpu"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for backward compatibility.

        Returns:
            Dictionary representation of the model.
        """
        return self.model_dump()

    def to_json(self) -> str:
        """Convert model to JSON string.

        Returns:
            JSON string representation of the model.
        """
        return self.model_dump_json()


def create_embedding_model(
    model_name: str = "all-MiniLM-L6-v2", device: str = "auto"
) -> BaseEmbedding:
    """Create an embedding model using LlamaIndex's unified API.

    All models are free (no API key required) but vary in hardware requirements.

    Args:
        model_name: HuggingFace model identifier.
        device: "cpu", "cuda", or "auto" (auto-detects available hardware).

    Returns:
        Ready-to-use embedding model instance.

    Raises:
        ValueError: If requesting paid models (OpenAI, etc.).
    """
    # Handle model name formatting
    if not model_name.startswith(("sentence-transformers/", "BAAI/", "intfloat/")):
        model_name = f"sentence-transformers/{model_name}"

    # Handle device auto-detection
    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Auto-detected device: {device}")
        except ImportError:
            device = "cpu"
            logger.debug("PyTorch not available, defaulting to CPU")

    return HuggingFaceEmbedding(model_name=model_name, device=device)


def fetch_popular_embedding_models(max_models: int = 20) -> List[Dict[str, Any]]:
    """Dynamically fetch popular embedding models from HuggingFace API.

    This keeps the model list fresh without manual maintenance.

    Args:
        max_models: Maximum number of models to fetch.

    Returns:
        List of model metadata dictionaries.
    """
    try:
        # Query HuggingFace API for popular sentence-transformer models
        url = "https://huggingface.co/api/models"
        params = {
            "filter": "sentence-transformers",
            "sort": "downloads",  # Most popular first
            "limit": max_models,
            "full": True,
        }

        response = requests.get(url, params=params, timeout=10)  # type: ignore[arg-type]
        response.raise_for_status()

        models = response.json()

        # Filter for embedding models and add metadata
        embedding_models = []
        for model in models:
            model_id = model.get("id", "")
            downloads = model.get("downloads", 0)

            # Skip if not a sentence-transformer or too few downloads
            if (
                not any(
                    prefix in model_id
                    for prefix in ["sentence-transformers/", "BAAI/", "intfloat/"]
                )
                or downloads < 1000
            ):
                continue

            # Estimate hardware tier based on model size/name
            tier = _estimate_model_tier(model_id, model.get("safetensors", {}))

            embedding_models.append(
                {
                    "id": model_id,
                    "downloads": downloads,
                    "tier": tier,
                    "description": model.get("description", "")[:100],
                    "last_updated": model.get("lastModified", ""),
                    "tags": model.get("tags", []),
                }
            )

        return embedding_models[:max_models]

    except Exception as e:
        logger.warning(f"Could not fetch models from HuggingFace API: {e}")
        return []


def _estimate_model_tier(model_id: str, safetensors_info: Dict) -> str:
    """Estimate hardware requirements based on model ID and size.

    Args:
        model_id: HuggingFace model identifier.
        safetensors_info: Model file size information.

    Returns:
        Hardware tier classification.
    """
    model_id_lower = model_id.lower()

    # Check model size if available
    total_size = 0
    if isinstance(safetensors_info, dict):
        total_info = safetensors_info.get("total")
        if isinstance(total_info, dict):
            total_size = int(total_info.get("size", 0))
        elif isinstance(total_info, (int, float)):
            total_size = int(total_info)

    # Size-based classification (rough estimates)
    if total_size > 1_000_000_000:  # > 1GB
        return "multi-gpu"
    elif total_size > 200_000_000:  # > 200MB
        return "gpu"

    # Name-based heuristics
    if any(indicator in model_id_lower for indicator in ["large", "7b", "xl", "xxl"]):
        return "multi-gpu"
    elif any(
        indicator in model_id_lower for indicator in ["base", "medium", "bge-", "e5-"]
    ):
        return "gpu"
    else:
        return "cpu"


def get_cached_models() -> Optional[Dict[str, Dict[str, Any]]]:
    """Load cached model list from disk.

    Returns:
        Cached model dictionary if valid and recent, None otherwise.
    """
    cache_file = Path.home() / ".cache" / "dirac_dataset" / "embedding_models.json"

    try:
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)

            # Check if cache is recent (less than 7 days old)
            if time.time() - cached.get("timestamp", 0) < 7 * 24 * 3600:
                return cached.get("models", {})
    except Exception:
        pass

    return None


def cache_models(models: Dict[str, Dict[str, Any]]) -> None:
    """Save model list to disk cache.

    Args:
        models: Dictionary of models to cache.
    """
    cache_file = Path.home() / ".cache" / "dirac_dataset" / "embedding_models.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        cache_data = {"timestamp": time.time(), "models": models}

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception:
        pass  # Silent fail - caching is not critical


def get_available_models(use_dynamic: bool = True) -> Dict[str, EmbeddingModelInfo]:
    """Get available free embedding models.

    Args:
        use_dynamic: If True, tries to fetch latest models from HuggingFace API.
                    If False, uses hardcoded fallback list.

    Returns:
        Dictionary mapping model names to validated EmbeddingModelInfo instances.
    """
    if use_dynamic:
        # Try to get cached models first
        cached = get_cached_models()
        if cached:
            # Convert cached dict to Pydantic models
            try:
                return {
                    name: EmbeddingModelInfo.model_validate(data)
                    for name, data in cached.items()
                }
            except Exception as e:
                logger.warning(f"Invalid cached models, refreshing: {e}")

        # Try to fetch fresh models from HuggingFace API
        try:
            dynamic_models = fetch_popular_embedding_models()
            if dynamic_models:
                # Convert to Pydantic models with validation
                converted = {}
                for model in dynamic_models:
                    try:
                        name = model["id"].replace("sentence-transformers/", "")
                        model_info = EmbeddingModelInfo(
                            id=model["id"],
                            tier=model["tier"],
                            description=model["description"]
                            or "Popular embedding model",
                            downloads=model["downloads"],
                        )
                        converted[name] = model_info
                    except Exception as e:
                        logger.debug(
                            f"Skipping invalid model {model.get('id', 'unknown')}: {e}"
                        )

                # Only cache and return if we got some valid models
                if converted:
                    cache_dict = {
                        name: info.model_dump() for name, info in converted.items()
                    }
                    cache_models(cache_dict)
                    return converted
                else:
                    logger.warning("No valid models found in HuggingFace API response")
        except Exception as e:
            logger.warning(f"Dynamic model discovery failed: {e}")
            logger.debug("Falling back to curated model list")

    # Fallback to curated list (always reliable)
    return _get_curated_models()


def _get_curated_models() -> Dict[str, EmbeddingModelInfo]:
    """Fallback curated list - essential models only, with Pydantic validation.

    Returns:
        Dictionary mapping model names to validated EmbeddingModelInfo instances.
    """
    try:
        models = {
            "all-MiniLM-L6-v2": EmbeddingModelInfo(
                id="sentence-transformers/all-MiniLM-L6-v2",
                tier="cpu",
                description="Fast and efficient model for general purpose embedding",
                downloads=50000000,
                default=True,
            ),
            "all-mpnet-base-v2": EmbeddingModelInfo(
                id="sentence-transformers/all-mpnet-base-v2",
                tier="cpu",
                description="Higher quality CPU-friendly model with good performance",
                downloads=25000000,
            ),
            "BAAI/bge-base-en-v1.5": EmbeddingModelInfo(
                id="BAAI/bge-base-en-v1.5",
                tier="gpu",
                description="High-quality model optimized for GPU inference",
                downloads=15000000,
            ),
            "intfloat/e5-large-v2": EmbeddingModelInfo(
                id="intfloat/e5-large-v2",
                tier="multi-gpu",
                description="State-of-the-art embedding model requiring significant resources",
                downloads=8000000,
            ),
        }

        # Validate all models at creation time
        logger.debug(f"Created {len(models)} validated embedding models")
        return models

    except Exception as e:
        logger.error(f"Failed to create curated models: {e}")
        # Return empty dict if model creation fails
        return {}


def create_embedding_service(
    model_name: str = "all-MiniLM-L6-v2", device: str = "auto"
) -> BaseEmbedding:
    """Create an embedding service using free models.

    Args:
        model_name: Model identifier (defaults to CPU-friendly).
        device: Hardware target ("cpu", "cuda", "auto").

    Returns:
        Ready-to-use embedding model instance.
    """
    return create_embedding_model(model_name, device)


def get_models_by_tier(
    tier: Optional[ModelTier] = None,
) -> Dict[str, EmbeddingModelInfo]:
    """Get available models, optionally filtered by hardware tier.

    Args:
        tier: Hardware tier to filter by (cpu/gpu/multi-gpu), or None for all.

    Returns:
        Dictionary of model names to validated EmbeddingModelInfo instances.
    """
    models = get_available_models()

    if tier:
        models = {k: v for k, v in models.items() if v.tier == tier}

    logger.debug(f"Retrieved {len(models)} models for tier: {tier or 'all'}")
    return models


def get_cpu_models() -> List[str]:
    """Get list of CPU-friendly model names.

    Returns:
        List of model names suitable for CPU inference.
    """
    models = get_available_models()
    return [name for name, info in models.items() if info.is_cpu_friendly()]


def get_gpu_models() -> List[str]:
    """Get list of GPU-optimized model names.

    Returns:
        List of model names requiring GPU acceleration.
    """
    models = get_available_models()
    return [name for name, info in models.items() if info.requires_gpu()]


def get_default_model() -> str:
    """Get the default CPU-friendly model dynamically.

    Returns:
        Name of the default embedding model, or fallback if none marked default.
    """
    models = get_available_models()

    # Look for explicitly marked default model
    for name, info in models.items():
        if info.default:
            return name

    # Fallback: first CPU model available
    cpu_models = [name for name, info in models.items() if info.is_cpu_friendly()]
    if cpu_models:
        return cpu_models[0]

    # Final fallback
    return "all-MiniLM-L6-v2"


def get_recommended_model(tier: Optional[ModelTier] = None) -> Optional[str]:
    """Get the most downloaded (popular) model for a given tier.

    Args:
        tier: Hardware tier to filter by. If None, returns most popular overall.

    Returns:
        Name of the most popular model, or None if no models found.
    """
    models = get_models_by_tier(tier)

    if not models:
        return None

    # Sort by downloads (descending) and return the most popular
    sorted_models = sorted(models.items(), key=lambda x: x[1].downloads, reverse=True)
    return sorted_models[0][0] if sorted_models else None


def get_fastest_model(tier: Optional[ModelTier] = None) -> Optional[str]:
    """Get the fastest model for a given tier (based on heuristics).

    Args:
        tier: Hardware tier to filter by. If None, returns fastest overall.

    Returns:
        Name of the fastest model, or None if no models found.
    """
    models = get_models_by_tier(tier)

    if not models:
        return None

    # Heuristic: models with "mini" or "small" in name are usually faster
    # Sort by: 1) name hints, 2) downloads as tiebreaker
    def speed_score(item):
        name, info = item
        name_lower = name.lower()
        score = 0

        if "mini" in name_lower:
            score += 100
        elif "small" in name_lower:
            score += 50
        elif "base" in name_lower:
            score += 25

        # Add downloads as tiebreaker (normalized)
        score += min(info.downloads / 1000000, 50)  # Cap at 50 points
        return score

    sorted_models = sorted(models.items(), key=speed_score, reverse=True)
    return sorted_models[0][0] if sorted_models else None


def list_models(
    tier: Optional[ModelTier] = None,
    sort_by: Literal["downloads", "name", "default"] = "downloads",
) -> List[str]:
    """List available models with flexible sorting options.

    Args:
        tier: Hardware tier to filter by (cpu/gpu/multi-gpu), or None for all.
        sort_by: How to sort the results - by downloads, name, or defaults first.

    Returns:
        List of model names sorted according to criteria.
    """
    models = get_models_by_tier(tier)

    if sort_by == "downloads":
        # Most popular first
        sorted_items = sorted(
            models.items(), key=lambda x: x[1].downloads, reverse=True
        )
    elif sort_by == "name":
        # Alphabetical
        sorted_items = sorted(models.items(), key=lambda x: x[0])
    elif sort_by == "default":
        # Default models first, then by downloads
        sorted_items = sorted(
            models.items(), key=lambda x: (not x[1].default, -x[1].downloads)
        )
    else:
        # Fallback to downloads
        sorted_items = sorted(
            models.items(), key=lambda x: x[1].downloads, reverse=True
        )

    return [name for name, _ in sorted_items]


def search_models(query: str) -> List[str]:
    """Search for models by name or description.

    Args:
        query: Search term to match against model names and descriptions.

    Returns:
        List of matching model names sorted by relevance.
    """
    models = get_available_models()
    query_lower = query.lower()

    matches = []
    for name, info in models.items():
        score = 0
        name_lower = name.lower()
        desc_lower = info.description.lower()

        # Exact name match gets highest score
        if query_lower == name_lower:
            score += 100
        # Name contains query
        elif query_lower in name_lower:
            score += 50
        # Description contains query
        elif query_lower in desc_lower:
            score += 25

        if score > 0:
            matches.append((name, score))

    # Sort by score (descending) then by name
    matches.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in matches]


def refresh_model_list() -> bool:
    """Force refresh of available models from HuggingFace API.

    Returns:
        True if refresh succeeded, False otherwise.
    """
    logger.info("Refreshing embedding model list from HuggingFace API")

    try:
        dynamic_models = fetch_popular_embedding_models()
        if dynamic_models:
            converted = {}
            for model in dynamic_models:
                try:
                    name = model["id"].replace("sentence-transformers/", "")
                    model_info = EmbeddingModelInfo(
                        id=model["id"],
                        tier=model["tier"],
                        description=model["description"] or "Popular embedding model",
                        downloads=model["downloads"],
                    )
                    converted[name] = (
                        model_info.model_dump()
                    )  # Convert to dict for caching
                except Exception as e:
                    logger.warning(
                        f"Skipping invalid model {model.get('id', 'unknown')}: {e}"
                    )

            cache_models(converted)
            logger.info(f"Successfully fetched {len(converted)} embedding models")
            logger.debug("Model list cached for 7 days")
            return True
        else:
            logger.warning("No embedding models found from HuggingFace API")
            return False

    except Exception as e:
        logger.error(f"Failed to refresh embedding models: {e}")
        return False


def validate_model_name(model_name: str) -> bool:
    """Validate if a model name exists in our available models list.

    This is different from Pydantic validation - this checks if a user-provided
    model name actually exists in our curated or dynamically fetched model list.

    Args:
        model_name: Name of the model to validate (user input).

    Returns:
        True if model exists in available models, False otherwise.
    """
    try:
        available_models = get_available_models()

        # Direct match in available models
        if model_name in available_models:
            return True

        # Check if it's a full HuggingFace path that matches any model ID
        for model_info in available_models.values():
            if model_name == model_info.id:
                return True

        # Check if it can be prefixed with sentence-transformers/
        if "/" not in model_name:
            prefixed_name = f"sentence-transformers/{model_name}"
            for model_info in available_models.values():
                if prefixed_name == model_info.id:
                    return True

        logger.debug(f"Model '{model_name}' not found in available models")
        return False

    except Exception as e:
        logger.warning(f"Could not validate model {model_name}: {e}")
        return False


def get_model_info(model_name: str) -> Optional[EmbeddingModelInfo]:
    """Get detailed information about a specific model.

    Args:
        model_name: Name of the model to get information for.

    Returns:
        EmbeddingModelInfo instance if model exists, None otherwise.
    """
    models = get_available_models()
    return models.get(model_name)


def get_model_statistics() -> Dict[str, Any]:
    """Get statistics about available models.

    Returns:
        Dictionary containing model statistics and counts by tier.
    """
    models = get_available_models()

    tiers = {"cpu": 0, "gpu": 0, "multi-gpu": 0}
    total_downloads = 0

    for model_info in models.values():
        tiers[model_info.tier] = tiers.get(model_info.tier, 0) + 1
        total_downloads += model_info.downloads

    stats = {
        "cpu_models": tiers["cpu"],
        "gpu_models": tiers["gpu"],
        "multi_gpu_models": tiers["multi-gpu"],
        "total_downloads": total_downloads,
        "is_dynamic": models != _get_curated_models(),
        "total_models": len(models),
        "default_model": get_default_model(),
        "recommended_cpu": get_recommended_model("cpu"),
        "recommended_gpu": get_recommended_model("gpu"),
    }

    logger.debug(f"Model statistics: {stats}")
    return stats


def get_model_summary() -> Dict[str, Any]:
    """Get structured model summary data for presentation layers.

    Returns:
        Dictionary containing all model information structured for display.
    """
    models = get_available_models()
    stats = get_model_statistics()

    # Organize models by tier with details
    models_by_tier = {}
    for tier_str in ["cpu", "gpu", "multi-gpu"]:
        tier: ModelTier = tier_str  # type: ignore[assignment]
        tier_models = list_models(tier=tier, sort_by="downloads")
        if tier_models:
            models_by_tier[tier] = [
                {
                    "name": model_name,
                    "info": models[model_name],
                    "display_name": models[model_name].get_display_name(),
                    "is_default": models[model_name].default,
                    "downloads_formatted": f"{models[model_name].downloads:,}",
                }
                for model_name in tier_models
            ]

    return {
        "statistics": stats,
        "models_by_tier": models_by_tier,
        "recommendations": {
            "default": stats["default_model"],
            "cpu": stats["recommended_cpu"],
            "gpu": stats["recommended_gpu"],
        },
    }
