"""
Database loading business logic
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from datasets import DatasetDict
from dirac_dataset.log import logger
from dirac_dataset.business_logic.embedding_factory import (
    create_embedding_service,
    validate_model_name,
    get_cpu_models,
)
from dirac_dataset.data_access.database_factory import DatabaseFactory


def load_dataset_to_database(
    dataset_path: Path,
    db_type: str = "milvus",
    db_path: Path = Path("./vector_db"),
    collection_name: str = "doc_embeddings",
    embedding_model: str = "all-MiniLM-L6-v2",
    verbose: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """Load HuggingFace dataset, generate embeddings, and store in vector database"""
    # Configure logging
    logger.setLevel("DEBUG" if verbose else "INFO")
    logger.info(f"Starting database loading from HuggingFace dataset: {dataset_path}")
    logger.info(f"Using {embedding_model} embeddings and {db_type} database")

    # Load HuggingFace dataset
    logger.info(f"Loading HuggingFace dataset from {dataset_path}")
    if progress_callback:
        progress_callback("dataset_loading", 0, 1)

    try:
        ds_splits = DatasetDict.load_from_disk(dataset_path)
        logger.info(f"Successfully loaded dataset with {len(ds_splits)} splits")

        # Gather dataset info
        dataset_info = {}
        total_records = 0
        for split, ds in ds_splits.items():
            dataset_info[split] = {
                "count": len(ds),
                "sample": ds[0] if len(ds) > 0 else None,
            }
            total_records += len(ds)
            logger.debug(f"Split '{split}': {len(ds)} records")

        if progress_callback:
            progress_callback("dataset_loading", 1, 1)

    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        raise ValueError(f"Could not load HuggingFace dataset: {e}")

    # Validate embedding model before proceeding
    if not validate_model_name(embedding_model):
        available_models = list(get_cpu_models())[:3]  # Show first 3 CPU models
        raise ValueError(
            f"Invalid embedding model '{embedding_model}'. "
            f"Available CPU-friendly models: {', '.join(available_models)}. "
            f"Use --embedding <model_name> to specify a different model."
        )

    # Get embedding model and database instances
    logger.debug(f"Initializing {embedding_model} embedding service")
    embedding_service = create_embedding_service(embedding_model)

    logger.debug(f"Initializing {db_type} database service at {db_path}")
    database_service = DatabaseFactory.create_database_service(
        db_type, str(db_path), collection_name
    )

    results = {}

    # Process each split
    for source, ds in ds_splits.items():
        logger.info(f"Processing {len(ds)} documents from '{source}' split")

        if len(ds) == 0:
            logger.info(f"Skipping empty split: {source}")
            results[f"{source}_embeddings"] = 0
            continue

        if progress_callback:
            progress_callback(f"{source}_embedding", 0, len(ds))

        # Extract texts from dataset
        texts = [record["text"] for record in ds]
        logger.debug(f"Extracted {len(texts)} texts from {source} split")

        # Generate embeddings
        logger.debug(f"Generating embeddings for {source} documents")
        embeddings = embedding_service.get_text_embedding_batch(texts)

        # Store in database
        inserted_count = database_service.store_embeddings(
            embeddings, texts, source=source
        )
        logger.info(f"Stored {inserted_count} {source} embeddings")
        results[f"{source}_embeddings"] = inserted_count

        if progress_callback:
            progress_callback(f"{source}_embedding", len(ds), len(ds))

    # Calculate totals
    total_embeddings = sum(results.values())
    logger.info(
        f"Database loading complete - Total embeddings stored: {total_embeddings}"
    )
    logger.info(f"Database saved to {db_path}")

    return {
        "dataset_info": dataset_info,
        "total_records": total_records,
        "total_embeddings": total_embeddings,
        "embedding_model": embedding_model,
        "db_type": db_type,
        "db_path": str(db_path),
        "pdf_embeddings": results.get("papers_embeddings", 0),
        "md_embeddings": results.get("docs_embeddings", 0),
        "issue_embeddings": results.get("issues_embeddings", 0),
        **results,
    }
