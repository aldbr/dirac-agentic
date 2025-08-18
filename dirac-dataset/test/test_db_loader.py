"""
Unit tests for database loading logic
"""

from pathlib import Path
import pytest

from dirac_dataset.business_logic import db_loader as db_loader_mod
from dirac_dataset.business_logic.db_loader import load_dataset_to_database


class FakeEmbeddingService:
    def get_text_embedding_batch(self, texts):
        # Return 2-D vectors for simplicity
        return [[0.1, 0.2] for _ in texts]


class FakeDBService:
    def __init__(self):
        self.insert_calls = []

    def store_embeddings(self, embeddings, texts, source="generic") -> int:
        self.insert_calls.append(
            {
                "source": source,
                "count": len(texts),
            }
        )
        return len(texts)


class FakeFactory:
    @staticmethod
    def create_database_service(db_type, db_path, collection_name):
        return FakeDBService()


@pytest.fixture
def patch_common(monkeypatch):
    # Patch model validation to always succeed by default
    monkeypatch.setattr(db_loader_mod, "validate_model_name", lambda _: True)
    # Patch embedding service
    monkeypatch.setattr(
        db_loader_mod,
        "create_embedding_service",
        lambda *_args, **_kw: FakeEmbeddingService(),
    )
    # Patch DB factory
    monkeypatch.setattr(db_loader_mod, "DatabaseFactory", FakeFactory)


def test_load_dataset_to_database_success(monkeypatch, patch_common):
    # Mock DatasetDict.load_from_disk to return simple in-memory splits
    fake_splits = {
        "papers": [{"text": "paper a"}, {"text": "paper b"}],
        "docs": [{"text": "doc c"}],
    }
    monkeypatch.setattr(
        db_loader_mod.DatasetDict, "load_from_disk", lambda _p: fake_splits
    )

    result = load_dataset_to_database(
        Path("/tmp/fake_ds"), db_type="milvus", db_path=Path("./vector_db")
    )

    assert result["total_records"] == 3
    assert result["pdf_embeddings"] == 2  # papers split
    assert result["md_embeddings"] == 1  # docs split
    assert result["issue_embeddings"] == 0
    assert result["papers_embeddings"] == 2
    assert result["docs_embeddings"] == 1


def test_load_dataset_to_database_empty_split(monkeypatch, patch_common):
    fake_splits = {
        "issues": [],
        "docs": [{"text": "d1"}],
    }
    monkeypatch.setattr(
        db_loader_mod.DatasetDict, "load_from_disk", lambda _p: fake_splits
    )

    result = load_dataset_to_database(Path("/tmp/fake_ds"))

    assert result["issues_embeddings"] == 0
    assert result["docs_embeddings"] == 1


def test_load_dataset_to_database_invalid_model(monkeypatch):
    # Force validation failure
    monkeypatch.setattr(db_loader_mod, "validate_model_name", lambda _m: False)
    monkeypatch.setattr(db_loader_mod, "get_cpu_models", lambda: ["m1", "m2", "m3"])
    # Still patch DatasetDict to avoid disk access
    monkeypatch.setattr(
        db_loader_mod.DatasetDict,
        "load_from_disk",
        lambda _p: {"docs": [{"text": "x"}]},
    )

    with pytest.raises(ValueError, match="Invalid embedding model"):
        load_dataset_to_database(Path("/tmp/fake_ds"), embedding_model="bad-model")


def test_load_dataset_to_database_progress_callback(monkeypatch, patch_common):
    fake_splits = {
        "docs": [{"text": "a"}, {"text": "b"}],
    }
    monkeypatch.setattr(
        db_loader_mod.DatasetDict, "load_from_disk", lambda _p: fake_splits
    )

    calls = []

    def progress(phase, current, total):
        calls.append((phase, current, total))

    load_dataset_to_database(Path("/tmp/fake_ds"), progress_callback=progress)

    phases = [c[0] for c in calls]
    assert "dataset_loading" in phases
    assert "docs_embedding" in phases
    # at least start and end events for the docs split
    doc_calls = [c for c in calls if c[0] == "docs_embedding"]
    assert len(doc_calls) >= 1
