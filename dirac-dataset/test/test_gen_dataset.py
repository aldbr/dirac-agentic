"""
Unit tests for dataset generation public API
"""

import json
import tempfile
from pathlib import Path
import pytest
from pytest_mock import MockerFixture

from llama_index.core.schema import TextNode
from dirac_dataset.business_logic.gen_dataset import generate_dataset, Repo


@pytest.fixture
def mock_environment(monkeypatch):
    """Set up GitHub token in environment."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake_token")


@pytest.fixture
def temp_files():
    """Create temporary files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repos_file = Path(temp_dir) / "repos.json"
        pdfs_file = Path(temp_dir) / "pdfs.json"
        out_dir = Path(temp_dir) / "output"

        repos_file.write_text("{}")
        pdfs_file.write_text("[]")

        yield repos_file, pdfs_file, out_dir


def test_generate_dataset_success(mock_environment, temp_files, mocker: MockerFixture):
    """Test successful dataset generation with mocked dependencies."""
    repos_file, pdfs_file, out_dir = temp_files

    # Setup mocks
    mock_load_repos = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_repos_file"
    )
    mock_load_pdfs_file = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_pdfs_file"
    )
    mock_load_pdfs = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_pdfs")
    mock_load_doc = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_doc")
    mock_download_pdf = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._download_pdf"
    )
    mocker.patch("shutil.rmtree")

    # Configure mock return values
    mock_load_repos.return_value = [
        Repo(url="https://github.com/owner/repo", branch="main")
    ]
    mock_load_pdfs_file.return_value = ["https://example.com/paper.pdf"]

    mock_load_pdfs.return_value = [
        TextNode(text="PDF content", metadata={"source": "paper.pdf"})
    ]

    mock_load_doc.return_value = [
        TextNode(text="Repo content", metadata={"source": "README.md"})
    ]

    mock_download_pdf.return_value = Path("/tmp/paper.pdf")

    # Execute function
    result = generate_dataset(repos_file, pdfs_file, out_dir, verbose=True)

    # Verify result structure
    assert "pdf_docs" in result
    assert "md_docs" in result
    assert "total_records" in result
    assert "output_path" in result
    assert result["pdf_docs"] == 1
    assert result["md_docs"] == 1
    assert result["total_records"] == 2
    assert result["output_path"] == str(out_dir)

    # Verify mocks were called
    mock_load_repos.assert_called_once_with(repos_file)
    mock_load_pdfs_file.assert_called_once_with(pdfs_file)
    mock_load_pdfs.assert_called_once()
    mock_load_doc.assert_called_once_with(
        "https://github.com/owner/repo", branch="main"
    )
    mock_download_pdf.assert_called_once()


def test_generate_dataset_missing_github_token(temp_files, monkeypatch):
    """Test error when GITHUB_TOKEN is missing."""
    repos_file, pdfs_file, out_dir = temp_files

    # Clear environment variables
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(
        ValueError, match="GITHUB_TOKEN environment variable is required"
    ):
        generate_dataset(repos_file, pdfs_file, out_dir)


def test_generate_dataset_progress_callback(
    mock_environment, temp_files, mocker: MockerFixture
):
    """Test that progress callback is called during generation."""
    repos_file, pdfs_file, out_dir = temp_files
    progress_calls = []

    def mock_progress(phase, current, total):
        progress_calls.append((phase, current, total))

    # Setup mocks
    mock_load_repos = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_repos_file"
    )
    mock_load_pdfs_file = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_pdfs_file"
    )
    mock_load_pdfs = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_pdfs")
    mock_load_doc = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_doc")
    mock_download_pdf = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._download_pdf"
    )
    mocker.patch("shutil.rmtree")

    mock_load_repos.return_value = [
        Repo(url="https://github.com/owner/repo", branch="main")
    ]
    mock_load_pdfs_file.return_value = ["https://example.com/paper.pdf"]
    mock_load_pdfs.return_value = [TextNode(text="PDF", metadata={})]
    mock_load_doc.return_value = [TextNode(text="Doc", metadata={})]
    mock_download_pdf.return_value = Path("/tmp/test.pdf")

    generate_dataset(repos_file, pdfs_file, out_dir, progress_callback=mock_progress)

    # Verify progress callback was called
    assert len(progress_calls) > 0
    phase_names = [call[0] for call in progress_calls]
    assert "pdf_download" in phase_names
    assert "repo_processing" in phase_names

    # Verify progress tracking structure
    pdf_calls = [call for call in progress_calls if call[0] == "pdf_download"]
    repo_calls = [call for call in progress_calls if call[0] == "repo_processing"]

    assert len(pdf_calls) >= 2  # Start and end
    assert len(repo_calls) >= 2  # Start and end


def test_generate_dataset_multiple_repos(
    mock_environment, temp_files, mocker: MockerFixture
):
    """Test dataset generation with multiple repositories."""
    repos_file, pdfs_file, out_dir = temp_files

    # Setup mocks
    mock_load_repos = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_repos_file"
    )
    mock_load_pdfs_file = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_pdfs_file"
    )
    mock_load_pdfs = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_pdfs")
    mock_load_doc = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_doc")
    mocker.patch("dirac_dataset.business_logic.gen_dataset._download_pdf")
    mocker.patch("shutil.rmtree")

    # Configure multiple repos
    mock_load_repos.return_value = [
        Repo(url="https://github.com/owner/repo1", branch="main"),
        Repo(url="https://github.com/owner/repo2", branch="develop"),
    ]
    mock_load_pdfs_file.return_value = []
    mock_load_pdfs.return_value = []
    mock_load_doc.return_value = [
        TextNode(text="Content from repo1", metadata={"source": "repo1"})
    ]

    result = generate_dataset(repos_file, pdfs_file, out_dir)

    # Verify multiple repos were processed
    assert mock_load_doc.call_count == 2
    assert result["md_docs"] == 2  # mock_load_doc returns 1 doc but is called twice


def test_generate_dataset_multiple_pdfs(
    mock_environment, temp_files, mocker: MockerFixture
):
    """Test dataset generation with multiple PDFs."""
    repos_file, pdfs_file, out_dir = temp_files

    # Setup mocks
    mock_load_repos = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_repos_file"
    )
    mock_load_pdfs_file = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_pdfs_file"
    )
    mock_load_pdfs = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_pdfs")
    mock_download_pdf = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._download_pdf"
    )
    mocker.patch("dirac_dataset.business_logic.gen_dataset._load_doc")
    mocker.patch("shutil.rmtree")

    # Configure multiple PDFs
    mock_load_repos.return_value = []
    mock_load_pdfs_file.return_value = [
        "https://example.com/paper1.pdf",
        "https://example.com/paper2.pdf",
        "https://example.com/paper3.pdf",
    ]
    mock_load_pdfs.return_value = [
        TextNode(text="PDF 1 content", metadata={"source": "paper1.pdf"}),
        TextNode(text="PDF 2 content", metadata={"source": "paper2.pdf"}),
        TextNode(text="PDF 3 content", metadata={"source": "paper3.pdf"}),
    ]
    mock_download_pdf.return_value = Path("/tmp/test.pdf")

    result = generate_dataset(repos_file, pdfs_file, out_dir)

    # Verify multiple PDFs were processed
    assert mock_download_pdf.call_count == 3
    assert result["pdf_docs"] == 3


def test_generate_dataset_error_handling_repo_failure(
    mock_environment, temp_files, mocker: MockerFixture
):
    """Test that repo loading failures don't stop the entire process."""
    repos_file, pdfs_file, out_dir = temp_files

    # Setup mocks
    mock_load_repos = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_repos_file"
    )
    mock_load_pdfs_file = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_pdfs_file"
    )
    mock_load_pdfs = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_pdfs")
    mock_load_doc = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_doc")
    mocker.patch("dirac_dataset.business_logic.gen_dataset._download_pdf")
    mocker.patch("shutil.rmtree")

    # Configure repos with one that will fail
    mock_load_repos.return_value = [
        Repo(url="https://github.com/owner/good-repo", branch="main"),
        Repo(url="https://github.com/owner/bad-repo", branch="main"),
    ]
    mock_load_pdfs_file.return_value = []
    mock_load_pdfs.return_value = []

    # Make the second call fail
    mock_load_doc.side_effect = [
        [TextNode(text="Good repo content", metadata={})],
        Exception("Repository access failed"),
    ]

    # Should not raise an exception
    result = generate_dataset(repos_file, pdfs_file, out_dir)

    # Verify both repos were attempted
    assert mock_load_doc.call_count == 2
    assert result["md_docs"] == 1  # Only the successful one


def test_generate_dataset_error_handling_pdf_failure(
    mock_environment, temp_files, mocker: MockerFixture
):
    """Test that PDF download failures don't stop the entire process."""
    repos_file, pdfs_file, out_dir = temp_files

    # Setup mocks
    mock_load_repos = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_repos_file"
    )
    mock_load_pdfs_file = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._load_pdfs_file"
    )
    mock_load_pdfs = mocker.patch("dirac_dataset.business_logic.gen_dataset._load_pdfs")
    mock_download_pdf = mocker.patch(
        "dirac_dataset.business_logic.gen_dataset._download_pdf"
    )
    mocker.patch("dirac_dataset.business_logic.gen_dataset._load_doc")
    mocker.patch("shutil.rmtree")

    # Configure PDFs with some that will fail
    mock_load_repos.return_value = []
    mock_load_pdfs_file.return_value = [
        "https://example.com/good.pdf",
        "https://example.com/bad.pdf",
        "https://example.com/another-good.pdf",
    ]
    mock_load_pdfs.return_value = [
        TextNode(text="Good PDF content", metadata={}),
        TextNode(text="Another good PDF content", metadata={}),
    ]

    # Make some downloads fail
    mock_download_pdf.side_effect = [
        Path("/tmp/good.pdf"),
        Exception("Download failed"),
        Path("/tmp/another-good.pdf"),
    ]

    # Should not raise an exception
    result = generate_dataset(repos_file, pdfs_file, out_dir)

    # Verify all downloads were attempted
    assert mock_download_pdf.call_count == 3
    assert result["pdf_docs"] == 2  # Only the successful ones


# Fixtures for test data
@pytest.fixture
def repos_file():
    """Create a temporary repos file for testing."""
    repos_data = {
        "dirac": {"url": "https://github.com/DIRACGrid/DIRAC", "branch": "integration"},
        "diracx": {"url": "https://github.com/DIRACGrid/diracx", "branch": "main"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(repos_data, f)
        yield Path(f.name)

    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def pdfs_file():
    """Create a temporary PDFs file for testing."""
    pdfs_data = ["https://example.com/paper1.pdf", "https://example.com/paper2.pdf"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pdfs_data, f)
        yield Path(f.name)

    Path(f.name).unlink(missing_ok=True)
