# Fine-tuning Examples

Tools for fine-tuning language models with DIRAC domain knowledge.

## Scripts

### `generate_qa_dataset.py`
Generates question-answer pairs from documentation using a local quantized LLM (Qwen 2.5 0.5B). Features:
- 4-bit quantization for low VRAM usage
- Batch processing with configurable batch size
- Resume capability (saves progress to `qa_pairs.json`)
- Pydantic validation for generated Q&A pairs

### `converter.py`
Document conversion utilities:
- `rst_or_md_to_txt()` - Convert RST/Markdown to plain text
- `pdf_to_txt()` - Extract text from PDF files

### `log.py`
Shared logging configuration with Rich formatting.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Place your documentation text files in a `documentation_txt/` directory.
2. Run the QA generator:

```bash
python generate_qa_dataset.py
```

The output Q&A pairs are saved to `qa_pairs.json`.
