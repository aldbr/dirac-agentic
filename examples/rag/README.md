# RAG Chatbot Example

A self-contained RAG chatbot that indexes DIRAC documentation into a local Milvus vector DB and answers questions using HuggingFace Inference API with open-source models.

## Prerequisites

A HuggingFace dataset, either:

- Generated locally by `dirac-dataset`:
  ```bash
  pixi run -e dirac-dataset gen-dataset --repos-file data/repos.json --pdfs-file data/pdfs.json --out ./my_dataset
  ```
- Or available on HuggingFace Hub (e.g., `myorg/dirac-docs`)

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
```

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token (required for gated models / higher rate limits) | — |
| `HF_MODEL` | Model to use for chat generation | `meta-llama/Meta-Llama-3.1-8B-Instruct` |

## Usage

```bash
# Step 1: Build the vector DB from a local dataset or HuggingFace Hub
python rag_chatbot.py build ./my_dataset
python rag_chatbot.py build myorg/dirac-docs

# Step 2: Start the interactive chatbot
python rag_chatbot.py chat
```
