# RAG Chatbot Example

A self-contained RAG chatbot that indexes DIRAC documentation into a local Milvus vector DB and answers questions using any OpenAI-compatible LLM.

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
export OPENAI_API_KEY=your_key
```

### Using alternative LLM providers

Any OpenAI-compatible endpoint works:

```bash
# Ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=llama3

# Together AI
export OPENAI_BASE_URL=https://api.together.xyz/v1
export OPENAI_API_KEY=your_together_key
export OPENAI_MODEL=meta-llama/Llama-3-70b-chat-hf
```

## Usage

```bash
# Step 1: Build the vector DB from a local dataset or HuggingFace Hub
python rag_chatbot.py build ./my_dataset
python rag_chatbot.py build myorg/dirac-docs

# Step 2: Start the interactive chatbot
python rag_chatbot.py chat
```
