# Building a **DIRAC Q-A Dataset** (status: **shelved**)

|              |                                                                                          |
|--------------|------------------------------------------------------------------------------------------|
| **Date**     | 2025-04-28                                                                                |
| **Status**   | **Shelved** – superseded by a RAG + MCP approach                                          |
| **Stake**    | Summarise our exploration of fine-tuning a small-footprint chatbot with DIRAC knowledge   |
| **Context**  | We wanted an _offline_ helper that answers technical questions about the DIRAC middleware using only internal docs & scientific papers. The first idea was to generate a synthetic **Q-A dataset** and fine-tune a lightweight model (≤ 4 GB VRAM). |

---

## 1 Original plan

1. Chunk every `.rst` / `.txt` / PDF-to-text file.
2. Prompt a small instruction model to emit a **JSON array**
   ```json
   [{"question": "...", "answer": "..."}]
   ```
3. Review & filter in Argilla
4. Fine-tune with LoRA.

## 2 Iterative work & results

| Step | Change / optimisation | Goal | Outcome |
|------|----------------------|------|---------|
| 1 | **4-bit NF4 quant (BitsAndBytes)** | Fit on GTX-1050 (4 GB) | ✅ Loads & runs |
| 2 | **Micro-batch = 2 prompts** | Avoid re-loading weights | ✅ No OOM |
| 3 | Token-level chunker | Stable prompt length | ✅ Mem OK |
| 4 | Cache chunks with `datasets` | Skip re-tokenising | ✅ Cold run only |
| 5 | `pipeline("text-generation")` + **`apply_chat_template()`** | Proper chat markup | ✅ Model responds |


## 3 Next steps

| Step | Change / optimisation | Goal |
|------|----------------------|------|
| 6 | `do_sample=False`, shorter 200-tok output | Fewer hallucinations |
| 7 | *Self-heal* pass (“Fix this JSON…”) | Salvage bad outputs |
| 8 | Swap to **Mistral-7B-GPTQ** (CPU) | Better adherence |
| 9 | Test nightly `json_mode=True` | Stop at first valid JSON |

## 4  Why we’re pausing this track

* **Retrieval-Augmented Generation (RAG)** was evaluated in parallel and **answers technical queries directly** from fresh documentation without any supervised fine-tune.
* The **Model Customisation Pipeline (MCP)** of HF provides an end-to-end RAG→fine-tune workflow; we can feed the same documents as a _vector store_ first, then optionally distil them into supervised Q-A later.
* Given the parsing noise & latency of the current 0.5 B solution, **RAG delivers higher accuracy right now**, so our effort shifts to:
  1. Build a **DIRAC document index** with RAG.
  2. Expose it via an internal chatbot.
  3. Re-visit fine-tuning (LoRA) _after_ we collect real user queries + the automatically generated Q-A pairs.

_For posterity_: the scripts live under `/scripts/`.

## 5  Decision

* **Stop investing in JSON-generation fine-tune for now.**
* Prioritise **RAG + MCP** implementation; keep the Q-A generator as a fallback or a future distillation step.
