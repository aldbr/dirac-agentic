# generate_dataset.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterator

from datasets import Dataset, Features, Value
from pydantic import BaseModel
import torch
from tqdm.auto import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ------------------------------------------------------------------------------
# 1.  Config
# ------------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TXT_DIR = Path("documentation_txt")
MAX_PROMPT_TOKENS = 800  # max chars per prompt chunk
MAX_GEN_TOKENS = 400  # max tokens to generate
OUT_FILE = Path("qa_pairs.json")

BATCH_SIZE = 2  # prompts per forward‑pass ‑ tweak for VRAM
GEN_KWARGS = dict(
    max_new_tokens=MAX_GEN_TOKENS, do_sample=True, temperature=0.7, top_p=0.8, top_k=20
)


# ------------------------------------------------------------------------------
# 2.  Prompt helpers
# ------------------------------------------------------------------------------
def build_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant specialised in DIRAC middleware.\n"
                "Return **only** a JSON array of objects with keys "
                "`question`, `answer` – no extra text."
            ),
        },
        {"role": "user", "content": text},
    ]


# ------------------------------------------------------------------------------
# 3.  Pydantic schema & extractor
# ------------------------------------------------------------------------------
class QA(BaseModel):
    question: str
    answer: str


JSON_RE = re.compile(r"\[\s*{.*?}\s*]", re.DOTALL)


def extract_qas(raw: str) -> list[QA]:
    raw = re.sub(r"```(?:json|python)?", "", raw, flags=re.DOTALL)  # strip ``` fences
    blocks = JSON_RE.findall(raw)
    if not blocks:
        raise ValueError("no JSON found")
    data = json.loads(blocks[-1])  # last block -> real data

    qas = []
    for qa in data:
        question = qa.get("question")
        if question and isinstance(question, list):
            # If the question is a list, take the first one
            data["question"] = question[0]
        answer = qa.get("answer")
        if answer and isinstance(answer, list):
            # If the answer is a list, take the first one
            data["answer"] = answer[0]

        qas.append(QA(**qa))
    return qas


# ------------------------------------------------------------------------------
# 4.  Chunk generator
# ------------------------------------------------------------------------------
def iter_chunks(
    folder: Path, max_tokens: int, tokenizer: AutoTokenizer
) -> Iterator[str]:
    for fp in folder.glob("*.txt"):
        text = fp.read_text()
        words = text.split()
        cur: list[str] = []
        for w in words:
            if len(tokenizer(cur + [w]).input_ids) < max_tokens:
                cur.append(w)
            else:
                yield " ".join(cur)
                cur = [w]
        if cur:
            yield " ".join(cur)


# ------------------------------------------------------------------------------
# 5.  Main
# ------------------------------------------------------------------------------
def main() -> None:
    # --- load / resume ---
    collected: list[dict] = []
    if OUT_FILE.exists():
        collected = json.loads(OUT_FILE.read_text())
        print(f"▶ resuming – {len(collected)} QAs already stored")

    # ---- 4-bit quantised model ----
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- HF pipeline ---
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_cfg,
    )
    model.config.sliding_window = None  # silence SDPA warning
    chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # --- Dataset generator ---
    def gen():
        for chunk in iter_chunks(TXT_DIR, MAX_PROMPT_TOKENS, tokenizer):
            prompt = tokenizer.apply_chat_template(
                build_messages(chunk),
                tokenize=False,
                add_generation_prompt=True,
            )
            yield {"prompt": prompt}

    ds = Dataset.from_generator(gen, features=Features({"prompt": Value("string")}))

    # --- generation loop with batching ---
    for batch in tqdm(
        ds.iter(batch_size=BATCH_SIZE),
        total=len(ds) // BATCH_SIZE + 1,
        unit="batch",
        desc="LLM",
    ):
        outs = chat(
            batch["prompt"],
            return_full_text=False,
            batch_size=len(batch["prompt"]),
            **GEN_KWARGS,
        )
        for raw in outs:
            try:
                collected.extend(
                    q.model_dump() for q in extract_qas(raw[0]["generated_text"])
                )
            except Exception as e:
                print(f"❌ bad JSON: {e}\n", raw)
        OUT_FILE.write_text(json.dumps(collected, indent=2, ensure_ascii=False))

    print(f"✅ finished – total {len(collected)} QAs saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
