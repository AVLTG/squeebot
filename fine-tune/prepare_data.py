#!/usr/bin/env python3
"""
prepare_data.py — Convert training-pairs/*.json into Unsloth-ready JSONL.

Reads the 10 category files from ../training-pairs/, applies the training
system prompt, produces train.jsonl + eval.jsonl in OpenAI messages format
(ready for Unsloth SFTTrainer with Llama 3.1 chat template).

Output:
  fine-tune/data/train.jsonl
  fine-tune/data/eval.jsonl

Standard library only — no deps needed.
"""

import json
import random
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAIRS_DIR = ROOT / "training-pairs"
OUT_DIR = HERE / "data"
SYSTEM_PROMPT_PATH = HERE / "system-prompt-training.txt"

EVAL_FRAC = 0.05
SEED = 42


def to_messages(pair: dict, system_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["user"]},
            {"role": "assistant", "content": pair["assistant"]},
        ],
        "category": pair.get("category", ""),
        "tags": pair.get("tags", []),
    }


def main() -> None:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()

    by_category: dict[str, list[dict]] = {}
    for path in sorted(PAIRS_DIR.glob("*.json")):
        pairs = json.loads(path.read_text())
        by_category[path.stem] = pairs
        print(f"  {path.name}: {len(pairs)} pairs")

    # Stratified split: within each category, shuffle then slice off EVAL_FRAC
    train: list[dict] = []
    eval_: list[dict] = []
    for cat, pairs in by_category.items():
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * EVAL_FRAC))
        eval_.extend(shuffled[:n_eval])
        train.extend(shuffled[n_eval:])

    random.shuffle(train)
    random.shuffle(eval_)

    train_path = OUT_DIR / "train.jsonl"
    eval_path = OUT_DIR / "eval.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(to_messages(p, system_prompt), ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for p in eval_:
            f.write(json.dumps(to_messages(p, system_prompt), ensure_ascii=False) + "\n")

    print()
    print(f"Train: {len(train):>5} examples -> {train_path.relative_to(ROOT)}")
    print(f"Eval:  {len(eval_):>5} examples -> {eval_path.relative_to(ROOT)}")

    print("\nTrain category breakdown:")
    for cat, n in sorted(Counter(p["category"] for p in train).items()):
        print(f"  {cat:<25} {n}")
    print("\nEval category breakdown:")
    for cat, n in sorted(Counter(p["category"] for p in eval_).items()):
        print(f"  {cat:<25} {n}")

    sys_chars = len(system_prompt)
    avg_user = sum(len(p["user"]) for p in train) / len(train)
    avg_asst = sum(len(p["assistant"]) for p in train) / len(train)
    print("\nChar count sanity check:")
    print(f"  System prompt:      {sys_chars}")
    print(f"  Avg user turn:      {avg_user:.1f}")
    print(f"  Avg assistant turn: {avg_asst:.1f}")
    print(f"  Approx tokens/ex:   ~{(sys_chars + avg_user + avg_asst) / 4:.0f}  (1 token ≈ 4 chars)")


if __name__ == "__main__":
    main()
