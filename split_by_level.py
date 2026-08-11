"""
Split data/{train,eval}.jsonl into a low+medium file and a high file.

Why: the LLM-judge audit found "high" reasoning-level examples reliably
score lower on level_match (avg 3.53/5 vs 4.89/5 for low, 4.67/5 for medium)
-- the <think> blocks often skip the required "name a concrete alternative
and reject it" step. Rather than throw those examples away, keep them
separate: train the base model on low+medium (clean signal), then layer a
second LoRA/adapter pass trained on high (and any future "extra high") on
top of it.

Run this after every data/{train,eval}.jsonl rebuild (i.e. after
dedup_and_split.py):

  python3 split_by_level.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = ["train", "eval"]


def main():
    for split in SPLITS:
        src = ROOT / "data" / f"{split}.jsonl"
        if not src.exists():
            print(f"skip {src} (not found)")
            continue

        low_medium, high = [], []
        with open(src) as f:
            for line in f:
                ex = json.loads(line)
                (high if ex["meta"]["level"] == "high" else low_medium).append(ex)

        lm_path = ROOT / "data" / f"{split}_low_medium.jsonl"
        high_path = ROOT / "data" / f"{split}_high.jsonl"
        with open(lm_path, "w") as f:
            for ex in low_medium:
                f.write(json.dumps(ex) + "\n")
        with open(high_path, "w") as f:
            for ex in high:
                f.write(json.dumps(ex) + "\n")

        print(f"{split}: low_medium={len(low_medium)} -> {lm_path.name}   "
              f"high={len(high)} -> {high_path.name}")


if __name__ == "__main__":
    main()
