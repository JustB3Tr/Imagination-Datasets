#!/usr/bin/env python3
"""
Local, free deduplication + train/eval split for the generated dataset.

This is the step that would have caught the "50k rows, 29 unique
conversations" problem before it burned a GPU. Run it after generate.py,
before you touch a training run.

Usage:
  python dedup_and_split.py --sim-threshold 0.92 --eval-per-bucket 30

Reads every data/raw/*.jsonl file, embeds each example locally (no API
calls, no cost), greedily drops near-duplicates within each domain/level
bucket, then writes:
  data/train.jsonl
  data/eval.jsonl
  data/dedup_report.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"


# A real fraction of generated rows have a tool_call function.arguments
# string that isn't valid JSON -- usually an under-escaped backslash inside
# a regex/command/path (e.g. a regex \K written as \K instead of \\K) or a
# raw literal newline where \n was needed. Both are recoverable without
# guessing at content. Rows with a genuinely unescaped quote character
# (e.g. `grep -R "pip install"` with the inner quotes never escaped) are
# NOT recoverable this way -- the string boundary itself is ambiguous, and
# guessing where a quote belongs risks silently corrupting the example, so
# those are left for the downstream consumer (train.py) to drop.
_JSON_VALID_ESCAPES = set('"\\/bfnrtu')


def _try_parse_dict(s: str, strict: bool = True):
    try:
        obj = json.loads(s, strict=strict)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _double_stray_backslashes(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] not in _JSON_VALID_ESCAPES:
            out.append("\\\\")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def repair_tool_call_arguments(s: str) -> str | None:
    """Return a JSON string that decodes to a dict, repairing under-escaped
    backslashes and/or raw control characters if needed, or None if it's
    not safely recoverable."""
    if _try_parse_dict(s) is not None or _try_parse_dict(s, strict=False) is not None:
        return s
    fixed = _double_stray_backslashes(s)
    if _try_parse_dict(fixed) is not None or _try_parse_dict(fixed, strict=False) is not None:
        return fixed
    return None


def repair_example_arguments(ex: dict) -> int:
    """Repair tool_call arguments in place across every message. Returns
    the number of arguments strings that were actually changed."""
    n_repaired = 0
    for m in ex.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str) or _try_parse_dict(args) is not None:
                continue
            fixed = repair_tool_call_arguments(args)
            if fixed is not None and fixed != args:
                fn["arguments"] = fixed
                n_repaired += 1
    return n_repaired


def load_all_raw():
    examples = []
    n_repaired = 0
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_repaired += repair_example_arguments(ex)
                examples.append(ex)
    if n_repaired:
        print(f"Repaired {n_repaired} tool_call argument string(s) with under-escaped "
              f"backslashes/control characters (still-broken ones are left for "
              f"train.py to drop -- those have an unescaped quote, not safely "
              f"auto-fixable).")
    return examples


def example_text(ex: dict) -> str:
    parts = []
    for m in ex.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)[:4000]  # cap for embedding speed


def greedy_dedup(examples: list[dict], embeddings: np.ndarray, threshold: float):
    """Keep an example only if it's not too similar to one already kept,
    within the same (domain, level, mode) bucket."""
    kept_idx = []
    kept_embeds_by_bucket = defaultdict(list)  # bucket -> list[(idx, vec)]

    for i, ex in enumerate(examples):
        meta = ex.get("meta", {})
        bucket = (meta.get("domain"), meta.get("level"), meta.get("mode"))
        vec = embeddings[i]
        is_dup = False
        for _, other_vec in kept_embeds_by_bucket[bucket]:
            sim = float(np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec) + 1e-8))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_idx.append(i)
            kept_embeds_by_bucket[bucket].append((i, vec))
    return kept_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-threshold", type=float, default=0.87,
                     help="cosine similarity above this = treated as duplicate. "
                          "Calibrated against real paraphrase pairs: tight rewordings "
                          "of the same task score ~0.86-0.90, loose rewordings drop to "
                          "~0.72-0.77 and start overlapping with genuinely different but "
                          "related tasks. 0.87 catches tight paraphrase spam without "
                          "merging real variety. This is a blunt instrument, not a "
                          "guarantee, still hand-spot-check the output.")
    ap.add_argument("--eval-per-bucket", type=int, default=30,
                     help="how many examples per domain/level bucket to hold out for eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    examples = load_all_raw()
    if not examples:
        print("No raw examples found in data/raw/. Run generate.py first.")
        return
    print(f"Loaded {len(examples)} raw examples.")

    # Identity examples are DELIBERATELY repetitive (same canonical answer,
    # different question phrasing) so the model learns them by heart -- that
    # is exactly the pattern near-dup filtering is designed to remove, so
    # they skip it entirely and always survive intact.
    identity_examples = [ex for ex in examples if ex.get("meta", {}).get("domain") == "identity"]
    dedupable_examples = [ex for ex in examples if ex.get("meta", {}).get("domain") != "identity"]

    print("Embedding locally (all-MiniLM-L6-v2, first run downloads the model)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [example_text(ex) for ex in dedupable_examples]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    kept_idx = greedy_dedup(dedupable_examples, embeddings, args.sim_threshold)
    removed = len(dedupable_examples) - len(kept_idx)
    print(f"Kept {len(kept_idx)}, dropped {removed} near-duplicates "
          f"({removed/len(dedupable_examples)*100:.1f}%) at threshold {args.sim_threshold}.")
    if identity_examples:
        print(f"({len(identity_examples)} identity examples kept as-is, exempt from dedup.)")

    kept = [dedupable_examples[i] for i in kept_idx] + identity_examples

    # bucket counts, so you can see coverage per domain/level before training
    counts = defaultdict(int)
    for ex in kept:
        meta = ex.get("meta", {})
        counts[f"{meta.get('domain')}/{meta.get('level')}/{meta.get('mode')}"] += 1

    print("\nPer-bucket counts after dedup:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    # train/eval split, stratified per domain+level+mode bucket
    rng = np.random.default_rng(args.seed)
    by_bucket = defaultdict(list)
    for ex in kept:
        meta = ex.get("meta", {})
        by_bucket[(meta.get("domain"), meta.get("level"), meta.get("mode"))].append(ex)

    train, eval_ = [], []
    for bucket, items in by_bucket.items():
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_eval = min(args.eval_per_bucket, max(1, len(items) // 10))
        eval_idx = set(idx[:n_eval])
        for i, ex in enumerate(items):
            (eval_ if i in eval_idx else train).append(ex)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(OUT_DIR / "eval.jsonl", "w") as f:
        for ex in eval_:
            f.write(json.dumps(ex) + "\n")

    report = {
        "raw_count": len(examples),
        "kept_after_dedup": len(kept),
        "removed_as_duplicate": removed,
        "sim_threshold": args.sim_threshold,
        "train_count": len(train),
        "eval_count": len(eval_),
        "bucket_counts": counts,
    }
    with open(OUT_DIR / "dedup_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {len(train)} train / {len(eval_)} eval examples.")
    print("Check dedup_report.json bucket_counts for any bucket that's suspiciously thin,")
    print("that's your signal to generate more for that specific domain/level before training.")


if __name__ == "__main__":
    main()
