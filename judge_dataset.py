#!/usr/bin/env python3
"""
LLM-as-judge pass over the generated dataset.

Everything else in this pipeline checks structure (validate_example),
similarity (dedup_and_split.py), and a handful of manually-read examples.
This is the missing layer: an LLM actually reads each example and scores
it against a concrete rubric, at a scale manual reading can't cover.

Uses the same OpenAI-compatible client pattern as generate.py. Defaults to
the free Gemini tier since DeepSeek's budget is exhausted -- set:
  export IMG2_API_KEY="..."
  export IMG2_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai"
  export IMG2_MODEL="gemini-3.5-flash-lite"

Usage:
  python judge_dataset.py --samples-per-bucket 15 --max-calls 300
  python judge_dataset.py --input data/eval.jsonl --samples-per-bucket 5

Output: judge_results.jsonl (one verdict per judged example, appended --
safe to re-run / resume) and a printed summary at the end.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent
API_KEY = os.environ.get("IMG2_API_KEY")
API_BASE = os.environ.get("IMG2_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai")
MODEL = os.environ.get("IMG2_MODEL", "gemini-3.5-flash-lite")
REASONING_EFFORT = os.environ.get("IMG2_REASONING_EFFORT")

_call_lock = threading.Lock()
_call_state = {"count": 0}


def _try_claim_call(max_calls):
    with _call_lock:
        if max_calls is not None and _call_state["count"] >= max_calls:
            return False
        _call_state["count"] += 1
        return True


JUDGE_INSTRUCTIONS = """
You are grading ONE training example for a coding/agentic assistant finetune.
The example has a stated reasoning_effort level (low/medium/high) in its
system prompt -- judge whether the answer's actual reasoning depth matches
what that level claims, not just whether it's present.

Score each of these 1-5 (5 = excellent, 1 = badly broken):
- correctness: is the technical content actually right? Does the code/logic
  make sense, would it plausibly work, no contradictions between what a
  tool "found" and what the final answer claims?
- level_match: does the reasoning depth genuinely match the stated level?
  low should be terse with no filler. medium should show real (if brief)
  deliberation. high should show thorough reasoning that considers at
  least one alternative before deciding. A "high" example with a shallow
  one-line <think> is a level_match failure even if it's technically
  correct.
- realism: are specifics concrete (real-sounding file names, error
  messages, versions, numbers) rather than generic placeholders like
  "some function" or "an error occurred"?

Return ONLY a JSON object, no markdown fences, no commentary:
{
  "correctness": <1-5>,
  "level_match": <1-5>,
  "realism": <1-5>,
  "issues": ["<short specific problem, if any>", ...],
  "verdict": "pass" or "fail"
}

"fail" means at least one score is 2 or below, or there's a real problem
worth a human looking at. Otherwise "pass". Keep "issues" empty if there
genuinely aren't any -- don't invent nitpicks to fill the list.
""".strip()


def format_example_for_judge(ex: dict) -> str:
    lines = [f"Domain: {ex['meta']['domain']}  Level: {ex['meta']['level']}  Mode: {ex['meta']['mode']}", ""]
    for m in ex["messages"]:
        role = m["role"]
        content = m.get("content") or ""
        if m.get("tool_calls"):
            calls = "; ".join(
                f"{tc['function']['name']}({tc['function']['arguments']})"
                for tc in m["tool_calls"]
            )
            lines.append(f"[{role} -> tool_calls] {calls}")
        else:
            lines.append(f"[{role}] {content[:3000]}")
    return "\n".join(lines)


def load_stratified_sample(path: str, samples_per_bucket: int, seed: int,
                            level_filter: str | None = None) -> list[dict]:
    by_bucket = defaultdict(list)
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["domain"] == "identity":
                continue  # trivially uniform by design, nothing to judge
            if level_filter and ex["meta"]["level"] != level_filter:
                continue
            key = (ex["meta"]["domain"], ex["meta"]["level"], ex["meta"]["mode"])
            by_bucket[key].append(ex)

    rng = random.Random(seed)
    sample = []
    for key, items in sorted(by_bucket.items()):
        picked = rng.sample(items, min(samples_per_bucket, len(items)))
        sample.extend(picked)
    return sample


def judge_one(client: OpenAI, ex: dict, max_calls) -> dict | None:
    if not _try_claim_call(max_calls):
        return "CAPPED"

    formatted = format_example_for_judge(ex)
    extra_body = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You output only valid JSON, nothing else."},
                    {"role": "user", "content": JUDGE_INSTRUCTIONS + "\n\n---\n\n" + formatted},
                ],
                temperature=0.3,
                max_tokens=600,
                extra_body=extra_body,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^<thought>.*?</thought>\s*", "", raw, count=1, flags=re.DOTALL)
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            verdict = json.loads(raw)
            verdict["_meta"] = ex["meta"]
            return verdict
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries:
                # Free-tier per-minute quotas recover fast; back off and retry
                # instead of permanently skipping a good example.
                delay = 20 * (attempt + 1)
                print(f"  [rate limited, retry {attempt+1}/{max_retries} in {delay}s] "
                      f"{ex['meta']['domain']}/{ex['meta']['level']}/{ex['meta']['mode']}", file=sys.stderr)
                time.sleep(delay)
                continue
            print(f"  [skip] {ex['meta']['domain']}/{ex['meta']['level']}/{ex['meta']['mode']}: {e}", file=sys.stderr)
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/train.jsonl")
    ap.add_argument("--output", default="judge_results.jsonl")
    ap.add_argument("--samples-per-bucket", type=int, default=15)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2,
                     help="Keep low for free-tier providers with per-minute quotas.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--level", choices=["low", "medium", "high"], default=None,
                     help="Judge only buckets at this reasoning level.")
    args = ap.parse_args()

    if not API_KEY:
        print("Set IMG2_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    sample = load_stratified_sample(args.input, args.samples_per_bucket, args.seed, args.level)
    print(f"Judging {len(sample)} examples ({args.samples_per_bucket}/bucket) with {MODEL} ...")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    results = []
    out_f = open(args.output, "a")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_one, client, ex, args.max_calls): ex for ex in sample}
            for fut in as_completed(futures):
                verdict = fut.result()
                if verdict in (None, "CAPPED"):
                    continue
                out_f.write(json.dumps(verdict) + "\n")
                out_f.flush()
                results.append(verdict)
    finally:
        out_f.close()

    if not results:
        print("No verdicts collected.")
        return

    print(f"\n{len(results)} verdicts collected -> {args.output}")

    fails = [r for r in results if r.get("verdict") == "fail"]
    print(f"\nOverall: {len(results) - len(fails)}/{len(results)} pass ({(len(results)-len(fails))/len(results)*100:.1f}%)")

    for metric in ["correctness", "level_match", "realism"]:
        vals = [r[metric] for r in results if metric in r]
        if vals:
            print(f"  avg {metric}: {sum(vals)/len(vals):.2f}")

    by_level = defaultdict(list)
    for r in results:
        by_level[r["_meta"]["level"]].append(r)
    print("\nlevel_match by stated level (the key thing to watch):")
    for lv in ["low", "medium", "high"]:
        vals = [r["level_match"] for r in by_level.get(lv, []) if "level_match" in r]
        if vals:
            print(f"  {lv:8s}: avg {sum(vals)/len(vals):.2f}  (n={len(vals)})")

    if fails:
        print(f"\n{len(fails)} failed examples:")
        for r in fails[:20]:
            m = r["_meta"]
            print(f"  {m['domain']}/{m['level']}/{m['mode']}: {r.get('issues')}")


if __name__ == "__main__":
    main()
