#!/usr/bin/env python3
"""
Generate SFT training examples for Imagination 2 Pro.

Works with any OpenAI-compatible API (DeepSeek, z.ai/GLM, etc). Set:
  export IMG2_API_KEY="..."
  export IMG2_API_BASE="https://api.deepseek.com"   # or your provider's base url
  export IMG2_MODEL="deepseek-chat"                  # or your provider's model name

Usage:
  python generate.py --domain agentic_tool_use --level low --variants 3
  python generate.py --domain subagent_orchestration --level high --variants 2
  python generate.py --all   # runs every domain x level combo using --variants

Output: appends JSONL rows to data/raw/<domain>__<level>.jsonl

Also emits a SUBAGENT-mode variant of a fraction of direct-mode examples
(see mode_for_domain in schema_templates.py) so you get subagent-execution
training data almost for free instead of generating a whole extra dataset.
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
from pathlib import Path

from openai import OpenAI

from schema_templates import (
    DOMAINS,
    LEVELS,
    MAX_TOKENS_BY_LEVEL,
    EXAMPLE_JSON_INSTRUCTIONS,
    RUN_SUBAGENT_TOOL_SCHEMA,
    build_system_prompt,
    mode_for_domain,
)

ROOT = Path(__file__).parent
SEEDS_DIR = ROOT / "seeds"
# Override for test/comparison runs so they don't append into the same raw
# files as real production generations (those get merged via dedup_and_split.py).
OUT_DIR = Path(os.environ.get("IMG2_OUT_DIR", str(ROOT / "data" / "raw")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("IMG2_API_KEY")
API_BASE = os.environ.get("IMG2_API_BASE", "https://api.deepseek.com")
MODEL = os.environ.get("IMG2_MODEL", "deepseek-chat")

# Some providers (e.g. Gemini) run hidden "thinking" tokens that eat into
# max_tokens before any visible output is produced, silently truncating
# generations that would otherwise fit. Set IMG2_REASONING_EFFORT=none (or
# whatever value the provider expects) to suppress that; passed through as
# extra_body so it's a no-op for providers that don't recognize the field.
REASONING_EFFORT = os.environ.get("IMG2_REASONING_EFFORT")

# OpenRouter routes a given model through multiple backend providers, often
# at different prices/quantization. Set IMG2_OPENROUTER_PROVIDER to pin a
# specific one (e.g. "gmicloud" for their fp8 DeepSeek V4 Flash route)
# instead of letting OpenRouter auto-select. No-op for non-OpenRouter bases.
OPENROUTER_PROVIDER = os.environ.get("IMG2_OPENROUTER_PROVIDER")

# $ per 1M tokens. Defaults are direct DeepSeek V4 Flash rates as of Aug 2026.
# If you're on OpenRouter, override these (their DeepSeek route runs a bit
# higher, roughly $0.21/$0.31 per 1M as of this writing, check your dashboard).
PRICE_INPUT_PER_1M = float(os.environ.get("IMG2_PRICE_INPUT_PER_1M", "0.14"))
PRICE_OUTPUT_PER_1M = float(os.environ.get("IMG2_PRICE_OUTPUT_PER_1M", "0.28"))

_spend_lock = threading.Lock()
_spend_state = {"usd": 0.0, "capped": False}


def _add_spend(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (prompt_tokens / 1_000_000) * PRICE_INPUT_PER_1M + \
           (completion_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M
    with _spend_lock:
        _spend_state["usd"] += cost
        total = _spend_state["usd"]
    return total


def _over_cap(max_spend: float) -> bool:
    with _spend_lock:
        return _spend_state["usd"] >= max_spend


# Separate from the dollar-based spend cap: some providers (e.g. OpenRouter's
# free-tier models) charge $0 but enforce a hard request-count quota instead
# (their docs note failed attempts count against the quota too, so this
# increments on every attempted call, not just successful ones).
_call_lock = threading.Lock()
_call_state = {"count": 0}


def _try_claim_call(max_calls: int | None) -> bool:
    """Atomically check-and-increment: returns True if this call is allowed to
    proceed (and counts it), False if the cap is already reached. Doing the
    check and increment under one lock (instead of two separate calls) is
    what keeps concurrent workers from all passing the check together and
    overshooting the cap by up to --workers."""
    with _call_lock:
        if max_calls is not None and _call_state["count"] >= max_calls:
            return False
        _call_state["count"] += 1
        return True


# Fraction of agentic_tool_use / general_code examples that also get a
# SUBAGENT-mode variant emitted (same task, different system prompt/framing,
# no larger conversation context). This is how "being a subagent" gets
# trained without a dedicated fourth dataset.
SUBAGENT_VARIANT_RATE = 0.35


def load_seeds(domain: str) -> list[str]:
    with open(SEEDS_DIR / f"{domain}.json") as f:
        return json.load(f)


def build_prompt(domain: str, level: str, seed_task: str, mode: str) -> tuple[str, str]:
    system_prompt = build_system_prompt(mode, level)
    generator_instructions = f"""
You are generating ONE synthetic training example for finetuning a coding/
agentic assistant. The example should teach the DOMAIN "{domain}" at
reasoning level "{level}" in {mode.upper()} mode.

Seed task (use as inspiration, do not restate verbatim, make it concrete
and specific with real details): "{seed_task}"

The system prompt for this example must be exactly:
\"\"\"{system_prompt}\"\"\"

{f'The ONLY tool you may call is run_subagent, schema: '
  f'{json.dumps(RUN_SUBAGENT_TOOL_SCHEMA)}. Do not invent or call any other tools.'
  if mode == 'orchestrator' else
  'Invent 1-2 realistic tools appropriate to the task (e.g. fetch, shell, '
  'file_search, sql_query) with sensible JSON parameter schemas.'}

{EXAMPLE_JSON_INSTRUCTIONS}
""".strip()
    return system_prompt, generator_instructions


# Extra output-token headroom beyond MAX_TOKENS_BY_LEVEL's content budget, to
# cover the JSON wrapper (roles, tool_call ids, escaped nested JSON in tool
# args/results). ORCHESTRATOR mode conversations run several run_subagent
# round-trips and measurably truncate later than direct/subagent mode at the
# same reasoning level (~6.5k vs ~5k chars into the completion in testing),
# so it gets a bigger allowance.
WRAPPER_TOKEN_BUFFER = 800
ORCHESTRATOR_EXTRA_TOKENS = 1200


def max_output_tokens(level: str, mode: str) -> int:
    extra = WRAPPER_TOKEN_BUFFER + (ORCHESTRATOR_EXTRA_TOKENS if mode == "orchestrator" else 0)
    return MAX_TOKENS_BY_LEVEL[level] + extra


def estimate_call_tokens(level: str, mode: str, user_prompt: str) -> tuple[int, int]:
    """Rough input/output token estimate, used by --dry-run and as a fallback
    if a provider doesn't return usage stats. ~4 chars/token is a coarse but
    fine approximation for this purpose."""
    input_tokens = len(user_prompt) // 4 + 30  # +30 for the short system msg
    output_tokens = max_output_tokens(level, mode)  # worst case; dry-run is a ceiling, not an average
    return input_tokens, output_tokens


def validate_example(example: dict, level: str) -> str | None:
    """Return an error string if the example is structurally incomplete or
    malformed, else None. Catches truncated generations that still parse as
    valid JSON (the model got cut off mid-conversation, e.g. right after a
    tool call, with no final answer) -- these silently pass json.loads and
    would otherwise poison the training set undetected."""
    messages = example.get("messages")
    if not messages:
        return "no messages"
    if len(messages) < 3:
        return f"only {len(messages)} message(s), missing system/user/answer structure"
    if messages[0].get("role") != "system":
        return f"first message role is '{messages[0].get('role')}', not system"
    if messages[1].get("role") != "user":
        return f"second message role is '{messages[1].get('role')}', not user"

    last = messages[-1]
    if last.get("role") != "assistant":
        return f"last message role is '{last.get('role')}', not assistant (truncated conversation)"
    if last.get("tool_calls"):
        return "last assistant message still calls a tool (truncated conversation)"
    content = last.get("content")
    if not content or not isinstance(content, str) or len(content.strip()) < 5:
        return "last assistant message has empty/near-empty content"

    if level == "low" and "<think>" in content:
        return "level is low but final answer contains a <think> block"
    if level in ("medium", "high"):
        if not content.lstrip().startswith("<think>"):
            return f"level is {level} but final answer doesn't start with a <think> block"
        after_think = re.sub(r"^<think>.*?</think>", "", content.lstrip(), count=1, flags=re.DOTALL).strip()
        if len(after_think) < 10:
            return f"level is {level} but there's no real answer after the <think> block (truncated)"

    # every tool_call must be immediately followed by its matching tool result,
    # and must actually name a function -- a tool_call with arguments but no
    # function.name is unusable (nothing to dispatch) and json.loads happily
    # accepts it since it's still valid JSON, so this needs an explicit check.
    for i, m in enumerate(messages):
        for call in m.get("tool_calls") or []:
            call_id = call.get("id")
            fn = call.get("function") or {}
            if not fn.get("name"):
                return f"tool_call {call_id!r} is missing function.name"
            if "arguments" not in fn:
                return f"tool_call {call_id!r} is missing function.arguments"
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if not nxt or nxt.get("role") != "tool" or nxt.get("tool_call_id") != call_id:
                return f"tool_call {call_id!r} has no matching tool result message"

    return None


def call_api(client: OpenAI, domain: str, level: str, seed_task: str, mode: str,
             max_spend: float, max_calls: int | None, dry_run: bool) -> dict | str | None:
    system_prompt, user_prompt = build_prompt(domain, level, seed_task, mode)

    if dry_run:
        in_tok, out_tok = estimate_call_tokens(level, mode, user_prompt)
        total = _add_spend(in_tok, out_tok)
        return {"_dry_run": True, "est_total_usd": total}

    if _over_cap(max_spend) or not _try_claim_call(max_calls):
        return "CAPPED"  # cap already hit, don't make the call

    try:
        if max_calls is not None:
            with _call_lock:
                call_num = _call_state["count"]
            print(f"  [call {call_num}/{max_calls}] {domain}/{level}/{mode}", file=sys.stderr)
        extra_body = {}
        if REASONING_EFFORT:
            extra_body["reasoning_effort"] = REASONING_EFFORT
        if OPENROUTER_PROVIDER:
            extra_body["provider"] = {"order": [OPENROUTER_PROVIDER], "allow_fallbacks": False}
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You output only valid JSON, nothing else."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
            max_tokens=max_output_tokens(level, mode),
            extra_body=extra_body,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            total = _add_spend(usage.prompt_tokens, usage.completion_tokens)
        else:
            in_tok, out_tok = estimate_call_tokens(level, mode, user_prompt)
            total = _add_spend(in_tok, out_tok)
        print(f"  [${total:6.3f} total] {domain}/{level}/{mode}", file=sys.stderr)

        raw = resp.choices[0].message.content.strip()
        # Some providers (e.g. Gemma via the Gemini API) inline their chain-of-
        # thought directly in the visible content as a leading <thought>...
        # </thought> block instead of hiding it -- strip it before parsing.
        # No-op for providers that never produce this pattern.
        raw = re.sub(r"^<thought>.*?</thought>\s*", "", raw, count=1, flags=re.DOTALL)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        example = json.loads(raw)

        invalid_reason = validate_example(example, level)
        if invalid_reason:
            print(f"  [skip] {domain}/{level}/{mode}: {invalid_reason}", file=sys.stderr)
            return None

        example["meta"] = {
            "domain": domain,
            "level": level,
            "mode": mode,
            "seed_task": seed_task,
        }
        return example
    except Exception as e:
        print(f"  [skip] {domain}/{level}/{mode}: {e}", file=sys.stderr)
        return None


def run_combo(client: OpenAI, domain: str, level: str, variants: int, workers: int,
              max_spend: float, max_calls: int | None, dry_run: bool):
    seeds = load_seeds(domain)
    mode = mode_for_domain(domain)
    out_path = OUT_DIR / f"{domain}__{level}.jsonl"

    jobs = []
    for seed in seeds:
        for _ in range(variants):
            jobs.append((domain, level, seed, mode))
            if mode != "subagent" and random.random() < SUBAGENT_VARIANT_RATE:
                jobs.append((domain, level, seed, "subagent"))

    print(f"[{domain}/{level}] {len(jobs)} calls queued -> {out_path}")
    written = 0
    skipped_cap = 0
    skipped_error = 0
    f = None if dry_run else open(out_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(call_api, client, d, lv, s, m, max_spend, max_calls, dry_run): (d, lv, s, m)
                for (d, lv, s, m) in jobs
            }
            for fut in as_completed(futures):
                example = fut.result()
                if example == "CAPPED":
                    skipped_cap += 1
                elif example is None:
                    skipped_error += 1
                elif dry_run:
                    written += 1
                else:
                    f.write(json.dumps(example) + "\n")
                    written += 1
    finally:
        if f:
            f.close()

    if dry_run:
        print(f"[{domain}/{level}] would generate {written} examples")
    else:
        msg = f"[{domain}/{level}] wrote {written}/{len(jobs)} examples"
        details = []
        if skipped_cap:
            details.append(f"{skipped_cap} skipped, spend/call cap reached")
        if skipped_error:
            details.append(f"{skipped_error} skipped, parse/API error")
        if details:
            msg += "  (" + "; ".join(details) + ")"
        print(msg)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--level", choices=LEVELS)
    ap.add_argument("--all", action="store_true", help="run every domain x level combo")
    ap.add_argument("--variants", type=int, default=3, help="generations per seed task")
    ap.add_argument("--workers", type=int, default=6, help="concurrent API calls")
    ap.add_argument("--max-spend", type=float, default=2.0,
                     help="hard stop, in USD, using the real per-call cost from the "
                          "API's usage stats. The script checks this before every call, "
                          "so actual spend may overshoot slightly (up to ~--workers calls "
                          "worth) since in-flight calls aren't cancelled mid-request.")
    ap.add_argument("--max-calls", type=int, default=None,
                     help="hard stop on total API call ATTEMPTS (not just successes), "
                          "independent of --max-spend. For providers that charge $0 but "
                          "enforce a request-count quota instead (e.g. OpenRouter free-tier "
                          "models, which count failed attempts against the quota too).")
    ap.add_argument("--dry-run", action="store_true",
                     help="estimate cost with ZERO API calls made, using worst-case "
                          "token counts per level. Run this first.")
    args = ap.parse_args()

    if args.max_calls is not None and args.max_calls < 1:
        print("--max-calls must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and not API_KEY:
        print("Set IMG2_API_KEY first (or use --dry-run, which needs no key).", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY or "dry-run", base_url=API_BASE)

    total = 0
    t0 = time.time()
    if args.all:
        for domain in DOMAINS:
            for level in LEVELS:
                total += run_combo(client, domain, level, args.variants, args.workers,
                                    args.max_spend, args.max_calls, args.dry_run)
    else:
        if not (args.domain and args.level):
            print("Pass --domain and --level, or --all.", file=sys.stderr)
            sys.exit(1)
        total += run_combo(client, args.domain, args.level, args.variants, args.workers,
                            args.max_spend, args.max_calls, args.dry_run)

    with _spend_lock:
        final_spend = _spend_state["usd"]

    if args.dry_run:
        print(f"\n[DRY RUN] {total} examples would be generated. "
              f"Worst-case estimated cost: ${final_spend:.2f}")
        print("This is a ceiling (assumes every call maxes out its token budget), "
              "real cost is usually lower. Re-run without --dry-run when ready.")
    else:
        with _call_lock:
            final_calls = _call_state["count"]
        msg = f"\nDone. {total} examples written, ${final_spend:.2f} spent, " \
              f"{final_calls} API calls made, {time.time()-t0:.0f}s. " \
              f"Spend cap ${args.max_spend:.2f}"
        if args.max_calls is not None:
            msg += f", call cap {args.max_calls}"
        print(msg + ".")


if __name__ == "__main__":
    main()
