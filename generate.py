#!/usr/bin/env python3
"""
Generate SFT training examples for Imagination 2.1 Pro.

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
    ULTRA_DOMAINS,
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
    return _add_actual_spend(cost)


def _add_actual_spend(cost: float) -> float:
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

# Fraction of high/max direct-or-subagent-mode examples generated as a
# "no good tool applies" negative -- live testing against the deployed
# low+medium+high model found it collapses into emitting the base model's
# native <tool_call> token instead of the trained <think> convention when a
# real request has no relevant tool available, almost certainly because
# every prior generated example invents 1-2 tools regardless of whether the
# task needs one (see build_prompt below), so the model never saw a genuine
# "just answer, don't call anything" scenario at these effort levels. NOT
# applied via ordinary NO_TOOL_VARIANT_RATE sampling to "ultra" (its main
# scenario is tool-call-heavy verify/iterate) or "orchestrator" mode (which
# by definition always delegates via run_subagent) -- ultra DOES get a
# no-tool variant, but only as a deliberate --force-no-tool backfill pass
# run separately after the main tool-using generation, never mixed in via
# random sampling (see roll_no_tool below and the level == "ultra" branches
# in build_prompt / validate_example for what that variant actually requires:
# self-correction via reasoning, not a tool-revealed failure).
NO_TOOL_VARIANT_RATE = 0.2
NO_TOOL_LEVELS = ("high", "max")


def load_seeds(domain: str) -> list[str]:
    with open(SEEDS_DIR / f"{domain}.json") as f:
        return json.load(f)


def build_prompt(domain: str, level: str, seed_task: str, mode: str, no_tool: bool = False) -> tuple[str, str]:
    system_prompt = build_system_prompt(mode, level)

    if mode == "orchestrator":
        tool_instructions = (
            f"The ONLY tool you may call is run_subagent, schema: "
            f"{json.dumps(RUN_SUBAGENT_TOOL_SCHEMA)}. Do not invent or call any other tools."
        )
    elif no_tool:
        tool_instructions = (
            "This specific example must NOT involve any tool call. Pick ONE of "
            "these two patterns (vary which one across different examples):\n"
            "(a) Don't mention any tool schema at all -- the task is answerable "
            "from reasoning/knowledge alone, so the assistant just thinks (if "
            "the effort level calls for it) and answers directly.\n"
            "(b) Invent exactly ONE tool that is clearly plausible for this "
            "domain but genuinely irrelevant to THIS specific task -- the "
            "assistant must recognize it doesn't apply and answer directly "
            "WITHOUT calling it, not force a call to it anyway.\n"
            "The final assistant message must contain no tool_calls at all."
        )
    else:
        tool_instructions = (
            "Invent 1-2 realistic tools appropriate to the task (e.g. fetch, shell, "
            "file_search, sql_query) with sensible JSON parameter schemas."
        )

    ultra_structural_note = ""
    if level == "ultra" and no_tool:
        # A genuinely tool-free ultra example -- the persistence/verify-then-
        # fix pattern still has to show real self-correction, it just can't
        # come from a tool result (there are none). It has to come from the
        # assistant catching a genuine flaw in its OWN reasoning or a stale
        # assumption it made, and explicitly correcting it, before the final
        # answer -- not tool-revealed, but not skippable either.
        ultra_structural_note = """
STRUCTURAL REQUIREMENT for this no-tool ultra variant: because this example
must not call any tool (see above), the required self-correction has to
come from the assistant's OWN reasoning, not a tool result. Somewhere after
the initial <think> block (either within it, or in a visible second pass),
the assistant must catch a genuine flaw in its own prior reasoning or a
stale/wrong assumption it made -- state the correction explicitly (e.g.
"Wait, that's not right because...", "Actually, reconsidering...") -- and
then give the corrected final answer. A clean single-pass answer with no
visible self-caught error is NOT a valid example of this tier and will be
rejected -- don't generate one. Do NOT fake this by inventing a pointless
error just to check a box; the corrected mistake must be a realistic one
someone could plausibly make on this specific task."""
    elif level == "ultra":
        ultra_structural_note = """
STRUCTURAL REQUIREMENT for this tier: the conversation MUST include at
least one tool call whose result reveals something wrong or incomplete
(a failing test, an error, unexpected/incorrect output, a stale
assumption disproven by what the tool actually returned) -- and the
assistant must notice it and take a further action to investigate or fix
it before the final answer. A conversation where every tool call succeeds
cleanly on the first try, with no detected problem, is NOT a valid example
of this tier and will be rejected -- don't generate one."""

    generator_instructions = f"""
You are generating ONE synthetic training example for finetuning a coding/
agentic assistant. The example should teach the DOMAIN "{domain}" at
reasoning level "{level}" in {mode.upper()} mode.

Seed task (use as inspiration, do not restate verbatim, make it concrete
and specific with real details): "{seed_task}"

The system prompt for this example must be exactly:
\"\"\"{system_prompt}\"\"\"

{tool_instructions}
{ultra_structural_note}

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


def validate_example(example: dict, level: str, no_tool: bool = False) -> str | None:
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
    for m in messages:
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            return f"message has invalid role {m.get('role')!r}"

    # A valid conversation only ever alternates assistant-with-tool_calls with
    # its matching tool-result, ending in one final plain assistant message.
    # Two assistant messages back to back should never happen -- when it does,
    # it's usually the model narrating a fake tool call in prose ("I'll call
    # file_search with a query...") instead of emitting real tool_calls, then
    # putting what should be a tool-role result into a second assistant
    # message. That teaches exactly the wrong tool-use pattern, so reject it.
    for i in range(len(messages) - 1):
        if messages[i].get("role") == "assistant" and messages[i + 1].get("role") == "assistant":
            return "two consecutive assistant messages (likely narrated a fake tool call in prose)"

    # A message with neither content nor tool_calls renders as nothing --
    # the chat template does unconditional string concatenation on content
    # for any message not in its tool_calls branch, so a None/missing
    # content with no tool_calls crashes template rendering at train time
    # instead of just being a wasted turn.
    for m in messages:
        if not m.get("content") and not m.get("tool_calls"):
            return f"message with role {m.get('role')!r} has neither content nor tool_calls"

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
    if level in ("medium", "high", "max"):
        if not content.lstrip().startswith("<think>"):
            return f"level is {level} but final answer doesn't start with a <think> block"
        think_match = re.match(r"^<think>(.*?)</think>", content.lstrip(), flags=re.DOTALL)
        if not think_match:
            return f"level is {level} but <think> block is never closed with </think> (truncated)"
        after_think = content.lstrip()[think_match.end():].strip()
        if len(after_think) < 10:
            return f"level is {level} but there's no real answer after the <think> block (truncated)"
        # high/max use a labeled structure (not just open prose) -- an LLM-
        # judge audit found ~39% of open-prose "high" examples silently
        # skipped naming a real alternative approach despite the prose
        # instruction saying to; a calibration test comparing prose vs this
        # labeled scaffold on identical seeds found 100% label compliance
        # for the scaffold vs ~12-50% for prose (see PLAN.md). Enforcing the
        # labels here means a miss triggers a retry instead of silently
        # shipping a non-compliant example. Labels must start a line (not
        # just appear as a substring anywhere, e.g. inside an example or
        # code snippet the model quotes) -- MULTILINE anchors "^" to each
        # line start, not just the string start.
        think_text = think_match.group(1)
        if level == "high":
            required = ["Problem:", "Alternative considered:", "Rejected because:", "Chosen because:"]
            missing = [r for r in required
                       if not re.search(rf"^{re.escape(r)}", think_text, flags=re.MULTILINE)]
            if missing:
                return f"level is high but <think> block is missing required label(s): {missing}"
        if level == "max":
            required = ["Problem:", "Alternative 1 considered:", "Alternative 2 considered:",
                        "Chosen approach:", "Why this wins:"]
            missing = [r for r in required
                       if not re.search(rf"^{re.escape(r)}", think_text, flags=re.MULTILINE)]
            if missing:
                return f"level is max but <think> block is missing required label(s): {missing}"
            n_rejected = len(re.findall(r"^Rejected because:", think_text, flags=re.MULTILINE))
            if n_rejected < 2:
                return "level is max but <think> block doesn't reject both alternatives"

    if level == "ultra":
        # BUG (found via LLM-judge audit on the first real ultra batch,
        # ~15% pass rate / avg level_match 1.95): this used to be a
        # nested "if level == 'ultra':" INSIDE the "if level in (medium,
        # high, max):" block above -- since "ultra" is never a member of
        # that tuple, the whole check was dead code and never ran. That
        # let ultra's <think> block requirement go completely
        # unenforced, including its placement: unlike medium/high/max
        # (typically single-turn, so checking the LAST message works),
        # ultra's system prompt requires the <think> block BEFORE
        # acting -- i.e. in the FIRST assistant message, before any
        # tool_calls -- not the last one. The judge's #1 complaint was
        # examples with the think block only at the end, after all tool
        # use, which is exactly what an unenforced check would produce.
        first_assistant = next((m for m in messages if m.get("role") == "assistant"), None)
        if first_assistant is None:
            return "level is ultra but no assistant message found"
        first_content = first_assistant.get("content") or ""
        if not isinstance(first_content, str) or not first_content.lstrip().startswith("<think>"):
            return ("level is ultra but the FIRST assistant message doesn't start with a "
                    "<think> block (it must come before acting, not after)")
        first_think_match = re.match(r"^<think>(.*?)</think>", first_content.lstrip(), flags=re.DOTALL)
        if not first_think_match:
            return "level is ultra but the first assistant message's <think> block is never closed with </think>"
        first_think_text = first_think_match.group(1)
        required = ["Problem:", "Plan:", "Verification strategy:"]
        missing = [r for r in required
                   if not re.search(rf"^{re.escape(r)}", first_think_text, flags=re.MULTILINE)]
        if missing:
            return f"level is ultra but the first <think> block is missing required label(s): {missing}"
        # NOT applying medium/high/max's "len(after_think) < 10" check here
        # -- for a tool-using ultra example the first message legitimately
        # has little/no prose after </think> (the "action" is the
        # tool_calls on that same message, not prose); for the no-tool
        # variant this first message IS the whole answer, already covered
        # by the last-message content-length check earlier in this function.

    if level == "ultra" and no_tool:
        # The no-tool ultra variant (deliberate --force-no-tool backfill
        # only, see NO_TOOL_LEVELS/roll_no_tool -- never mixed into the main
        # run via random sampling): self-correction must come from the
        # assistant's own reasoning, not a tool result, and there must be NO
        # tool calls at all (a "no tool" example that slipped one in defeats
        # the point and would teach a contradictory pattern).
        tool_call_count = sum(len(m.get("tool_calls") or []) for m in messages)
        if tool_call_count > 0:
            return "level is ultra (no_tool variant) but contains tool_calls -- should have none"
        correction_markers = re.compile(
            r"\b(wait,?|actually,?|on second thought|hold on|let me "
            r"reconsider|reconsidering|re-?check(ing)?|that'?s not right|"
            r"I was (wrong|mistaken)|doesn'?t (hold|work)|flaw in|correcting "
            r"(myself|that)|scratch that)\b", re.IGNORECASE,
        )
        assistant_text = "\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "assistant"
        )
        if not correction_markers.search(assistant_text):
            return "level is ultra (no_tool variant) but no visible self-correction language found"
    elif level == "ultra":
        # Structural requirement (see PLAN.md): a clean one-pass success with
        # no self-correction isn't a valid ultra example -- the whole point
        # of this tier is verify-then-fix, not just verify. Heuristic: at
        # least 2 tool calls (one that surfaces a problem, one that responds
        # to it), and at least one tool-role result whose content looks
        # failure/error-shaped, with the conversation continuing past it
        # rather than ending immediately after.
        tool_call_count = sum(len(m.get("tool_calls") or []) for m in messages)
        if tool_call_count < 2:
            return "level is ultra but has fewer than 2 tool calls (no room for a failure+correction cycle)"
        failure_markers = re.compile(
            r"\b(error|exception|traceback|fail(ed|ure)?|incorrect|unexpected|"
            r"assertionerror|0 passed|not found|mismatch|invalid)\b", re.IGNORECASE,
        )
        flagged_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "tool" and failure_markers.search(str(m.get("content") or ""))
        ]
        if not flagged_indices:
            return "level is ultra but no tool result looks like it surfaced a failure/problem"
        last_flagged = flagged_indices[-1]
        if last_flagged >= len(messages) - 2:
            return "level is ultra but the conversation ends right after the flagged failure (no visible correction)"

    # every tool_call must be immediately followed by its matching tool result,
    # and must actually name a function -- a tool_call with arguments but no
    # function.name is unusable (nothing to dispatch) and json.loads happily
    # accepts it since it's still valid JSON, so this needs an explicit check.
    for i, m in enumerate(messages):
        calls = m.get("tool_calls") or []
        # A message can carry multiple tool_calls (OpenAI-style parallel
        # calls); each must have a matching "role": "tool" result somewhere
        # in the next len(calls) messages, not necessarily at i+1 -- checking
        # only messages[i+1] would reject every valid multi-call message
        # after the first.
        following_tool_ids = {
            messages[j].get("tool_call_id")
            for j in range(i + 1, min(i + 1 + len(calls), len(messages)))
            if messages[j].get("role") == "tool"
        }
        for call in calls:
            call_id = call.get("id")
            fn = call.get("function") or {}
            if not fn.get("name"):
                return f"tool_call {call_id!r} is missing function.name"
            if "arguments" not in fn:
                return f"tool_call {call_id!r} is missing function.arguments"
            if not isinstance(fn["arguments"], str):
                # Must be a JSON-encoded string per the OpenAI tool-calling
                # format, not a raw object -- models occasionally emit the
                # object directly. json.loads doesn't catch this (a dict is
                # still valid JSON), but it silently breaks anything that
                # assumes a string here, including datasets.Dataset.from_list
                # at train time (PyArrow needs one consistent struct schema
                # across every message, and a str-vs-dict split fails that).
                return f"tool_call {call_id!r} has non-string function.arguments ({type(fn['arguments']).__name__})"
            try:
                if not isinstance(json.loads(fn["arguments"]), dict):
                    return f"tool_call {call_id!r} function.arguments doesn't decode to a JSON object"
            except json.JSONDecodeError:
                # A string that isn't itself valid JSON -- usually an
                # under-escaped backslash inside a regex/command/path (e.g.
                # \K written instead of \\K). Downstream chat-template
                # rendering needs to json.loads this back into a dict, so it
                # has to actually be valid JSON, not just any string.
                return f"tool_call {call_id!r} function.arguments isn't valid JSON"
            if call_id not in following_tool_ids:
                return f"tool_call {call_id!r} has no matching tool result message"

    return None


# Retry temperatures for a single seed/domain/level/mode job. Some providers
# occasionally emit a premature stop (finish_reason="stop" well under the
# token budget) mid-JSON-string on long, nested-content generations -- not a
# token-budget problem, a one-off model quirk. Lowering temperature on retry
# reduces how often that recurs; each attempt claims its own call slot and
# counts against max_spend/max_calls like any other call.
RETRY_TEMPERATURES = [0.9, 0.7, 0.5]


def call_api(client: OpenAI, domain: str, level: str, seed_task: str, mode: str,
             max_spend: float, max_calls: int | None, dry_run: bool, no_tool: bool = False) -> dict | str | None:
    system_prompt, user_prompt = build_prompt(domain, level, seed_task, mode, no_tool=no_tool)

    if dry_run:
        in_tok, out_tok = estimate_call_tokens(level, mode, user_prompt)
        total = _add_spend(in_tok, out_tok)
        return {"_dry_run": True, "est_total_usd": total}

    last_err = None
    for attempt, temperature in enumerate(RETRY_TEMPERATURES, start=1):
        if _over_cap(max_spend) or not _try_claim_call(max_calls):
            return "CAPPED"  # cap already hit, don't make the call

        try:
            if max_calls is not None:
                with _call_lock:
                    call_num = _call_state["count"]
                retry_tag = f" (retry {attempt - 1})" if attempt > 1 else ""
                print(f"  [call {call_num}/{max_calls}] {domain}/{level}/{mode}{retry_tag}", file=sys.stderr)
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
                temperature=temperature,
                max_tokens=max_output_tokens(level, mode),
                extra_body=extra_body,
            )
            usage = getattr(resp, "usage", None)
            # OpenRouter reports the real, authoritative per-call dollar cost as
            # usage.cost (a non-standard field the openai SDK still exposes via
            # its pydantic extra="allow" passthrough). Prefer that over our own
            # price-per-token estimate when it's present -- it can't drift from
            # whatever OpenRouter actually billed, unlike a hardcoded rate that
            # might be stale for a specific pinned provider/route. Direct
            # provider APIs (e.g. DeepSeek's own endpoint) don't return this
            # field, so this falls back to the token-based estimate for those.
            actual_cost = getattr(usage, "cost", None) if usage else None
            if actual_cost is not None:
                total = _add_actual_spend(float(actual_cost))
            elif usage:
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
            try:
                example = json.loads(raw)
            except json.JSONDecodeError:
                # Code-heavy answers (general_code especially) often contain
                # a real, literal newline inside a JSON string value where
                # the model should have written \n -- strict mode treats
                # that as an invalid control character and, depending on
                # where it falls, can cascade into "Unterminated string"
                # once the parser goes hunting for the real closing quote.
                # strict=False accepts literal control characters in
                # strings, recovering this specific (common) case for free
                # -- same fallback pattern already used in train.py/
                # dedup_and_split.py's _try_parse_dict for the analogous
                # tool_call-arguments issue. Doesn't recover a genuinely
                # unescaped quote character breaking the string boundary
                # itself -- that's still unrecoverable without guessing.
                example = json.loads(raw, strict=False)

            invalid_reason = validate_example(example, level, no_tool)
            if invalid_reason:
                raise ValueError(invalid_reason)

            example["meta"] = {
                "domain": domain,
                "level": level,
                "mode": mode,
                "seed_task": seed_task,
                "no_tool_variant": no_tool,
            }
            return example
        except Exception as e:
            last_err = e
            continue

    print(f"  [skip] {domain}/{level}/{mode}: {last_err}", file=sys.stderr)
    return None


def run_combo(client: OpenAI, domain: str, level: str, variants: int, workers: int,
              max_spend: float, max_calls: int | None, dry_run: bool, force_no_tool: bool = False):
    seeds = load_seeds(domain)
    mode = mode_for_domain(domain)
    out_path = OUT_DIR / f"{domain}__{level}.jsonl"

    def roll_no_tool(job_mode: str) -> bool:
        if job_mode == "orchestrator":
            return False
        if force_no_tool:
            # Targeted backfill mode (--force-no-tool): every eligible job
            # is a no_tool variant, instead of the usual NO_TOOL_VARIANT_RATE
            # sampling -- for topping up a specific domain's negative-example
            # coverage without needing ~5x the volume to hit the normal rate
            # by chance. "ultra" is deliberately excluded from ordinary
            # NO_TOOL_LEVELS sampling (see that constant's comment) but IS
            # allowed here -- --force-no-tool is always an explicit,
            # separate backfill invocation, never mixed into a main run.
            return level in NO_TOOL_LEVELS or level == "ultra"
        return level in NO_TOOL_LEVELS and random.random() < NO_TOOL_VARIANT_RATE

    jobs = []
    for seed in seeds:
        for _ in range(variants):
            jobs.append((domain, level, seed, mode, roll_no_tool(mode)))
            if mode != "subagent" and random.random() < SUBAGENT_VARIANT_RATE:
                jobs.append((domain, level, seed, "subagent", roll_no_tool("subagent")))

    print(f"[{domain}/{level}] {len(jobs)} calls queued -> {out_path}")
    written = 0
    skipped_cap = 0
    skipped_error = 0
    f = None if dry_run else open(out_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(call_api, client, d, lv, s, m, max_spend, max_calls, dry_run, nt): (d, lv, s, m)
                for (d, lv, s, m, nt) in jobs
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
    ap.add_argument("--force-no-tool", action="store_true",
                     help="make EVERY eligible job (high/max, non-orchestrator) a "
                          "no_tool variant instead of sampling at NO_TOOL_VARIANT_RATE. "
                          "For targeted backfill of a domain's negative-example coverage "
                          "without needing ~5x the volume to hit the normal rate by chance.")
    args = ap.parse_args()

    if args.max_calls is not None and args.max_calls < 1:
        print("--max-calls must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and not API_KEY:
        print("Set IMG2_API_KEY first (or use --dry-run, which needs no key).", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY or "dry-run", base_url=API_BASE)

    if args.level == "ultra" and args.domain and args.domain not in ULTRA_DOMAINS:
        print(f"'ultra' is only generated for {ULTRA_DOMAINS} (see PLAN.md) -- "
              f"'{args.domain}' isn't one of them.", file=sys.stderr)
        sys.exit(1)

    total = 0
    t0 = time.time()
    if args.all:
        for domain in DOMAINS:
            for level in LEVELS:
                if level == "ultra" and domain not in ULTRA_DOMAINS:
                    continue
                total += run_combo(client, domain, level, args.variants, args.workers,
                                    args.max_spend, args.max_calls, args.dry_run, args.force_no_tool)
    else:
        if not (args.domain and args.level):
            print("Pass --domain and --level, or --all.", file=sys.stderr)
            sys.exit(1)
        total += run_combo(client, args.domain, args.level, args.variants, args.workers,
                            args.max_spend, args.max_calls, args.dry_run, args.force_no_tool)

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
