# Imagination 2.1 Pro — Reasoning Level Roadmap

Five reasoning tiers, each trained as its own LoRA adapter stacked on top of
the previous one. This doc is the design reference for all five — lock in
`low`/`medium`/`high` behavior here too (not just the two new tiers) so the
whole ladder is documented in one place as it grows.

## Status as of this writing

| Tier | Dataset | Training |
|---|---|---|
| low + medium | `data/train_low_medium.jsonl` (5,042) / `eval_low_medium.jsonl` (328) | **In progress** — Colab A100, LoRA stage 1, resumed from checkpoint, ~906 total steps |
| high | `data/train_high.jsonl` / `eval_high.jsonl`, plus a live generation run adding more (OpenRouter `deepseek-chat`, capped at $2.10) | Not started — queued as stage 2, pending judge validation of the new examples |
| max | Not started | Planning (this doc) |
| ultra | Not started | Planning (this doc) |

## The two-axis model

Reasoning depth and task-completion persistence are separate dimensions, and
they diverge sharply at the top of the ladder:

- **Reasoning depth** — how much genuine deliberation happens before/during
  acting. Ladder: `low < medium < high < ultra < max`.
- **Tool-call / persistence volume** — how much the model keeps working,
  checking, and iterating rather than stopping at a plausible first attempt.
  Ladder: `low < medium < high < max < ultra`.

`max` maximizes axis 1: think as hard as possible, then act.
`ultra` maximizes axis 2, and axis 2 is what actually wins on real task
correctness — a model that verifies its own output and fixes what's wrong
before declaring done is more reliably correct than one that reasoned hard
once and committed. `ultra` doesn't out-reason `max` per step, but it's the
more practically capable tier because it doesn't stop until the task is
*verifiably* finished, not just *apparently* finished. This is the same
insight behind why agentic coding tools with test-and-iterate loops beat
pure chain-of-thought models on real coding tasks even at comparable base
reasoning strength.

## Level definitions

```python
LEVEL_INSTRUCTIONS = {
    "low": (
        "Do NOT include a <think> block. Go straight "
        "to the action/answer. Keep it tight."
    ),
    "medium": (
        "Include a short <think>...</think> block "
        "(2-5 sentences) covering the key decision points, then the answer."
    ),
    "high": (
        "Before calling any tool or taking any "
        "action, include a thorough <think>...</think> block written as "
        "genuine upfront deliberation -- NOT a summary of what you already "
        "did or are about to report. Inside it: (1) restate the core "
        "problem in your own words, (2) name ONE specific, concrete "
        "alternative approach and give a specific technical reason it was "
        "rejected (not a generic dismissal like 'this could also work but "
        "is more complex'), (3) explain concretely why the chosen approach "
        "is better for this exact situation. After the think block, write "
        "a COMPLETE final answer -- finish any code you start, do not stop "
        "partway through."
    ),
    "max": (
        "This is maximum reasoning effort. Before acting, think as deeply "
        "as the problem genuinely requires -- restate the problem, surface "
        "non-obvious edge cases, seriously weigh at least two real "
        "alternative approaches with concrete technical tradeoffs (not "
        "token-effort padding), and commit to the one best justified. "
        "Reasoning depth is the point here, not tool-call volume -- use "
        "tools as needed, but the quality of the thinking is what this "
        "tier is graded on."
    ),
    "ultra": (
        "This is maximum persistence effort. Solid reasoning per step, but "
        "the defining trait of this tier is that you do NOT stop at a "
        "plausible first attempt -- verify your work with real tool calls "
        "(run the test, check the actual output, re-read the file) before "
        "declaring anything done. This conversation MUST include at least "
        "one place where a check reveals something wrong or incomplete, "
        "and you notice it and fix it -- a clean one-pass success without "
        "any self-correction is not a valid example of this tier. Keep "
        "working, keep checking, until the task is verifiably, actually "
        "finished -- not until it looks finished."
    ),
}
```

## Structural requirements (beyond the instruction text)

Instruction text alone won't reliably produce the right shape of data —
models default to clean success paths unless forced otherwise. The
generator prompt needs explicit structural rules per tier:

- **`ultra`**: MUST include at least one failure/imperfect-result →
  detection → correction cycle. Enforce this in `generate.py`'s prompt
  (not just the system-prompt instruction), and consider a
  `validate_example` check that rejects `ultra` rows with zero tool
  calls after the first one that returns a non-trivial issue (heuristic:
  scan tool results for error-shaped content and require the conversation
  to continue past it, rather than ending immediately after a clean
  result).
- **`max`**: no minimum tool-call count, but the `<think>` block should be
  the longest/densest of any tier. Consider a minimum character/token
  floor on the think block content as a generation-time check, mirroring
  how `high`'s "no <think> block" / "must start with <think>" checks
  already work in `validate_example`.

## Token budgets

Existing (`MAX_TOKENS_BY_LEVEL` in `schema_templates.py`):
`low=500, medium=1000, high=6000` (plus wrapper/orchestrator buffers).

`max` and `ultra` need real empirical calibration before committing numbers
— same lesson as `high`, where 2000 and 3200 both silently truncated
before 6000 proved sufficient. Starting points to test against with a
small batch before any real spend:

- `max`: ~10,000-14,000 tokens (reasoning-dense, fewer cycles)
- `ultra`: ~16,000-20,000 tokens (multiple tool cycles including the
  required failure+retry round-trip)

## Domain scope

- `max`: all three domains (`agentic_tool_use`, `subagent_orchestration`,
  `general_code`), same as `low`/`medium`/`high` today.
- `ultra`: primarily `agentic_tool_use` and `general_code` — both are
  naturally tool-call-heavy and coding-adjacent, which is where the
  verify-and-iterate pattern matters most. Open to extending to
  `subagent_orchestration` later if it turns out useful there too.

## Proposed repo restructuring: per-level folders

Current layout is domain-first, level-suffixed:
`data/raw/<domain>__<level>.jsonl` (e.g. `general_code__high.jsonl`).

Proposed layout, level-first:

```
data/raw/
  low/
    agentic_tool_use.jsonl
    subagent_orchestration.jsonl
    general_code.jsonl
    identity.jsonl
  medium/
    ...
  high/
    ...
  max/
    ...
  ultra/
    ...
```

This makes it trivial to see what exists per tier at a glance, and keeps
each new tier's data physically isolated while it's being built out
(no risk of an in-progress `max` generation run touching files a `high`
rebuild also touches). Requires updating:

- `generate.py`'s `OUT_DIR` / output path logic
- `dedup_and_split.py`'s `RAW_DIR` glob (`RAW_DIR.glob("*.jsonl")` →
  needs to walk the new nested structure)
- `split_by_level.py` (may become redundant once storage is already
  level-first — could simplify to just pointing `train.py` at the right
  subfolder instead of post-hoc splitting)

**Not migrating yet** — `generate.py` is actively writing to the current
flat paths in the live `high` generation run, and stage-1 training is
actively reading from `data/train_low_medium.jsonl`. Restructuring live
paths mid-run risks breaking something for no benefit right now. This
migration is next up once the current `high` run is judged and stage 1
training finishes.

## Full sequence

1. ~~Generate low/medium/high base data~~ (done, low+medium training in progress)
2. Finish current `high` top-up generation run, judge it, report quality
3. Migrate `data/raw/` to per-level folders
4. Calibrate `max`/`ultra` token budgets with a small test batch (no real
   spend until budgets are confirmed not to truncate)
5. Add `max`/`ultra` to `schema_templates.py` (`LEVELS`, instructions,
   token budgets) and `generate.py` (structural requirements, esp.
   `ultra`'s failure-recovery rule)
6. Build/expand seed banks if `max`/`ultra` need domain-specific seed
   tasks beyond what `low`/`medium`/`high` already use
7. Generate `max` dataset (budget-capped, same discipline as every prior
   run)
8. Generate `ultra` dataset (budget-capped)
9. Judge-validate both
10. Train stage 3 (`max`) — merge stage 2's (`high`) adapter into the
    base, train `max` on top
11. Train stage 4 (`ultra`) — merge stage 3's output, train `ultra` on top

## Open decisions

- `ultra` domain scope (proposed: `agentic_tool_use` + `general_code` only)
  — confirm or adjust.
- Whether `subagent_orchestration` needs its own `ultra` treatment given
  orchestrator mode is already inherently multi-step via `run_subagent`
  chains, or whether that's a distinct enough pattern from `ultra`'s
  verify-and-iterate loop that it doesn't apply the same way.
