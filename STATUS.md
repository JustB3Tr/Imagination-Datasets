# Imagination Project — Status / Handoff

Last updated: 2026-09-14, by Claude Code (session `session_01We9LLTYVHVtLtbrTrH5mfd`).

This is a full-state snapshot of where the Imagination project is, across
all three repos, written as a handoff so a fresh session (or a human) can
pick up without re-deriving any of this from scratch.

---

## 1. The three repos

| Repo | Role | State |
|---|---|---|
| `JustB3Tr/Imagination-Datasets` (this repo) | SFT training data generation pipeline | Active — this doc lives here |
| `JustB3Tr/imagination-agent-sandbox` | Real code-execution backend (Docker-out-of-Docker per-user containers) for the model's tool calls | Built, PR'd, merged earlier this project (see §5) |
| `JustB3Tr/imagination-ui` | Next.js chat UI: Supabase auth, conversation persistence, tool dispatch against the sandbox | Built, PR'd, merged earlier this project (see §5) |

This document focuses mainly on **this repo** (the dataset pipeline), since
that's what the current work-in-progress (Imagination 2.2) touches. §5
summarizes the other two repos' state for completeness.

---

## 2. Current branch / PR state

- Active branch: **`claude/ultra-judge-bug-fixes`**
- Open PR: **[#40](https://github.com/JustB3Tr/Imagination-Datasets/pull/40)** — "Imagination 2.2: new domains + versioned prompts + free-tier rate limiting for ox-alpha generation", targeting `main`, `mergeable_state: clean`, **not yet merged**.
- This PR also carries earlier work from the same branch: the ultra-tier judge bug fixes (see §4.4) and the ultra dataset generation (§4.3), which were already complete and stable before 2.2 work started.

**Important**: the repo's "designated" branch in some session configs is
`claude/imagination2-data-gen-hbo5kc`, which is a much older/stale branch —
**do not develop there**. All real, current work is on
`claude/ultra-judge-bug-fixes`. If a fresh session starts on the wrong
branch (this has happened before), run:

```bash
git fetch origin claude/ultra-judge-bug-fixes
git checkout claude/ultra-judge-bug-fixes
```

---

## 3. Imagination 2.1 — status: done, trained, stable

The original dataset. Not touched by any 2.2 work (2.2 writes to entirely
separate files/dirs — see §4.1).

- **Domains** (3): `agentic_tool_use`, `subagent_orchestration`, `general_code`
- **Reasoning levels** (5): `low`, `medium`, `high`, `max` — all trained/merged
  checkpoints exist. `ultra` — dataset exists and is complete (§4.3) but
  **has no trained checkpoint yet** (`schema_templates.py`'s
  `build_system_prompt` raises `ValueError` if you try to send `"ultra"` at
  inference time in the sandbox/UI repos — this is intentional, not a bug).
- **Data files**: `data/train.jsonl` (9095) / `data/eval.jsonl` (631) is the
  main low/medium/high/max/ultra-combined set. There are also separately-staged
  files from the staged-training approach (`train_low_medium.jsonl`,
  `train_high.jsonl`, `train_max.jsonl`, `train_ultra.jsonl` + matching
  `eval_*` files) — see `training/` and the Colab guide referenced in old
  commits for how those get used in staged base+adapter training.
- **System prompt identity string**: `"Imagination 2.1 Pro"`, byte-for-byte
  identical across three places (this is load-bearing — the model was
  fine-tuned on this exact text):
  - `Imagination-Datasets/schema_templates.py` (source of truth)
  - `imagination-agent-sandbox/orchestrator/prompt.py`
  - `imagination-ui/src/lib/prompt.ts`

---

## 4. Imagination 2.2 — status: IN PROGRESS, blocked on funding

### 4.1 What 2.2 adds

Same idea as 2.1, additive, not a replacement:

- **All 5 reasoning levels** (low/medium/high/max/ultra), same as 2.1.
- **Same 3 domains** as 2.1, **plus 3 new ones** (user-selected via
  `AskUserQuestion` earlier this session):
  - `multi_file_project` — tasks spanning multiple files/paths in one
    sandbox session, exercising the real multi-file coordination the
    sandbox repo now actually supports.
  - `long_context_reasoning` — tasks with a large pasted artifact (log
    dump, config, diff) the model must search/reason over. Deliberately
    **excluded from `ultra`** (comprehension skill, not an iterate-until-
    fixed one — same reasoning that excluded `subagent_orchestration`
    from `ultra` originally).
  - `conversational_followup` — multi-turn tasks where the user corrects/
    redirects mid-task ("actually use Postgres not Mongo"), training
    recovery from changed requirements.
- **System prompt identity**: `"Imagination 2.2 Pro"` — a NEW,
  independently-versioned string. `schema_templates.py`'s
  `build_system_prompt(mode, level, version="2.1")` takes a `version` param;
  default `"2.1"` reproduces the exact original byte-for-byte contract
  everywhere it's already called (verified — zero behavior change), so
  passing nothing keeps existing 2.1 callers safe.
- **Output isolation**: 2.2 raw generations go to `data/raw_2_2/`, never
  `data/raw/`. Controlled by `generate.py --version 2.2` (or `IMG2_OUT_DIR_2_2`
  env var to override the directory directly). 2.1's pipeline is completely
  untouched by any of this.

Key constants (all in `schema_templates.py`):
```python
DOMAINS_2_2       = DOMAINS + ["multi_file_project", "long_context_reasoning", "conversational_followup"]
ULTRA_DOMAINS_2_2 = ULTRA_DOMAINS + ["multi_file_project", "conversational_followup"]  # NOT long_context_reasoning
DOMAIN_HINTS      = {...}  # extra generator guidance for the 3 new domains' distinct shapes
```

### 4.2 The provider saga (READ THIS before touching the model config)

This is the part most likely to trip up a fresh session, so it's detailed:

1. **Started on `stealth/ox-alpha`** (OpenRouter), a free anonymous
   "stealth" preview model, because it was free and benchmarking well.
2. **Found a serious bug**: this model (and its serving stack) treats a
   literal `<think>...</think>` span *anywhere in the output* as a real
   architectural control sequence — the serving layer **silently strips it
   out of `content` and reroutes it into the separate `reasoning` API
   field**, regardless of whether it's the model's own hidden reasoning or
   text we explicitly asked it to reproduce verbatim. Since every
   medium/high/max/ultra example's entire training signal IS a literal,
   visible `<think>` block, this would have silently corrupted every such
   example.
3. **Fix**: `IMG2_THINK_PLACEHOLDER=1` env var. When set, `generate.py`:
   - Tells the generator to write a placeholder sentinel instead of the
     real tags anywhere a `<think>`/`</think>` tag would appear (both in
     the reproduced system-prompt text AND the assistant's own think
     block).
   - Substitutes the placeholder back to literal `<think>`/`</think>` on
     the raw response, BEFORE JSON parsing.
   - The sentinels are **collision-resistant internal markers**, not plain
     brackets — a PR review (JustB3Tr) correctly flagged that generic
     `[THINK]`/`[/THINK]` could theoretically collide with real dataset
     content (code, logs, pasted config — especially relevant given
     `long_context_reasoning` exists now). Current sentinels:
     ```
     __IMG2_INTERNAL_THINK_OPEN_7F31A9__
     __IMG2_INTERNAL_THINK_CLOSE_7F31A9__
     ```
   - **Verified working end-to-end** on real generated examples (zero
     corruption, zero leakage, checked programmatically) — see the PR #40
     comment thread for the full calibration checklist and results.
4. **`stealth/ox-alpha`'s free preview ENDED mid-project** (discovered
   2026-09-13/14, roughly 3-4 weeks after its ~2026-08-20 launch — stealth
   previews on OpenRouter are typically ~1-2 week windows). The model ID now
   returns `404`, with the error body identifying it as **ZAI's
   GLM-5.3-Flash**, now live at `z-ai/glm-5.3-flash` — **paid, not free**
   ($0.075/1M input, $0.25/1M output tokens — still very cheap, roughly
   $0.0005-0.001/call observed in practice). User explicitly chose to
   switch to this paid model rather than chase another free stealth model.
   **`reasoning: {"mandatory": true}`** on this model — it always runs
   hidden reasoning (efforts: low/high/max, no "none" to disable), so the
   `<think>`-interception behavior and the placeholder workaround are still
   both necessary and were re-verified against the real named model, not
   just the preview.
5. **Rate limiting / error handling built into `generate.py`** to survive
   this provider's quirks (all opt-in via env vars, zero effect on other
   providers like DeepSeek):
   - `IMG2_MIN_CALL_INTERVAL` / `--min-interval`: minimum seconds between
     the START of any two API calls, shared globally across all worker
     threads (a lock-based throttle, not per-worker) — needed to stay under
     a provider's requests-per-minute cap.
   - **429 handling**: backs off `15s * attempt_number` before retrying, on
     any error containing `"429"` or `"rate"`.
   - **402 `in_flight_budget_exhausted` handling** (found live, fixed in
     commit `7b0876a`): OpenRouter reserves *worst-case* cost per in-flight
     request against account balance, so concurrent `--workers` can each
     get rejected even though actual spend is tiny. The API's own
     `Retry-After` header said 120s — much longer than the 429 backoff — so
     this gets its own longer (130s) backoff instead of hammering the same
     wall on every retry attempt.
   - **These backoffs do NOT help if the account is actually out of money**
     (see §4.5) — that surfaces as a *different* 402 message
     (`"This request requires more credits..."` / `"maximum cost exceeds
     your available credits"`), which no amount of backoff fixes. Check
     real balance (see §4.5's curl command) before assuming a 402 is
     transient.

### 4.3 Ultra tier for 2.1 (already done, separate from the 2.2 work above)

Fully generated and merged before 2.2 work started:
- `data/train_ultra.jsonl` (262) / `data/eval_ultra.jsonl` (30) — 292 total,
  19.9% no-tool-variant coverage (deliberately downsampled from 55.3% down
  to ~20% — a genuine no-tool "ultra" variant exists so the model has a
  sane fallback if a user leaves reasoning_effort on ultra but asks
  something simple, without that becoming the majority pattern).
- `LEVEL_INSTRUCTIONS["ultra"]` was reworded mid-project to fix a real
  prompt/behavior contradiction (the original wording implicitly demanded
  tool-based verification even for genuinely tool-free tasks) — already
  fixed, already in `schema_templates.py`.

### 4.4 Judge pipeline bugs (already fixed)

Two real bugs found auditing the first ultra judge run, both fixed and
merged:
1. `generate.py`'s `validate_example()` had a **dead code path** — an
   `ultra`-specific think-block check was nested inside an
   `if level in ("medium","high","max")` block, so `"ultra"` (never a
   member of that tuple) never actually got checked. Fixed with a new,
   correctly-scoped top-level check.
2. `judge_dataset.py`'s `format_example_for_judge()` had an **if/else** that
   showed `tool_calls` OR `content`, never both — silently hiding every
   ultra `<think>` block from the judge whenever a message also had
   `tool_calls` (ultra's normal, correct shape). Fixed to show both
   independently.

### 4.5 CURRENT BLOCKER: OpenRouter account is out of credits

```
$ curl -s -H "Authorization: Bearer $IMG2_API_KEY" "https://openrouter.ai/api/v1/credits"
{"data":{"total_credits":25,"total_usage":24.995993719}}
```

**$25 total lifetime credits, $24.996 spent. ~$0.004 left.** This is the
account's full accumulated spend across the whole project (2.1's DeepSeek-
via-OpenRouter generation + this session's 2.2 work), not just tonight.

**To resume**: add credits at https://openrouter.ai/settings/credits (even
$10-20 would very likely finish the entire 2.2 dataset at observed per-call
cost), then re-run the exact command in §4.7.

**The API key itself is deliberately NOT included in this document or
anywhere else in this git history** — this repo is **public**
(`visibility: "public"` confirmed via the GitHub API), and a real,
still-valid credential committed to a public repo gets scraped and abused
by bots within minutes, refunded or not. The key lives in the local `.env`
file (gitignored, never committed) in whatever container/session is
running this — if a fresh session doesn't have it, the account owner needs
to paste it in again (as happened this session) or the local `.env` needs
to be restored/synced some other way. If you're that fresh session and you
don't have `.env`: ask the user for the key rather than assuming you can
find it somewhere in git — it was never there.

### 4.6 What's actually been generated so far (2.2)

All in `data/raw_2_2/` (committed — see the `.gitignore` note explaining
why this is committed unlike `data/raw/`), plus the deduped/split output:

| Domain | low | medium | high | max | ultra |
|---|---|---|---|---|---|
| `agentic_tool_use` | 131 | 129 | 141 | 94 | 42 |
| `subagent_orchestration` | 43 (orchestrator+subagent mode) | — | — | — | — |
| `general_code` | — | — | — | — | — |
| `multi_file_project` | — | — | — | — | — |
| `long_context_reasoning` | — | — | — | — | n/a |
| `conversational_followup` | — | — | — | — | — |

**580 raw examples total.** After `dedup_and_split.py` (see §4.8): **572
kept (8 near-duplicates dropped, 1.4%), split into 519 train / 53 eval**
→ `data/train_2_2.jsonl` / `data/eval_2_2.jsonl` / `data/dedup_report_2_2.json`.

Remaining work once funded: `subagent_orchestration` (medium/high/max —
`ultra` is not applicable to this domain, see §4.1) + all 5 domains/levels
for `general_code`, `multi_file_project`, `long_context_reasoning` (4
levels, no ultra), `conversational_followup`.

### 4.7 Exact command to resume generation

Once `.env` has a funded `IMG2_API_KEY` (OpenRouter):

```bash
cd Imagination-Datasets
pip install -r requirements.txt   # openai, sentence-transformers, numpy
set -a && source .env && set +a
export IMG2_MODEL="z-ai/glm-5.3-flash" IMG2_THINK_PLACEHOLDER=1 IMG2_REASONING_EFFORT=low

# Check real balance first -- don't assume, this bit the project once already:
curl -s -H "Authorization: Bearer $IMG2_API_KEY" "https://openrouter.ai/api/v1/credits"

# Resume ONLY the remaining domains (do not use bare --all, it would
# restart from agentic_tool_use and re-spend on work already done):
for domain in subagent_orchestration general_code multi_file_project long_context_reasoning conversational_followup; do
  for level in low medium high max ultra; do
    # skip ultra for domains not in ULTRA_DOMAINS_2_2 (long_context_reasoning only, currently)
    if [ "$level" == "ultra" ] && [ "$domain" == "long_context_reasoning" ]; then continue; fi
    python3 generate.py --version 2.2 --domain "$domain" --level "$level" \
      --variants 2 --workers 2 --min-interval 3 --max-calls 300 --max-spend 2.00
  done
done
```

`--workers 2` and `--min-interval 3` are deliberately conservative given
this provider's in-flight-budget quirk (§4.2 point 5) — could likely go
higher once there's real balance headroom to test against, but verify with
a small batch first rather than assuming.

### 4.8 Regenerating the train/eval split after more data comes in

```bash
python3 dedup_and_split.py --raw-dir data/raw_2_2 --suffix _2_2 --eval-per-bucket 10
```

(`--raw-dir` and `--suffix` are new flags added this session specifically
for 2.2 — defaults are unchanged so plain `python3 dedup_and_split.py` with
no args still does exactly the original 2.1 behavior, writing
`train.jsonl`/`eval.jsonl`.)

### 4.9 Open PR review items (PR #40)

Copilot's automated review flagged one item, already fixed: `run_combo()`
originally hardcoded 2.2's output dir instead of respecting `IMG2_OUT_DIR`
the way 2.1 does — fixed in commit `713172d` (`OUT_DIR_2_2` now defaults to
a sibling of wherever `OUT_DIR` resolves to, with its own
`IMG2_OUT_DIR_2_2` override).

JustB3Tr's review requested a live calibration gate before scaling — done,
results posted to the PR thread (2026-09-14): 6/7 go/no-go checks passed
with hard evidence (zero malformed JSON, zero `<think>` corruption, zero
sentinel leakage, zero branding mixups, zero out-of-dir writes, 92%
structural acceptance, no persistent 429 loop). **The one open item: a
formal automated judge pass** (`judge_dataset.py`) hasn't been run against
2.2 data yet — the session's Gemini judge key wasn't available. Manual
reading of several examples showed strong quality (realistic infra
scenarios, coherent reasoning, correct think-block structure), but that's
not equivalent to the real judge script. **Next session with a judge-model
key available should run**:
```bash
export IMG2_API_KEY="<gemini-key>"
export IMG2_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai"
export IMG2_MODEL="gemini-3.5-flash-lite"
python3 judge_dataset.py --input data/train_2_2.jsonl --samples-per-bucket 10 --output judge_results_2_2.jsonl
```

---

## 5. Other repos (summary, not actively worked on this session)

### `imagination-agent-sandbox`
Real code-execution backend. Docker-out-of-Docker pattern: an
`orchestrator-server` mounts `/var/run/docker.sock` to manage one
persistent per-user container (keyed `sandbox-<supabase-user-id>`, not
per-conversation) with a named volume at `/workspace` that survives
container recreation. Exposes `execute_shell`/`read_file`/`write_file`/
`list_files` tools plus the internal-token-gated `/tools/bash_exec` and
`/tools/schema` endpoints. `orchestrator/prompt.py` mirrors
`schema_templates.py` byte-for-byte (raises `ValueError` for `"ultra"`,
same as the source of truth — no trained checkpoint yet). 45 tests passing
as of last check. Built on branch `claude/build-from-plan-327r12`
(pre-existing, already-reviewed orchestrator — this project built on top of
it rather than rebuilding).

### `imagination-ui`
Next.js 16 chat UI. Supabase auth (Google OAuth via `@supabase/ssr`),
conversations persisted to Supabase Postgres (RLS-scoped to `auth.uid()`),
real tool dispatch loop (`use-chat.ts`, max 8 round-trips/turn) against the
sandbox's execute endpoint. Fixed a real SSRF risk (client-supplied
`baseUrl`/`authToken` blindly fetched server-side) by moving to server-only
env vars. Fixed a real missing-`LENGTH_INSTRUCTION` bug in `prompt.ts` found
by programmatically diffing prompt output across all three repos rather
than trusting a visual read.

---

## 6. Recurring bug class across this whole project (worth knowing before touching any of this)

The single most common root cause of real bugs found across 2.1 and 2.2
work has been **training/generating an example where the system prompt
says one thing and the actual behavior does another** — a
prompt/behavior contradiction that silently erodes instruction-following
once trained on. Concrete instances:
1. The original `<tool_call>`/`<think>` collapse investigation (pre-dates
   detailed records, but shaped a lot of downstream design decisions).
2. The no-tool-ultra-vs-"using real tool calls" contradiction in
   `LEVEL_INSTRUCTIONS["ultra"]` (found, reworded, fixed).
3. `validate_example()`'s dead ultra-label-check code letting
   `<think>`-block-placement violations through undetected (§4.4).
4. `stealth/ox-alpha`/`z-ai/glm-5.3-flash` treating `<think>` as a real
   control token rather than literal text (§4.2) — same underlying class
   of issue, just surfacing at the PROVIDER level instead of the training-
   data level: the model's serving stack has its own opinion about what
   `<think>` means, independent of what we're asking it to output.

**When debugging something that "shouldn't be possible" in this pipeline,
check for this pattern first.**

## 7. Verify-before-spend discipline (why this doc is this detailed)

This project has an established, repeatedly-reinforced pattern: **verify
before spending real money or real time**, and **be honest about actual
yield/cost rather than assuming a plan translated to reality.** Concrete
instances worth knowing about:
- A `--max-calls`/`--workers` misconfiguration once produced misleading
  "0 written, all CAPPED" results with no visible reason (concurrent
  workers claiming all call slots on first attempts leaves zero retry
  headroom) — always set `--max-calls` well above `--workers`.
- An initial no-tool ultra backfill produced 55.3% no-tool examples instead
  of the intended small minority, because no-tool examples have much
  higher generation yield despite a smaller budget allocation — caught by
  actually computing the ratio, not assuming the budget split reflected
  the data split.
- This session's own credit-exhaustion incident (§4.5): the account balance
  was never actually checked before launching what was assumed to be a
  cheap paid run. **Lesson applied going forward**: check
  `GET /api/v1/credits` before any real-money generation run, not just the
  observed per-call rate.
