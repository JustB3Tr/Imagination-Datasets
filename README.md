# Imagination 2 Pro — SFT data pipeline

Real, runnable pipeline for generating the QLoRA finetuning dataset:
3 domains (agentic_tool_use, subagent_orchestration, general_code) x 3
reasoning levels (low, medium, high), with subagent-mode variants emitted
automatically alongside the direct-mode examples.

Tested: `dedup_and_split.py` has been run against a known set of
duplicate/near-duplicate examples and confirmed to actually collapse them
(see the threshold note in the script). `generate.py` needs your own API
key to run since it calls a paid model.

## 1. Setup

```bash
pip install -r requirements.txt
export IMG2_API_KEY="your-key-here"
export IMG2_API_BASE="https://api.deepseek.com"   # or your provider's base url
export IMG2_MODEL="deepseek-chat"                  # or your provider's model name
```

Any OpenAI-compatible chat completions endpoint works. DeepSeek and most
z.ai/GLM endpoints are OpenAI-SDK compatible, just point `IMG2_API_BASE`
and `IMG2_MODEL` at them.

## 2. Generate

**First, preview cost with zero spend.** `--dry-run` needs no API key and makes
no calls, it estimates a worst-case cost ceiling (assumes every call maxes
out its token budget, real cost is usually lower):

```bash
python generate.py --dry-run --all --variants 30
```

At `--variants 30` that's roughly ~10k examples across all buckets for a
worst-case ceiling around $4-5 at current DeepSeek pricing. Push variants
higher if you want more, the dry-run tells you the ceiling before you spend
a cent.

**Then set a hard spend cap for the real run.** The script checks actual
spend (from the API's real usage stats, not the estimate) before every
single call and stops queuing new ones once the cap is hit. It prints a
running total to stderr as it goes, so you can watch it live in your
terminal instead of tabbing over to a billing dashboard:

```bash
python generate.py --domain agentic_tool_use --level low --variants 3 --max-spend 0.50
cat data/raw/agentic_tool_use__low.jsonl | head -1 | python -m json.tool
```

Note: because calls run concurrently (`--workers`, default 6), actual spend
can overshoot the cap slightly, by up to roughly one batch of in-flight
calls, since a call already in progress when the cap trips isn't cancelled
mid-request. Set `--max-spend` a bit under your real ceiling, and drop
`--workers` to 1-2 for the tightest possible control on a first test run.

Read that one example. Does it match the schema in `schema_templates.py`?
Does the reasoning depth actually look like "low"? If it looks wrong, fix
the prompt in `generate.py` before you generate thousands more of it.

Once it looks right, run everything with a cap you're comfortable with:

```bash
python generate.py --all --variants 30 --workers 6 --max-spend 10.00
```

`--variants 3` means each of the ~25-30 seed tasks per domain gets
generated 3 times (each call uses temperature 0.9 and is told to vary
specifics), plus roughly 35% of agentic_tool_use/general_code examples
also get a subagent-mode variant for free. That lands you in the
~2k-4k-per-bucket range this was sized for. Bump `--variants` up if a
bucket comes out thin after dedup (see step 3).

Cost control: `MAX_TOKENS_BY_LEVEL` in `schema_templates.py` caps output
size per level (low=500, medium=1000, high=2000 tokens, plus ~400 for the
JSON wrapper). Watch your provider's usage dashboard for the first ~50
calls to confirm your actual per-example cost before letting `--all` run
unattended.

## 3. Dedup + split

```bash
python dedup_and_split.py
```

This runs 100% locally (a small embedding model, no API calls, no cost).
It:
- embeds every generated example
- drops near-duplicates within each domain/level/mode bucket
- prints per-bucket counts so you can see if any bucket is thin
- writes `data/train.jsonl`, `data/eval.jsonl`, `data/dedup_report.json`

Default similarity threshold is 0.87, calibrated against real paraphrase
pairs (see the comment in the script). It's a blunt instrument. **Also
manually read 100-200 random examples from `data/train.jsonl` yourself.**
Automated dedup catches literal/near-literal rewording, it will not catch
"different words, same underlying pattern" repetition, which is exactly
what silently wrecked the earlier 50k-row dataset.

## 4. Before you train

- Check `data/dedup_report.json` bucket_counts. Any domain/level/mode combo
  that's noticeably thinner than the rest, go generate more for that
  specific combo before you spend GPU money.
- `data/eval.jsonl` is held out, never train on it. Use it after training
  to actually check whether low/medium/high produce visibly different
  reasoning depth and whether orchestrator-mode examples correctly call
  `run_subagent` one at a time instead of trying to do the work directly.
