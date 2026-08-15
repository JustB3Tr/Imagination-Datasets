"""
Split data/{train,eval}.jsonl into one file per training stage: a combined
low+medium file (stage 1's base training set), then one file per higher
tier -- high, max, ultra -- each trained as its own stacked LoRA adapter on
top of the previous stage.

Why low+medium share a file but high/max/ultra don't: the LLM-judge audit
found "high" reasoning-level examples reliably score lower on level_match
(avg 3.53/5 vs 4.89/5 for low, 4.67/5 for medium) -- the <think> blocks
often skip the required "name a concrete alternative and reject it" step.
Rather than throw those examples away, keep them separate: train the base
model on low+medium (clean signal), then layer a second LoRA/adapter pass
trained on high on top of it, then max on top of that, then ultra on top
of that (see PLAN.md).

This used to only ever check level == "high", silently dumping everything
else -- including "max" once that tier started flowing through
dedup_and_split.py -- into the low_medium file. That would have
contaminated stage-1 training with max-tier examples (a much longer
system prompt, much longer expected output) without anyone noticing.
Bucketing explicitly by level instead of by exclusion avoids that class of
bug recurring when a future tier gets added.

Run this after every data/{train,eval}.jsonl rebuild (i.e. after
dedup_and_split.py):

  python3 split_by_level.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = ["train", "eval"]

# Levels sharing a file, in training-stage order. Anything not listed here
# (there shouldn't be anything, but see the ValueError below) is a bug, not
# silently swept into the wrong bucket.
LEVEL_GROUPS = {
    "low_medium": ("low", "medium"),
    "high": ("high",),
    "max": ("max",),
    "ultra": ("ultra",),
}


def main():
    for split in SPLITS:
        src = ROOT / "data" / f"{split}.jsonl"
        if not src.exists():
            print(f"skip {src} (not found)")
            continue

        buckets = {name: [] for name in LEVEL_GROUPS}
        level_to_group = {lv: name for name, levels in LEVEL_GROUPS.items() for lv in levels}

        with open(src) as f:
            for line in f:
                ex = json.loads(line)
                level = ex["meta"]["level"]
                group = level_to_group.get(level)
                if group is None:
                    raise ValueError(
                        f"Unrecognized level {level!r} in {src} -- add it to "
                        f"LEVEL_GROUPS in split_by_level.py instead of letting "
                        f"it fall through silently."
                    )
                buckets[group].append(ex)

        summary = []
        for group, examples in buckets.items():
            out_path = ROOT / "data" / f"{split}_{group}.jsonl"
            if not examples:
                # Don't write/leave a stale empty or outdated file for a
                # tier that isn't generated yet (e.g. ultra before its
                # first real run).
                continue
            with open(out_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            summary.append(f"{group}={len(examples)} -> {out_path.name}")

        print(f"{split}: " + "   ".join(summary))


if __name__ == "__main__":
    main()
