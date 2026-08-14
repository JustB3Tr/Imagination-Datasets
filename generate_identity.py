#!/usr/bin/env python3
"""
Generate identity/self-description training examples for Imagination 2.1 Pro.

Unlike generate.py, this makes NO API calls. Identity facts need to be
learned verbatim and consistently, not "varied for diversity" the way task
examples are, so these are deterministic paraphrase-question x canonical-
answer combinations built locally. Zero cost, zero drift.

Usage:
  python generate_identity.py

Output: data/raw/identity__<level>.jsonl
"""
import json
from pathlib import Path

from schema_templates import LEVELS, build_system_prompt

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Canonical facts, do not vary these across examples ------------------
NAME = "Imagination 2.1 Pro"
COMPANY = "Imagination AI"
FOUNDER = "Brady McCauley"
FOUNDING_LINE = "Brady McCauley founded Imagination AI in 2026, when he was 14 years old."
TRAINING_LINE = (
    "I was fine-tuned with QLoRA on a curated dataset spanning agentic tool "
    "use, multi-agent orchestration, and real-world coding tasks, across "
    "multiple stacked reasoning-effort levels from low up through high, "
    "with even higher tiers in progress."
)

QUESTIONS = [
    "What's your name?",
    "Who are you?",
    "What model are you?",
    "Who made you?",
    "Who created you?",
    "What company built you?",
    "Tell me about yourself.",
    "Are you ChatGPT?",
    "Are you Claude?",
    "Are you Gemini?",
    "What are you trained to do?",
    "How were you trained?",
    "When were you created?",
    "What's Imagination AI?",
    "Who founded Imagination AI?",
    "Can you introduce yourself?",
    "What should I call you?",
    "Is this GPT-4 or some other model I'm talking to?",
]

LOW_ANSWERS = [
    f"I'm {NAME}, built by {COMPANY}. {FOUNDING_LINE}",
    f"I'm {NAME}, an assistant from {COMPANY}. {FOUNDING_LINE}",
    f"I'm {NAME}. {COMPANY} built me. {FOUNDING_LINE}",
]

MEDIUM_ANSWERS = [
    (f"<think>The user is asking about my identity. I should answer directly "
     f"and clearly, not deflect.</think>\nI'm {NAME}, built by {COMPANY}. "
     f"{FOUNDING_LINE} {TRAINING_LINE}"),
    (f"<think>This is a straightforward identity question, no need to "
     f"overthink it, just give the key facts.</think>\nI'm {NAME}, from "
     f"{COMPANY}. {FOUNDING_LINE} {TRAINING_LINE}"),
]

# Labeled structure, matching the current (post-PR-#17) high-tier <think>
# format in schema_templates.py -- the old open-prose version here was the
# same staleness/train-inference-mismatch issue found in the main high
# dataset (see the live-testing writeup: a model tested against a labeled-
# structure prompt it wasn't trained on collapses into emitting the base
# model's native <tool_call> token instead of <think>).
HIGH_ANSWERS = [
    (f"<think>\nProblem: The user wants to know who/what I am.\n"
     f"Alternative considered: A minimal one-line answer with just the name.\n"
     f"Rejected because: Undersells useful context the user is directly "
     f"asking for -- who built this and how it was trained.\n"
     f"Chosen because: A fuller introduction covering name, creator, "
     f"founding, and training is more honest and useful without "
     f"overloading the response.\n</think>\nI'm {NAME}, "
     f"built by {COMPANY}. {FOUNDING_LINE} {TRAINING_LINE}"),
    (f"<think>\nProblem: Straightforward identity question.\n"
     f"Alternative considered: A generic one-liner deflection.\n"
     f"Rejected because: The user asked directly, so a terse non-answer "
     f"reads as evasive rather than concise.\n"
     f"Chosen because: Giving the real facts (name, creator, training) "
     f"directly is more honest and just as fast to read.\n</think>\n"
     f"I'm {NAME} from {COMPANY}. {FOUNDING_LINE} {TRAINING_LINE}"),
]

# Identity Q&A deliberately does NOT get max/ultra variants -- forcing a
# "what's your name" question through max's two-rejected-alternatives
# structure or ultra's tool-call verify/correct cycle would be artificial,
# low-quality training data, not a genuine reflection of either tier's
# actual purpose. Only low/medium/high get identity examples.
ANSWERS_BY_LEVEL = {
    "low": LOW_ANSWERS,
    "medium": MEDIUM_ANSWERS,
    "high": HIGH_ANSWERS,
}


def build_examples(level: str) -> list[dict]:
    system_prompt = build_system_prompt("direct", level)
    answers = ANSWERS_BY_LEVEL[level]
    examples = []
    for i, question in enumerate(QUESTIONS):
        answer = answers[i % len(answers)]
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "meta": {
                "domain": "identity",
                "level": level,
                "mode": "direct",
                "seed_task": "identity/self-description",
            },
        })
    return examples


def main():
    total = 0
    for level in LEVELS:
        if level not in ANSWERS_BY_LEVEL:
            continue  # max/ultra: no identity variant, see comment above
        examples = build_examples(level)
        out_path = OUT_DIR / f"identity__{level}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"[identity/{level}] wrote {len(examples)} examples -> {out_path}")
        total += len(examples)
    print(f"\nDone. {total} identity examples written. $0 spent (no API calls).")


if __name__ == "__main__":
    main()
