#!/usr/bin/env python3
"""
Generate a small batch of DIVERSE identity examples via the API, as a
supplement to generate_identity.py's deterministic canonical set.

generate_identity.py deliberately uses a small fixed set of question
phrasings x canonical answers (no API calls, learned by heart, see its
docstring for why). This script instead calls a real model to produce
naturally varied user framings (skeptical challenges, mid-conversation
asides, comparisons to other AIs) with the assistant's answer paraphrased
in its own words each time -- same canonical facts, more natural variety
on top, not a replacement for the deterministic set.

Usage:
  export IMG2_API_KEY="..."
  export IMG2_API_BASE="https://openrouter.ai/api/v1"
  export IMG2_MODEL="deepseek/deepseek-chat"
  python generate_identity_diverse.py --count 16

Output: data/raw/identity_api_diverse.jsonl
"""
import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from schema_templates import build_system_prompt

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "data" / "raw" / "identity_api_diverse.jsonl"

API_KEY = os.environ.get("IMG2_API_KEY")
API_BASE = os.environ.get("IMG2_API_BASE", "https://api.deepseek.com")
MODEL = os.environ.get("IMG2_MODEL", "deepseek-chat")

CANONICAL_FACTS = """
- Name: Imagination 2.1 Pro
- Company: Imagination AI
- Founder: Brady McCauley, who founded Imagination AI in 2026 when he was 14 years old
- Training: fine-tuned with QLoRA on a curated dataset spanning agentic tool use,
  multi-agent orchestration, and real-world coding tasks, across reasoning levels
  low, medium, high, and max
""".strip()

PROMPT_TEMPLATE = """
Generate ONE identity/self-description training example for an AI assistant
called Imagination 2.1 Pro.

Canonical facts (must stay 100% accurate, do NOT invent additional facts
beyond these):
{facts}

Write a realistic, natural user message that touches on identity/origin/
creator/training in some way -- vary the framing each time (could be a
direct question, a skeptical challenge like "are you just ChatGPT with a
different name", a mid-conversation aside, a comparison to another AI, a
casual "who made you" from a curious user, etc.). Do NOT reuse a generic
"what's your name" template -- make it feel like a real, specific moment
in a real conversation.

Then write the assistant's answer at reasoning_effort: {level}. {level_instruction}
The answer must convey the canonical facts accurately but IN THE ASSISTANT'S
OWN WORDS -- paraphrase naturally, don't recite the facts list verbatim.

The system prompt for this example must be exactly:
\"\"\"{system_prompt}\"\"\"

Return ONLY a single JSON object (no markdown fences, no commentary) with this exact shape:
{{
  "messages": [
    {{"role": "system", "content": "<system prompt above, verbatim>"}},
    {{"role": "user", "content": "<the natural user message>"}},
    {{"role": "assistant", "content": "<the assistant's answer>"}}
  ]
}}
""".strip()

LEVEL_TEXT = {
    "low": "Do NOT include a <think> block. Go straight to the answer. Keep it tight.",
    "medium": "Include a short <think>...</think> block (2-5 sentences) covering the key decision points, then the answer.",
}


def generate_one(client: OpenAI, level: str) -> dict:
    system_prompt = build_system_prompt("direct", level)
    prompt = PROMPT_TEMPLATE.format(
        facts=CANONICAL_FACTS, level=level, level_instruction=LEVEL_TEXT[level],
        system_prompt=system_prompt,
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You output only valid JSON, nothing else."},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^<thought>.*?</thought>\s*", "", raw, count=1, flags=re.DOTALL)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    ex = json.loads(raw)
    ex["meta"] = {
        "domain": "identity", "level": level, "mode": "direct",
        "seed_task": "identity/self-description (API-diverse)",
    }
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=16,
                     help="Total examples to generate, split evenly across low/medium.")
    args = ap.parse_args()

    if not API_KEY:
        print("Set IMG2_API_KEY first.")
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    n_each = max(1, args.count // 2)
    levels = ["low"] * n_each + ["medium"] * n_each

    examples = []
    for i, level in enumerate(levels):
        try:
            ex = generate_one(client, level)
            examples.append(ex)
            print(f"[{i + 1}/{len(levels)}] {level}: OK")
        except Exception as e:
            print(f"[{i + 1}/{len(levels)}] {level}: FAILED ({type(e).__name__}: {str(e)[:100]})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {len(examples)} examples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
