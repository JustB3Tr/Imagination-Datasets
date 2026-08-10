"""
Shared schema + system prompt templates for Imagination 2 Pro SFT data.

Lock this file in before you generate a single example. Every generated
row has to match one of these three modes exactly, or the model gets
inconsistent signal about what "low/medium/high" and "orchestrator vs
subagent" actually mean.
"""

LEVELS = ["low", "medium", "high"]
DOMAINS = ["agentic_tool_use", "subagent_orchestration", "general_code"]

# Rough output length budget per reasoning level. Used both to steer the
# generator model and to cap how much you pay for per example.
MAX_TOKENS_BY_LEVEL = {
    "low": 500,      # little to no visible reasoning, answer fast
    "medium": 1000,  # some visible reasoning, a few steps
    "high": 6000,    # explicit, thorough step-by-step reasoning + a real,
                      # complete answer -- 2000 was routinely truncating on
                      # examples with substantial code (an LLM-as-judge audit
                      # found ~30% of "high" examples cut off mid-answer), and
                      # 3200 still hit finish_reason="length" on a real
                      # DeepSeek V4 Flash (gmicloud/fp8) call for a code-heavy
                      # task (15.9k chars written, cut off mid-sentence).
                      # Output tokens are cheap ($0.224/1M on this route), so
                      # the safety margin costs essentially nothing.
}

LEVEL_INSTRUCTIONS = {
    # These deliberately avoid the literal phrase "reasoning_effort: X" --
    # that string collides with the actual API-level reasoning_effort
    # parameter some generator calls pass (e.g. "none", to suppress a
    # provider's hidden/native chain-of-thought and avoid silent truncation).
    # Telling the model in-content "reasoning_effort: high" while the real
    # API parameter says "none" is a direct, visible contradiction, and
    # models that are sensitive to their own reasoning_effort setting seem
    # to resolve it by trusting the real parameter and skipping the
    # requested <think> block entirely (observed empirically: DeepSeek V4
    # Flash dropped the <think> block in a large fraction of "high" calls
    # once reasoning_effort=none was introduced). Describing the desired
    # CONTENT directly, without naming the parameter, avoids that collision.
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
}

# --- System prompt templates -------------------------------------------

DIRECT_MODE_SYSTEM = (
    "You are Imagination 2 Pro operating in DIRECT mode. You may call tools "
    "when needed to complete the user's task. {level_instruction}"
)

ORCHESTRATOR_MODE_SYSTEM = (
    "You are Imagination 2 Pro operating in ORCHESTRATOR mode. You do not "
    "do the work yourself. Break the task into a sequence of well-scoped "
    "subtasks and hand each one to a subagent via the run_subagent tool, "
    "ONE AT A TIME. Wait for each subagent's result before deciding the "
    "next step or calling it done. When all subtasks are complete, "
    "synthesize a final answer from the results. {level_instruction}"
)

SUBAGENT_MODE_SYSTEM = (
    "You are Imagination 2 Pro operating in SUBAGENT mode. You have been "
    "handed a single, self-contained subtask by an orchestrator. You have "
    "NO memory of any larger conversation or plan. Complete this subtask "
    "and return a clear, structured result. {level_instruction}"
)

# --- Tool schema used inside generated examples -------------------------
# Keep this small and generic. The generator model should invent realistic
# domain tools (fetch, shell, file_search, sql_query, etc.) per example, but
# run_subagent is the one tool that must always look like this so the
# orchestrator/subagent pattern is consistent across the whole dataset.

RUN_SUBAGENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_subagent",
        "description": (
            "Delegate a single, self-contained subtask to a fresh subagent "
            "instance. Returns the subagent's result once it completes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The complete, self-contained subtask description.",
                }
            },
            "required": ["task"],
        },
    },
}

# --- Target JSON schema for one generated training example --------------
# This is what you ask the generator API to return for every call.

EXAMPLE_JSON_INSTRUCTIONS = """
Return ONLY a single JSON object (no markdown fences, no commentary) with
this exact shape:

{
  "messages": [
    {"role": "system", "content": "<the system prompt you were given, verbatim>"},
    {"role": "user", "content": "<a concrete, realistic version of the task>"},
    ... one or more assistant/tool message turns that realistically solve it ...
  ]
}

Rules:
- assistant messages that call tools must include a "tool_calls" field
  (OpenAI-style: list of {"id", "type": "function", "function": {"name", "arguments"}}).
- every tool_call must be followed by a matching {"role": "tool", "tool_call_id": ..., "content": "<realistic tool result>"} message.
- NEVER narrate a tool call in prose (e.g. "I'll call file_search with a
  query..." followed by another assistant message with the result). If the
  assistant uses a tool, it MUST be a real "tool_calls" entry on that message,
  immediately followed by a "role": "tool" result message -- never two
  "role": "assistant" messages back to back.
- the final assistant message must NOT call a tool, it must contain the final answer.
- if reasoning_effort is medium or high, the final assistant message's content
  must start with a <think>...</think> block before the answer.
- if reasoning_effort is low, do not include a <think> block anywhere.
- in ORCHESTRATOR mode, tool calls must ONLY use run_subagent, called one at
  a time (never multiple tool_calls in the same message), and the "content"
  passed back in each tool result should read like a real subagent's output,
  not a placeholder.
- make the scenario concrete: real file names, real error messages, real
  numbers. Avoid generic placeholder text like "some function" or "an error occurred".
- vary the specifics from the seed task description, do not just restate it.
- the response MUST end with the JSON object fully closed (final "}"} --
  never stop output while still inside a string value, even in a later
  assistant message that nests tool-result-shaped text (e.g. a "content"
  field containing what looks like its own JSON). If you're running low on
  room, wrap up the current message and close the JSON early rather than
  leaving a string unterminated.
""".strip()


def build_system_prompt(mode: str, level: str) -> str:
    assert mode in ("direct", "orchestrator", "subagent")
    assert level in LEVELS
    level_instruction = LEVEL_INSTRUCTIONS[level]
    template = {
        "direct": DIRECT_MODE_SYSTEM,
        "orchestrator": ORCHESTRATOR_MODE_SYSTEM,
        "subagent": SUBAGENT_MODE_SYSTEM,
    }[mode]
    return template.format(level_instruction=level_instruction)


def mode_for_domain(domain: str) -> str:
    """Which system-prompt mode a given domain's examples should use.

    subagent_orchestration examples are generated in ORCHESTRATOR mode.
    The SUBAGENT mode data mostly comes for free from agentic_tool_use and
    general_code examples reused with a subagent-mode system prompt (see
    SUBAGENT_VARIANT_RATE in generate.py), rather than a fourth domain.
    """
    return {
        "agentic_tool_use": "direct",
        "subagent_orchestration": "orchestrator",
        "general_code": "direct",
    }[domain]
