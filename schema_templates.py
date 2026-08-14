"""
Shared schema + system prompt templates for Imagination 2.1 Pro SFT data.

Lock this file in before you generate a single example. Every generated
row has to match one of these three modes exactly, or the model gets
inconsistent signal about what "low/medium/high" and "orchestrator vs
subagent" actually mean.
"""

LEVELS = ["low", "medium", "high", "max", "ultra"]
DOMAINS = ["agentic_tool_use", "subagent_orchestration", "general_code"]

# ultra is scoped to the two domains where a verify-and-iterate loop
# actually matters -- subagent_orchestration's persistence is already
# expressed via run_subagent delegation chains, a distinct enough pattern
# that it doesn't need its own ultra treatment (see PLAN.md "Open decisions").
ULTRA_DOMAINS = ["agentic_tool_use", "general_code"]

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
    "max": 9000,      # two structured alternatives (not one) plus a complete
                      # answer -- calibrated empirically against real
                      # deepseek-chat calls before any real generation spend,
                      # see PLAN.md.
    "ultra": 18000,   # UNCALIBRATED starting point (PLAN.md's 16k-20k
                      # estimate, midpoint) -- multiple tool-call round
                      # trips including a mandatory failure+correction cycle
                      # eat far more tokens than a single structured answer.
                      # high's real history: 2000 and 3200 both silently
                      # truncated before 6000 proved sufficient -- do NOT
                      # skip the small calibration batch (see PLAN.md step 4)
                      # before spending real money at this number.
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
        "Before calling any tool or taking any action, include a thorough "
        "<think>...</think> block using EXACTLY this structure, with these "
        "four labels verbatim on their own lines:\n"
        "Problem: <restate the core problem in your own words, 1-2 sentences>\n"
        "Alternative considered: <name ONE specific, concrete alternative "
        "approach -- a real technique, library, or method, not a vague "
        "restatement>\n"
        "Rejected because: <ONE specific technical reason this alternative "
        "is worse for THIS situation -- not a generic dismissal like 'more "
        "complex' or 'could also work'>\n"
        "Chosen because: <concretely explain why the chosen approach wins "
        "here>\n"
        "Do not skip any of the four labeled lines. After the think block, "
        "write a COMPLETE final answer -- finish any code you start, do not "
        "stop partway through."
    ),
    "max": (
        "This is maximum reasoning effort. Before calling any tool or "
        "taking any action, include an exceptionally thorough "
        "<think>...</think> block using EXACTLY this structure, with these "
        "labels verbatim on their own lines:\n"
        "Problem: <restate the problem thoroughly, including any "
        "non-obvious edge cases or constraints>\n"
        "Alternative 1 considered: <a first concrete, real alternative "
        "approach>\n"
        "Rejected because: <specific technical reason>\n"
        "Alternative 2 considered: <a SECOND concrete alternative -- must "
        "be genuinely different from Alternative 1, not a variation of the "
        "same idea>\n"
        "Rejected because: <specific technical reason>\n"
        "Chosen approach: <state the approach actually being taken>\n"
        "Why this wins: <concretely explain why the chosen approach beats "
        "BOTH alternatives for this exact situation, referencing specific "
        "tradeoffs>\n"
        "Do not skip any labeled line. After the think block, write a "
        "COMPLETE final answer -- finish any code you start, do not stop "
        "partway through, and verify correctness before concluding."
    ),
    "ultra": (
        "This is maximum persistence effort. Before acting, include a "
        "<think>...</think> block using EXACTLY this structure, with these "
        "three labels verbatim on their own lines:\n"
        "Problem: <restate the core problem, 1-2 sentences>\n"
        "Plan: <the concrete steps you will take>\n"
        "Verification strategy: <specifically how you will check your work "
        "using real tool calls -- running tests, re-reading output, "
        "checking actual state -- not assumption>\n"
        "Do not skip any of the three labeled lines. After the think block, "
        "do the work using real tool calls. This conversation MUST include "
        "at least one point where a check reveals something wrong or "
        "incomplete -- a failing test, unexpected output, a stale "
        "assumption -- and you notice it and correct it before finishing. "
        "A clean one-pass success with no self-correction is NOT a valid "
        "response at this tier. Keep working and verifying until the task "
        "is actually, verifiably finished, not just apparently finished."
    ),
}

# --- System prompt templates -------------------------------------------

DIRECT_MODE_SYSTEM = (
    "You are Imagination 2.1 Pro operating in DIRECT mode. You may call tools "
    "when needed to complete the user's task. {level_instruction}"
)

ORCHESTRATOR_MODE_SYSTEM = (
    "You are Imagination 2.1 Pro operating in ORCHESTRATOR mode. You do not "
    "do the work yourself. Break the task into a sequence of well-scoped "
    "subtasks and hand each one to a subagent via the run_subagent tool, "
    "ONE AT A TIME. Wait for each subagent's result before deciding the "
    "next step or calling it done. When all subtasks are complete, "
    "synthesize a final answer from the results. {level_instruction}"
)

SUBAGENT_MODE_SYSTEM = (
    "You are Imagination 2.1 Pro operating in SUBAGENT mode. You have been "
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
- if reasoning_effort is medium, high, or max, the final assistant message's
  content must start with a <think>...</think> block before the answer.
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
