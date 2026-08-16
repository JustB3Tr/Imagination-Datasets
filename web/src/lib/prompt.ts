/**
 * The Imagination 2.1 Pro system-prompt contract.
 *
 * These strings are reproduced byte-for-byte from `schema_templates.py` in
 * the Imagination-Datasets training repo. The model was fine-tuned to
 * recognize this EXACT text and respond with the corresponding behavior
 * (e.g. emitting a structured <think> block at a given effort level). Do
 * NOT paraphrase, reformat, or "clean up" any of the strings below — any
 * deviation silently degrades model behavior. If the training-side
 * templates ever change, update this file to match them exactly and bump
 * a comment noting the sync date.
 *
 * See README.md "System prompt contract" section for the full writeup.
 */

export type Mode = "direct" | "orchestrator" | "subagent";

export type EffortLevel = "low" | "medium" | "high" | "max" | "ultra";

const MODE_TEMPLATES: Record<Mode, string> = {
  direct:
    "You are Imagination 2.1 Pro operating in DIRECT mode. You may call tools " +
    "when needed to complete the user's task. {level_instruction}",

  orchestrator:
    "You are Imagination 2.1 Pro operating in ORCHESTRATOR mode. You do not " +
    "do the work yourself. Break the task into a sequence of well-scoped " +
    "subtasks and hand each one to a subagent via the run_subagent tool, " +
    "ONE AT A TIME. Wait for each subagent's result before deciding the " +
    "next step or calling it done. When all subtasks are complete, " +
    "synthesize a final answer from the results. {level_instruction}",

  subagent:
    "You are Imagination 2.1 Pro operating in SUBAGENT mode. You have been " +
    "handed a single, self-contained subtask by an orchestrator. You have " +
    "NO memory of any larger conversation or plan. Complete this subtask " +
    "and return a clear, structured result. {level_instruction}",
};

/**
 * Shared across every mode and level -- the reasoning_effort level controls
 * how much the model deliberates, not how long its output is. Appended
 * after {level_instruction} in buildSystemPrompt() rather than folded into
 * each LEVEL_INSTRUCTIONS entry, mirroring schema_templates.py's
 * LENGTH_INSTRUCTION exactly.
 */
const LENGTH_INSTRUCTION =
  " Your response length is dictated entirely by what the task actually " +
  "requires, never by the reasoning_effort level -- write as much as " +
  "needed to fully and correctly finish it, and do not cut a thought, " +
  "explanation, or piece of code short because of an assumed length " +
  "limit. A low-effort task that happens to require a large amount of " +
  "code should still produce all of it in full; a max-effort task with a " +
  "small, simple scope should not be padded with unnecessary length. " +
  "reasoning_effort is about how much you deliberate before and during " +
  "acting, not about producing a longer or shorter final answer.";

const LEVEL_INSTRUCTIONS: Record<Exclude<EffortLevel, "ultra">, string> = {
  low: "Do NOT include a <think> block. Go straight to the action/answer. Keep it tight.",

  medium:
    "Include a short <think>...</think> block (2-5 sentences) covering the " +
    "key decision points, then the answer.",

  high:
    "Before calling any tool or taking any action, include a thorough " +
    "<think>...</think> block using EXACTLY this structure, with these " +
    "four labels verbatim on their own lines:\n" +
    "Problem: <restate the core problem in your own words, 1-2 sentences>\n" +
    "Alternative considered: <name ONE specific, concrete alternative " +
    "approach -- a real technique, library, or method, not a vague " +
    "restatement>\n" +
    "Rejected because: <ONE specific technical reason this alternative " +
    "is worse for THIS situation -- not a generic dismissal like 'more " +
    "complex' or 'could also work'>\n" +
    "Chosen because: <concretely explain why the chosen approach wins " +
    "here>\n" +
    "Do not skip any of the four labeled lines. After the think block, " +
    "write a COMPLETE final answer -- finish any code you start, do not " +
    "stop partway through.",

  max:
    "This is maximum reasoning effort. Before calling any tool or " +
    "taking any action, include an exceptionally thorough " +
    "<think>...</think> block using EXACTLY this structure, with these " +
    "labels verbatim on their own lines:\n" +
    "Problem: <restate the problem thoroughly, including any " +
    "non-obvious edge cases or constraints>\n" +
    "Alternative 1 considered: <a first concrete, real alternative " +
    "approach>\n" +
    "Rejected because: <specific technical reason>\n" +
    "Alternative 2 considered: <a SECOND concrete alternative -- must " +
    "be genuinely different from Alternative 1, not a variation of the " +
    "same idea>\n" +
    "Rejected because: <specific technical reason>\n" +
    "Chosen approach: <state the approach actually being taken>\n" +
    "Why this wins: <concretely explain why the chosen approach beats " +
    "BOTH alternatives for this exact situation, referencing specific " +
    "tradeoffs>\n" +
    "Do not skip any labeled line. After the think block, write a " +
    "COMPLETE final answer -- finish any code you start, do not stop " +
    "partway through, and verify correctness before concluding.",
};

/**
 * Builds the exact system prompt string sent as the `system` role message
 * on every request: MODE_TEMPLATES[mode].format(level_instruction=...).
 *
 * "ultra" is not yet trained (see EffortConfig below) — callers should not
 * be able to reach this function with level="ultra" while it's disabled,
 * but if they do, we fail loudly rather than silently sending a broken
 * prompt to the model.
 */
export function buildSystemPrompt(mode: Mode, level: EffortLevel): string {
  if (level === "ultra") {
    throw new Error(
      "The 'ultra' effort level has no trained instruction text yet. " +
        "Do not send it to the model until LEVEL_INSTRUCTIONS.ultra is added.",
    );
  }
  const template = MODE_TEMPLATES[mode];
  return template.replace("{level_instruction}", LEVEL_INSTRUCTIONS[level]) + LENGTH_INSTRUCTION;
}

export interface EffortConfig {
  level: EffortLevel;
  label: string;
  /** Short human summary of intent, shown in the selector's tooltip — NOT the raw instruction text. */
  description: string;
  /** Relative intensity for the visual indicator, 0-1. */
  intensity: number;
  disabled?: boolean;
}

/**
 * Config-driven effort levels for the UI. Adding "ultra" once it's trained
 * is a one-line change: flip `disabled: false` and give it a description
 * once LEVEL_INSTRUCTIONS.ultra exists above.
 */
export const EFFORT_LEVELS: EffortConfig[] = [
  {
    level: "low",
    label: "Low",
    description: "Straight to the answer, no reasoning block.",
    intensity: 0.2,
  },
  {
    level: "medium",
    label: "Medium",
    description: "A short reasoning summary before the answer.",
    intensity: 0.45,
  },
  {
    level: "high",
    label: "High",
    description: "One alternative considered and rejected before committing.",
    intensity: 0.7,
  },
  {
    level: "max",
    label: "Max",
    description: "Two alternatives considered and rejected before committing.",
    intensity: 1,
  },
  {
    level: "ultra",
    label: "Ultra",
    description: "Coming soon — not yet trained.",
    intensity: 1,
    disabled: true,
  },
];

export interface ModeConfig {
  mode: Mode;
  label: string;
  description: string;
  disabled?: boolean;
}

/**
 * Phase 2 stub: only "direct" is enabled for v1. The builder function above
 * already accepts a `mode` param so wiring up a UI toggle later doesn't
 * require touching the prompt contract.
 */
export const MODES: ModeConfig[] = [
  {
    mode: "direct",
    label: "Direct",
    description: "The model does the work itself, calling tools as needed.",
  },
  {
    mode: "orchestrator",
    label: "Orchestrator",
    description: "Breaks the task into subtasks and delegates to subagents.",
    disabled: true,
  },
  {
    mode: "subagent",
    label: "Subagent",
    description: "Handles a single self-contained subtask with no broader context.",
    disabled: true,
  },
];
