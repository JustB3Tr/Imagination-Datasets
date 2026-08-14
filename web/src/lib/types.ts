import type { Mode, EffortLevel } from "@/lib/prompt";

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export type ChatRole = "system" | "user" | "assistant" | "tool";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  /** Raw content, including any literal <think>...</think> the model emitted. */
  content: string;
  toolCalls?: ToolCall[];
  /** Present on role: "tool" messages, pairs the result back to its call. */
  toolCallId?: string;
  /** Effort level active when this message was generated (assistant messages only). */
  effort?: EffortLevel;
  createdAt: number;
  /** True while an assistant message is still streaming in. */
  streaming?: boolean;
}

export interface OllamaSettings {
  baseUrl: string;
  model: string;
  authToken: string;
}

export const DEFAULT_SETTINGS: OllamaSettings = {
  baseUrl: "http://localhost:11434",
  model: "imagination-2.1-pro-lmh",
  authToken: "",
};

export interface AppState {
  settings: OllamaSettings;
  effort: EffortLevel;
  mode: Mode;
  messages: ChatMessage[];
}
