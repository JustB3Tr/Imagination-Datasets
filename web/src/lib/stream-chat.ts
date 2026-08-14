import type { Mode, EffortLevel } from "@/lib/prompt";
import type { ChatMessage, ToolCall } from "@/lib/types";

interface WireMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
}

export function toWireMessages(messages: ChatMessage[]): WireMessage[] {
  return messages.map((m) => ({
    role: m.role,
    content: m.content || (m.toolCalls?.length ? null : ""),
    tool_calls: m.toolCalls,
    tool_call_id: m.toolCallId,
  }));
}

interface StreamChatArgs {
  messages: ChatMessage[];
  mode: Mode;
  effort: EffortLevel;
  baseUrl: string;
  model: string;
  authToken: string;
  signal: AbortSignal;
  onContentDelta: (delta: string) => void;
  onToolCalls: (toolCalls: ToolCall[]) => void;
  onDone: () => void;
}

interface ToolCallAccumulator {
  id: string;
  name: string;
  arguments: string;
}

export class ChatStreamError extends Error {}

export async function streamChat({
  messages,
  mode,
  effort,
  baseUrl,
  model,
  authToken,
  signal,
  onContentDelta,
  onToolCalls,
  onDone,
}: StreamChatArgs): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: toWireMessages(messages),
      mode,
      effort,
      baseUrl,
      model,
      authToken,
    }),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.error) detail = data.error;
    } catch {
      // ignore
    }
    throw new ChatStreamError(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const toolCallsByIndex = new Map<number, ToolCallAccumulator>();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") continue;

      let json: {
        choices?: Array<{
          delta?: {
            content?: string;
            tool_calls?: Array<{
              index: number;
              id?: string;
              function?: { name?: string; arguments?: string };
            }>;
          };
        }>;
      };
      try {
        json = JSON.parse(payload);
      } catch {
        continue;
      }

      const delta = json.choices?.[0]?.delta;
      if (!delta) continue;

      if (delta.content) onContentDelta(delta.content);

      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const existing = toolCallsByIndex.get(tc.index) ?? { id: "", name: "", arguments: "" };
          if (tc.id) existing.id = tc.id;
          if (tc.function?.name) existing.name = tc.function.name;
          if (tc.function?.arguments) existing.arguments += tc.function.arguments;
          toolCallsByIndex.set(tc.index, existing);
        }
        onToolCalls(
          Array.from(toolCallsByIndex.entries())
            .sort(([a], [b]) => a - b)
            .map(([index, acc]) => ({
              id: acc.id || `call_${index}`,
              type: "function" as const,
              function: { name: acc.name, arguments: acc.arguments },
            })),
        );
      }
    }
  }

  onDone();
}
