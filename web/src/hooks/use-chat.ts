"use client";

import * as React from "react";
import { useAppStore } from "@/lib/store";
import { streamChat, ChatStreamError } from "@/lib/stream-chat";
import type { ChatMessage, ToolCall } from "@/lib/types";

function genId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export type ChatStatus = "idle" | "streaming" | "error";

const MAX_TOOL_ROUNDS = 4;

async function runTools(toolCalls: ToolCall[]) {
  const res = await fetch("/api/tools", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ toolCalls }),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.error) detail = data.error;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  const data = (await res.json()) as { results?: Array<{ toolCallId: string; content: string }> };
  return data.results ?? [];
}

export function useChat() {
  const messages = useAppStore((s) => s.messages);
  const effort = useAppStore((s) => s.effort);
  const mode = useAppStore((s) => s.mode);
  const settings = useAppStore((s) => s.settings);
  const addMessage = useAppStore((s) => s.addMessage);
  const updateMessage = useAppStore((s) => s.updateMessage);
  const appendToMessage = useAppStore((s) => s.appendToMessage);
  const removeMessage = useAppStore((s) => s.removeMessage);

  const [status, setStatus] = React.useState<ChatStatus>("idle");
  const [error, setError] = React.useState<string | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const historyRef = React.useRef<ChatMessage[]>(messages);
  React.useEffect(() => {
    historyRef.current = messages;
  }, [messages]);

  const runStream = React.useCallback(
    async (historyForRequest: ChatMessage[]) => {
      setError(null);
      setStatus("streaming");
      let currentHistory = historyForRequest;

      for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
        const assistantId = genId();
        const assistantMessage: ChatMessage = {
          id: assistantId,
          role: "assistant",
          content: "",
          effort,
          createdAt: Date.now(),
          streaming: true,
        };
        let assistantToolCalls: ToolCall[] = [];

        addMessage(assistantMessage);

        const controller = new AbortController();
        abortRef.current = controller;

        try {
          await streamChat({
            messages: currentHistory,
            mode,
            effort,
            baseUrl: settings.baseUrl,
            model: settings.model,
            authToken: settings.authToken,
            signal: controller.signal,
            onContentDelta: (delta) => {
              assistantMessage.content += delta;
              appendToMessage(assistantId, delta);
            },
            onToolCalls: (toolCalls) => {
              assistantToolCalls = toolCalls;
              assistantMessage.toolCalls = toolCalls;
              updateMessage(assistantId, { toolCalls });
            },
            onDone: () => updateMessage(assistantId, { streaming: false }),
          });
        } catch (err) {
          if (controller.signal.aborted) {
            updateMessage(assistantId, { streaming: false });
            setStatus("idle");
            return;
          }
          const message =
            err instanceof ChatStreamError || err instanceof Error
              ? err.message
              : "Something went wrong talking to Imagination 2.1 Pro.";
          removeMessage(assistantId);
          setError(message);
          setStatus("error");
          return;
        } finally {
          abortRef.current = null;
        }

        assistantMessage.streaming = false;
        currentHistory = [...currentHistory, { ...assistantMessage, toolCalls: assistantToolCalls }];

        if (assistantToolCalls.length === 0) {
          setStatus("idle");
          return;
        }

        try {
          const toolResults = await runTools(assistantToolCalls);
          const toolMessages: ChatMessage[] = toolResults.map((result) => ({
            id: genId(),
            role: "tool",
            content: result.content,
            toolCallId: result.toolCallId,
            createdAt: Date.now(),
          }));
          toolMessages.forEach(addMessage);
          currentHistory = [...currentHistory, ...toolMessages];
        } catch (err) {
          setError(err instanceof Error ? err.message : "Tool execution failed.");
          setStatus("error");
          return;
        }
      }

      setStatus("error");
      setError(`Stopped after ${MAX_TOOL_ROUNDS} tool rounds to avoid an infinite loop.`);
    },
    [addMessage, updateMessage, appendToMessage, removeMessage, effort, mode, settings],
  );

  const send = React.useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || status === "streaming") return;

      const userMessage: ChatMessage = {
        id: genId(),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
      };
      addMessage(userMessage);
      void runStream([...historyRef.current, userMessage]);
    },
    [status, addMessage, runStream],
  );

  const stop = React.useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const retryLast = React.useCallback(() => {
    if (status === "streaming") return;
    void runStream(historyRef.current);
  }, [status, runStream]);

  return { messages, status, error, send, stop, retryLast };
}
