"use client";

import * as React from "react";
import { useAppStore } from "@/lib/store";
import { streamChat, ChatStreamError } from "@/lib/stream-chat";
import type { ChatMessage } from "@/lib/types";

function genId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export type ChatStatus = "idle" | "streaming" | "error";

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

      const assistantId = genId();
      addMessage({
        id: assistantId,
        role: "assistant",
        content: "",
        effort,
        createdAt: Date.now(),
        streaming: true,
      });

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChat({
          messages: historyForRequest,
          mode,
          effort,
          baseUrl: settings.baseUrl,
          model: settings.model,
          authToken: settings.authToken,
          signal: controller.signal,
          onContentDelta: (delta) => appendToMessage(assistantId, delta),
          onToolCalls: (toolCalls) => updateMessage(assistantId, { toolCalls }),
          onDone: () => updateMessage(assistantId, { streaming: false }),
        });
        setStatus("idle");
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
      } finally {
        abortRef.current = null;
      }
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
