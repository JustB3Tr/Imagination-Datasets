"use client";

import * as React from "react";
import type { ChatMessage } from "@/lib/types";
import { Message } from "@/components/chat/message";

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const toolResults = React.useMemo(() => {
    const map = new Map<string, string>();
    for (const m of messages) {
      if (m.role === "tool" && m.toolCallId) map.set(m.toolCallId, m.content);
    }
    return map;
  }, [messages]);

  return (
    <div className="mx-auto flex w-full max-w-[72ch] flex-col gap-5 px-4 py-6">
      {messages.map((m) => (
        <Message key={m.id} message={m} toolResults={toolResults} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
