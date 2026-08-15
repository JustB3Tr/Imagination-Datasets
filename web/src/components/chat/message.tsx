"use client";

import { motion } from "framer-motion";
import { Sparkles, User } from "lucide-react";
import type { ChatMessage } from "@/lib/types";
import { parseThinkContent } from "@/lib/think";
import { ReasoningPanel } from "@/components/chat/reasoning-panel";
import { ToolCallCard } from "@/components/chat/tool-call-card";
import { Markdown } from "@/components/chat/markdown";
import { cn } from "@/lib/utils";

interface MessageProps {
  message: ChatMessage;
  toolResults: Map<string, string>;
}

export function Message({ message, toolResults }: MessageProps) {
  if (message.role === "tool") return null;

  if (message.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="flex justify-end gap-3"
      >
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-accent px-4 py-2.5 text-[15px] leading-relaxed text-accent-foreground">
          {message.content}
        </div>
        <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <User className="size-3.5" />
        </div>
      </motion.div>
    );
  }

  const { think, thinkComplete, answer } = parseThinkContent(message.content);
  const streamingThink = Boolean(message.streaming) && !thinkComplete;
  const hasToolCalls = message.toolCalls && message.toolCalls.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="flex gap-3"
    >
      <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-accent-muted text-accent">
        <Sparkles className="size-3.5" />
      </div>
      <div className="min-w-0 flex-1">
        {think !== null && <ReasoningPanel text={think} streaming={streamingThink} />}

        {hasToolCalls &&
          message.toolCalls!.map((tc) => (
            <ToolCallCard key={tc.id} toolCall={tc} result={toolResults.get(tc.id)} />
          ))}

        {answer.length > 0 ? (
          <Markdown content={answer} />
        ) : message.streaming && thinkComplete && !hasToolCalls ? (
          <span className={cn("inline-block h-4 w-1.5 animate-pulse bg-accent align-middle")} />
        ) : null}
      </div>
    </motion.div>
  );
}
