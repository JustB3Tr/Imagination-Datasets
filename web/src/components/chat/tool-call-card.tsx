"use client";

import * as React from "react";
import { Wrench, ArrowRight } from "lucide-react";
import type { ToolCall } from "@/lib/types";

interface ToolCallCardProps {
  toolCall: ToolCall;
  result?: string;
}

function prettyArgs(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

export function ToolCallCard({ toolCall, result }: ToolCallCardProps) {
  return (
    <div className="mb-3 overflow-hidden rounded-lg border border-accent/25 bg-accent-muted/40">
      <div className="flex items-center gap-2 border-b border-accent/20 px-3 py-2">
        <Wrench className="size-3.5 shrink-0 text-accent" />
        <span className="text-xs font-medium text-foreground">{toolCall.function.name}</span>
        <span className="rounded-full bg-accent/15 px-1.5 py-0.5 text-[10px] font-medium text-accent">
          tool call
        </span>
      </div>
      <pre className="overflow-x-auto px-3 py-2 font-mono text-[12px] leading-relaxed text-muted-foreground">
        {prettyArgs(toolCall.function.arguments)}
      </pre>
      {result !== undefined && (
        <div className="border-t border-accent/20 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
            <ArrowRight className="size-3" />
            result
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-foreground">
            {result}
          </pre>
        </div>
      )}
    </div>
  );
}
