"use client";

import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronRight, BrainCircuit } from "lucide-react";
import { cn } from "@/lib/utils";

interface ReasoningPanelProps {
  text: string;
  streaming: boolean;
}

export function ReasoningPanel({ text, streaming }: ReasoningPanelProps) {
  const [open, setOpen] = React.useState(() => streaming);

  return (
    <div className="mb-3 overflow-hidden rounded-lg border border-border bg-muted/60">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        <BrainCircuit className="size-3.5 shrink-0" />
        <span>Reasoning</span>
        {streaming && (
          <span className="flex items-center gap-0.5">
            <span className="size-1 animate-pulse rounded-full bg-accent [animation-delay:0ms]" />
            <span className="size-1 animate-pulse rounded-full bg-accent [animation-delay:150ms]" />
            <span className="size-1 animate-pulse rounded-full bg-accent [animation-delay:300ms]" />
          </span>
        )}
        <motion.span
          animate={{ rotate: open ? 90 : 0 }}
          transition={{ duration: 0.15 }}
          className="ml-auto"
        >
          <ChevronRight className="size-3.5" />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div
              className={cn(
                "whitespace-pre-wrap border-t border-border px-3 py-2.5 font-mono text-[12.5px] leading-relaxed text-muted-foreground",
              )}
            >
              {text.trim() || " "}
              {streaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-accent align-middle" />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
