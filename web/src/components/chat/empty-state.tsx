"use client";

import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { EFFORT_LEVELS } from "@/lib/prompt";

const EXAMPLE_PROMPTS = [
  "Write a Python function that merges overlapping intervals",
  "Explain the tradeoffs between REST and gRPC for a service mesh",
  "Refactor this function to avoid nested callbacks",
];

interface EmptyStateProps {
  onPick: (prompt: string) => void;
}

export function EmptyState({ onPick }: EmptyStateProps) {
  return (
    <div className="mx-auto flex w-full max-w-[72ch] flex-1 flex-col items-center justify-center px-4 py-12 text-center">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex size-12 items-center justify-center rounded-2xl bg-accent-muted text-accent"
      >
        <Sparkles className="size-6" />
      </motion.div>

      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.05 }}
        className="mt-4 text-2xl font-semibold tracking-tight text-foreground"
      >
        Imagination 2.1 Pro
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
        className="mt-2 max-w-md text-sm text-muted-foreground"
      >
        Pick a reasoning effort above, then start a conversation. Higher effort trades speed for
        deeper, structured thinking before each answer.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.15 }}
        className="mt-6 grid w-full max-w-md grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4"
      >
        {EFFORT_LEVELS.filter((l) => !l.disabled).map((l) => (
          <div key={l.level} className="rounded-lg border border-border bg-surface px-2.5 py-2 text-left">
            <div className="text-xs font-medium text-foreground">{l.label}</div>
            <div className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
              {l.description}
            </div>
          </div>
        ))}
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.2 }}
        className="mt-8 flex w-full max-w-md flex-col gap-2"
      >
        {EXAMPLE_PROMPTS.map((p) => (
          <button
            key={p}
            onClick={() => onPick(p)}
            className="rounded-xl border border-border bg-surface px-3.5 py-2.5 text-left text-sm text-foreground transition-colors hover:border-accent/40 hover:bg-muted"
          >
            {p}
          </button>
        ))}
      </motion.div>
    </div>
  );
}
