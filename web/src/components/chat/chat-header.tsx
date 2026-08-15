"use client";

import { Sparkles, SquarePen, Settings } from "lucide-react";
import { EffortSelector } from "@/components/effort/effort-selector";
import { ThemeToggle } from "@/components/theme/theme-toggle";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import type { EffortLevel } from "@/lib/prompt";

interface ChatHeaderProps {
  effort: EffortLevel;
  onEffortChange: (level: EffortLevel) => void;
  onNewConversation: () => void;
  onOpenSettings: () => void;
}

export function ChatHeader({
  effort,
  onEffortChange,
  onNewConversation,
  onOpenSettings,
}: ChatHeaderProps) {
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center justify-between gap-3 sm:justify-start">
          <div className="flex items-center gap-2">
            <div className="flex size-7 items-center justify-center rounded-lg bg-accent-muted text-accent">
              <Sparkles className="size-4" />
            </div>
            <span className="text-sm font-semibold tracking-tight text-foreground">
              Imagination Web
            </span>
          </div>
          <div className="flex items-center gap-1 sm:hidden">
            <Tooltip content="New conversation">
              <Button variant="ghost" size="icon" onClick={onNewConversation} aria-label="New conversation">
                <SquarePen className="size-4" />
              </Button>
            </Tooltip>
            <Tooltip content="Settings">
              <Button variant="ghost" size="icon" onClick={onOpenSettings} aria-label="Open settings">
                <Settings className="size-4" />
              </Button>
            </Tooltip>
            <ThemeToggle />
          </div>
        </div>

        <div className="flex items-center justify-center overflow-x-auto">
          <EffortSelector value={effort} onChange={onEffortChange} />
        </div>

        <div className="hidden items-center gap-1 sm:flex">
          <Tooltip content="New conversation">
            <Button variant="ghost" size="icon" onClick={onNewConversation} aria-label="New conversation">
              <SquarePen className="size-4" />
            </Button>
          </Tooltip>
          <Tooltip content="Settings">
            <Button variant="ghost" size="icon" onClick={onOpenSettings} aria-label="Open settings">
              <Settings className="size-4" />
            </Button>
          </Tooltip>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
