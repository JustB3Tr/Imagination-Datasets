"use client";

import { AlertTriangle, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ConnectionErrorProps {
  message: string;
  onRetry: () => void;
}

export function ConnectionError({ message, onRetry }: ConnectionErrorProps) {
  return (
    <div className="mx-auto flex w-full max-w-[72ch] items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-foreground">Can&apos;t reach Imagination 2.1 Pro</p>
        <p className="mt-0.5 break-words text-xs text-muted-foreground">{message}</p>
      </div>
      <Button variant="outline" size="sm" onClick={onRetry} className="shrink-0 gap-1.5">
        <RotateCw className="size-3.5" />
        Retry
      </Button>
    </div>
  );
}
