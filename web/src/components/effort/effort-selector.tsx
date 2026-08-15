"use client";

import { motion } from "framer-motion";
import { EFFORT_LEVELS, type EffortLevel } from "@/lib/prompt";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface EffortSelectorProps {
  value: EffortLevel;
  onChange: (level: EffortLevel) => void;
}

export function EffortSelector({ value, onChange }: EffortSelectorProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Reasoning effort"
      className="relative flex items-center gap-0.5 rounded-full border border-border bg-surface p-1"
    >
      {EFFORT_LEVELS.map((cfg) => {
        const active = cfg.level === value;
        return (
          <Tooltip
            key={cfg.level}
            content={
              <div>
                <div className="font-medium">{cfg.label}</div>
                <div className="mt-0.5 text-muted-foreground">{cfg.description}</div>
              </div>
            }
          >
            <button
              type="button"
              role="radio"
              aria-checked={active}
              disabled={cfg.disabled}
              onClick={() => !cfg.disabled && onChange(cfg.level)}
              className={cn(
                "relative z-10 flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-colors duration-150",
                cfg.disabled
                  ? "cursor-not-allowed text-muted-foreground/50"
                  : active
                    ? "text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground",
              )}
            >
              {active && (
                <motion.span
                  layoutId="effort-active-pill"
                  className="absolute inset-0 -z-10 rounded-full bg-accent"
                  transition={{ type: "spring", stiffness: 500, damping: 38 }}
                />
              )}
              <IntensityDots intensity={cfg.intensity} active={active} disabled={cfg.disabled} />
              <span>{cfg.label}</span>
            </button>
          </Tooltip>
        );
      })}
    </div>
  );
}

function IntensityDots({
  intensity,
  active,
  disabled,
}: {
  intensity: number;
  active: boolean;
  disabled?: boolean;
}) {
  const filled = Math.max(1, Math.round(intensity * 4));
  return (
    <span className="flex items-center gap-[2px]" aria-hidden="true">
      {Array.from({ length: 4 }).map((_, i) => (
        <span
          key={i}
          className={cn(
            "h-2.5 w-[3px] rounded-full transition-colors",
            i < filled
              ? active
                ? "bg-accent-foreground"
                : disabled
                  ? "bg-muted-foreground/30"
                  : "bg-accent"
              : "bg-border",
          )}
          style={{ height: `${6 + i * 2.5}px` }}
        />
      ))}
    </span>
  );
}
