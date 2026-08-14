"use client";

import * as React from "react";
import { useTheme } from "next-themes";
import { Check, Copy } from "lucide-react";
import { getHighlighter, normalizeLang } from "@/lib/highlighter";
import { cn } from "@/lib/utils";

interface CodeBlockProps {
  code: string;
  lang?: string;
}

export function CodeBlock({ code, lang }: CodeBlockProps) {
  const { resolvedTheme } = useTheme();
  const [html, setHtml] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState(false);
  const language = normalizeLang(lang);
  const dark = resolvedTheme !== "light";

  React.useEffect(() => {
    let cancelled = false;
    getHighlighter().then((hl) => {
      if (cancelled) return;
      try {
        const out = hl.codeToHtml(code, {
          lang: language,
          theme: dark ? "github-dark-default" : "github-light-default",
        });
        setHtml(out);
      } catch {
        setHtml(null);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [code, language, dark]);

  const onCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="group relative my-3 overflow-hidden rounded-lg border border-border">
      <div className="flex items-center justify-between border-b border-border bg-muted px-3 py-1.5">
        <span className="font-mono text-[11px] text-muted-foreground">{language}</span>
        <button
          onClick={onCopy}
          aria-label="Copy code"
          className={cn(
            "flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-border hover:text-foreground",
          )}
        >
          {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      {html ? (
        <div
          className="shiki-container overflow-x-auto [&_pre]:!bg-transparent [&_pre]:p-3 [&_pre]:text-[13px] [&_pre]:leading-relaxed"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="overflow-x-auto p-3 text-[13px] leading-relaxed text-foreground">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}
