import { createHighlighter, type Highlighter } from "shiki";

let highlighterPromise: Promise<Highlighter> | null = null;

const LANGS = [
  "typescript",
  "tsx",
  "javascript",
  "jsx",
  "json",
  "python",
  "bash",
  "shell",
  "yaml",
  "markdown",
  "html",
  "css",
  "sql",
  "rust",
  "go",
  "diff",
  "plaintext",
];

export function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: ["github-dark-default", "github-light-default"],
      langs: LANGS,
    });
  }
  return highlighterPromise;
}

export function normalizeLang(lang: string | undefined): string {
  if (!lang) return "plaintext";
  const l = lang.toLowerCase();
  if (l === "sh" || l === "zsh") return "bash";
  if (l === "ts") return "typescript";
  if (l === "js") return "javascript";
  if (l === "py") return "python";
  if (l === "yml") return "yaml";
  if (l === "md") return "markdown";
  return LANGS.includes(l) ? l : "plaintext";
}
