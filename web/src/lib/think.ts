export interface ParsedContent {
  think: string | null;
  /** false while the <think> block is still being streamed in (no closing tag yet). */
  thinkComplete: boolean;
  answer: string;
}

const OPEN = "<think>";
const CLOSE = "</think>";

/**
 * Splits a raw assistant message into its optional leading <think> block
 * and the remaining answer text. Tolerant of partial/streaming content:
 * if the opening tag has arrived but the closing tag hasn't yet, the think
 * block is returned as in-progress and the answer is empty so far.
 */
export function parseThinkContent(content: string): ParsedContent {
  const trimmedStart = content.replace(/^\s+/, "");
  if (!trimmedStart.startsWith(OPEN)) {
    return { think: null, thinkComplete: true, answer: content };
  }

  const afterOpen = trimmedStart.slice(OPEN.length);
  const closeIdx = afterOpen.indexOf(CLOSE);

  if (closeIdx === -1) {
    return { think: afterOpen, thinkComplete: false, answer: "" };
  }

  const think = afterOpen.slice(0, closeIdx);
  const answer = afterOpen.slice(closeIdx + CLOSE.length).replace(/^\s+/, "");
  return { think, thinkComplete: true, answer };
}
