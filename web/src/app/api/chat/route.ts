import { NextRequest } from "next/server";
import { buildSystemPrompt, type Mode, type EffortLevel } from "@/lib/prompt";

export const runtime = "nodejs";

interface ChatRequestBody {
  messages: Array<{
    role: "system" | "user" | "assistant" | "tool";
    content: string | null;
    tool_calls?: unknown;
    tool_call_id?: string;
  }>;
  mode: Mode;
  effort: EffortLevel;
  baseUrl?: string;
  model?: string;
  authToken?: string;
}

function jsonError(message: string, status: number) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export async function POST(req: NextRequest) {
  let body: ChatRequestBody;
  try {
    body = await req.json();
  } catch {
    return jsonError("Invalid JSON body.", 400);
  }

  const { messages, mode, effort } = body;
  if (!Array.isArray(messages) || !mode || !effort) {
    return jsonError("Missing messages, mode, or effort.", 400);
  }

  let systemPrompt: string;
  try {
    systemPrompt = buildSystemPrompt(mode, effort);
  } catch (err) {
    return jsonError(err instanceof Error ? err.message : "Invalid effort level.", 400);
  }

  const baseUrl = (body.baseUrl || process.env.OLLAMA_BASE_URL || "http://localhost:11434").replace(
    /\/+$/,
    "",
  );
  const model = body.model || process.env.OLLAMA_MODEL || "imagination-2.1-pro-lmh";
  const authToken = body.authToken || process.env.OLLAMA_AUTH_TOKEN || "";

  let upstreamUrl: URL;
  try {
    upstreamUrl = new URL(`${baseUrl}/v1/chat/completions`);
  } catch {
    return jsonError(`"${baseUrl}" is not a valid URL.`, 400);
  }

  const upstreamMessages = [
    { role: "system", content: systemPrompt },
    ...messages.filter((m) => m.role !== "system"),
  ];

  const headers: Record<string, string> = { "content-type": "application/json" };
  if (authToken) headers.authorization = `Bearer ${authToken}`;

  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model,
        messages: upstreamMessages,
        stream: true,
      }),
      signal: req.signal,
    });
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return jsonError(
      `Couldn't reach Imagination 2.1 Pro at ${baseUrl}. Is Ollama running and is the base URL correct? (${detail})`,
      502,
    );
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text().catch(() => "");
    return jsonError(
      `Ollama returned ${upstream.status} ${upstream.statusText}. ${text.slice(0, 500)}`,
      upstream.status || 502,
    );
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
