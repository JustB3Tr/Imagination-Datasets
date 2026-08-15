import { NextRequest } from "next/server";
import type { ToolCall } from "@/lib/types";
import { executeToolCalls } from "@/lib/tools";

export const runtime = "nodejs";

interface ToolExecutionBody {
  toolCalls?: ToolCall[];
}

function isAuthorizedRequest(req: NextRequest) {
  const origin = req.headers.get("origin");
  const host = req.headers.get("host");
  if (!origin || !host) return false;
  try {
    const originUrl = new URL(origin);
    return originUrl.host === host && (originUrl.protocol === "http:" || originUrl.protocol === "https:");
  } catch {
    return false;
  }
}

export async function POST(req: NextRequest) {
  if (!isAuthorizedRequest(req)) {
    return Response.json({ error: "Forbidden." }, { status: 403 });
  }

  let body: ToolExecutionBody;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  if (!Array.isArray(body.toolCalls)) {
    return Response.json({ error: 'Missing "toolCalls" array.' }, { status: 400 });
  }

  const results = await executeToolCalls(body.toolCalls);
  return Response.json({ results });
}
