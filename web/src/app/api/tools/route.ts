import { NextRequest } from "next/server";
import type { ToolCall } from "@/lib/types";
import { executeToolCalls } from "@/lib/tools";

export const runtime = "nodejs";

interface ToolExecutionBody {
  toolCalls?: ToolCall[];
}

export async function POST(req: NextRequest) {
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
