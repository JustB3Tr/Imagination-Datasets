import { NextRequest } from "next/server";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const baseUrl = (req.nextUrl.searchParams.get("baseUrl") || process.env.OLLAMA_BASE_URL || "http://localhost:11434").replace(
    /\/+$/,
    "",
  );
  const authToken = req.nextUrl.searchParams.get("authToken") || process.env.OLLAMA_AUTH_TOKEN || "";

  const headers: Record<string, string> = {};
  if (authToken) headers.authorization = `Bearer ${authToken}`;

  try {
    const res = await fetch(`${baseUrl}/v1/models`, { headers, signal: AbortSignal.timeout(6000) });
    if (!res.ok) {
      return Response.json({ ok: false, error: `${res.status} ${res.statusText}` }, { status: 200 });
    }
    const data = await res.json().catch(() => null);
    const models = Array.isArray(data?.data) ? data.data.map((m: { id: string }) => m.id) : [];
    return Response.json({ ok: true, models });
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return Response.json({ ok: false, error: detail }, { status: 200 });
  }
}
