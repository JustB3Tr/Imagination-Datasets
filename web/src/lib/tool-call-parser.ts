import type { ToolCall } from "@/lib/types";

interface ParsedToolCalls {
  toolCalls: ToolCall[];
  cleanContent: string;
}

function makeToolCall(name: string, args: unknown, index: number): ToolCall {
  return {
    id: `parsed_tool_${index + 1}`,
    type: "function",
    function: {
      name,
      arguments: JSON.stringify(args ?? {}),
    },
  };
}

function tryParseJson(raw: string): unknown | null {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function extractXmlArgs(body: string) {
  const nestedArgs: Record<string, string> = {};
  const nestedPattern = /<(arg|parameter|property)\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/\1>/gi;
  let nestedMatch: RegExpExecArray | null;
  while ((nestedMatch = nestedPattern.exec(body))) {
    nestedArgs[nestedMatch[2]] = nestedMatch[3].trim();
  }
  if (Object.keys(nestedArgs).length > 0) return nestedArgs;

  const parsedJson = tryParseJson(body.trim());
  if (parsedJson !== null) return parsedJson;

  const trimmed = body.trim();
  return trimmed ? { input: trimmed } : {};
}

function parseToolPayload(payload: unknown): Array<{ name: string; args: unknown }> {
  if (Array.isArray(payload)) {
    return payload.flatMap(parseToolPayload);
  }
  if (!payload || typeof payload !== "object") return [];

  const record = payload as Record<string, unknown>;

  if (typeof record.name === "string") {
    if (record.arguments && typeof record.arguments === "object") {
      return [{ name: record.name, args: record.arguments }];
    }
    if (typeof record.arguments === "string") {
      return [{ name: record.name, args: tryParseJson(record.arguments) ?? { input: record.arguments } }];
    }
    if (record.input && typeof record.input === "object") {
      return [{ name: record.name, args: record.input }];
    }
    if (typeof record.input === "string") {
      return [{ name: record.name, args: tryParseJson(record.input) ?? { input: record.input } }];
    }
    return [{ name: record.name, args: {} }];
  }

  if (Array.isArray(record.tool_calls)) {
    return record.tool_calls.flatMap((toolCall) => {
      if (!toolCall || typeof toolCall !== "object") return [];
      const tc = toolCall as Record<string, unknown>;
      if (tc.function && typeof tc.function === "object") {
        const fn = tc.function as Record<string, unknown>;
        if (typeof fn.name === "string") {
          const args =
            typeof fn.arguments === "string" ? tryParseJson(fn.arguments) ?? { input: fn.arguments } : fn.arguments ?? {};
          return [{ name: fn.name, args }];
        }
      }
      return [];
    });
  }

  return [];
}

function parseJsonBlocks(content: string) {
  const toolCalls: ToolCall[] = [];
  let cleanContent = content;
  const blocks: string[] = [];

  const fencePattern = /```(?:json|tool_call|tool_calls|function_call)?\s*([\s\S]*?)```/gi;
  let fenceMatch: RegExpExecArray | null;
  while ((fenceMatch = fencePattern.exec(content))) {
    const parsed = tryParseJson(fenceMatch[1].trim());
    const extracted = parseToolPayload(parsed);
    if (extracted.length === 0) continue;
    extracted.forEach((tool, index) => toolCalls.push(makeToolCall(tool.name, tool.args, toolCalls.length + index)));
    blocks.push(fenceMatch[0]);
  }

  const fullJson = tryParseJson(content.trim());
  const fullJsonTools = parseToolPayload(fullJson);
  if (blocks.length === 0 && fullJsonTools.length > 0) {
    fullJsonTools.forEach((tool, index) => toolCalls.push(makeToolCall(tool.name, tool.args, toolCalls.length + index)));
    cleanContent = "";
  } else {
    for (const block of blocks) cleanContent = cleanContent.replace(block, "");
  }

  return { toolCalls, cleanContent };
}

function parseXmlBlocks(content: string) {
  const toolCalls: ToolCall[] = [];
  let cleanContent = content;

  const toolUsePattern = /<tool_use>([\s\S]*?)<\/tool_use>/gi;
  let toolUseMatch: RegExpExecArray | null;
  while ((toolUseMatch = toolUsePattern.exec(content))) {
    const parsed = tryParseJson(toolUseMatch[1].trim());
    const extracted = parseToolPayload(parsed);
    extracted.forEach((tool, index) => toolCalls.push(makeToolCall(tool.name, tool.args, toolCalls.length + index)));
    if (extracted.length > 0) cleanContent = cleanContent.replace(toolUseMatch[0], "");
  }

  const xmlPattern = /<(tool_call|tool|function_call|invoke|use_tool)\b([^>]*)>([\s\S]*?)<\/\1>/gi;
  let xmlMatch: RegExpExecArray | null;
  while ((xmlMatch = xmlPattern.exec(content))) {
    const attrs = xmlMatch[2];
    const body = xmlMatch[3];
    const nameMatch = /\bname="([^"]+)"/i.exec(attrs);
    if (!nameMatch) continue;
    toolCalls.push(makeToolCall(nameMatch[1], extractXmlArgs(body), toolCalls.length));
    cleanContent = cleanContent.replace(xmlMatch[0], "");
  }

  return { toolCalls, cleanContent };
}

export function parseToolCallsFromContent(content: string): ParsedToolCalls {
  const xmlParsed = parseXmlBlocks(content);
  const jsonParsed = parseJsonBlocks(xmlParsed.cleanContent);
  const toolCalls = [...xmlParsed.toolCalls, ...jsonParsed.toolCalls];
  return {
    toolCalls,
    cleanContent: jsonParsed.cleanContent.trim(),
  };
}
