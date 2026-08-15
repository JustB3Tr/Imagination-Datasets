import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import type { ToolCall } from "@/lib/types";

const execFileAsync = promisify(execFile);

const DEFAULT_WORKSPACE_ROOT = path.resolve(process.cwd(), "..");
const WORKSPACE_ROOT = path.resolve(process.env.IMAGINATION_WORKSPACE_ROOT || DEFAULT_WORKSPACE_ROOT);
const COMMAND_TIMEOUT_MS = 30_000;
const MAX_COMMAND_OUTPUT_CHARS = 12_000;
const MAX_FILE_BYTES = 128 * 1024;
const MAX_LIST_ENTRIES = 200;
const MAX_WEB_RESULTS = 8;

interface ParsedToolCall {
  id: string;
  name: string;
  args: unknown;
}

interface ToolExecutionResult {
  toolCallId: string;
  content: string;
}

export const CHAT_TOOLS = [
  {
    type: "function",
    function: {
      name: "list_files",
      description: "List files and directories under the workspace or a relative subdirectory.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Optional relative path inside the workspace. Defaults to the workspace root.",
          },
        },
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read a UTF-8 text file from the workspace.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Required relative path to the file inside the workspace.",
          },
        },
        required: ["path"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description: "Create or overwrite a UTF-8 text file inside the workspace.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Required relative path to the file inside the workspace.",
          },
          content: {
            type: "string",
            description: "Required full file contents to write.",
          },
          append: {
            type: "boolean",
            description: "When true, append to the file instead of overwriting it.",
          },
        },
        required: ["path", "content"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_command",
      description:
        "Run a shell command inside the workspace. Use PowerShell on Windows and bash on Unix-like systems.",
      parameters: {
        type: "object",
        properties: {
          command: {
            type: "string",
            description: "Required shell command to execute.",
          },
        },
        required: ["command"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fetch_url",
      description: "Fetch a web page or API response from a URL and return a truncated text body.",
      parameters: {
        type: "object",
        properties: {
          url: {
            type: "string",
            description: "Required absolute http or https URL.",
          },
        },
        required: ["url"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web and return a short list of result titles and URLs.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Required natural-language search query.",
          },
        },
        required: ["query"],
        additionalProperties: false,
      },
    },
  },
] as const;

function truncate(text: string, limit = MAX_COMMAND_OUTPUT_CHARS) {
  return text.length <= limit ? text : `${text.slice(0, limit)}\n...[truncated]`;
}

function ensureString(value: unknown, name: string) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`"${name}" must be a non-empty string.`);
  }
  return value;
}

function ensureText(value: unknown, name: string) {
  if (typeof value !== "string") {
    throw new Error(`"${name}" must be a string.`);
  }
  return value;
}

function resolveWorkspacePath(inputPath = ".") {
  const candidate = path.resolve(WORKSPACE_ROOT, inputPath);
  const relative = path.relative(WORKSPACE_ROOT, candidate);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("Path must stay inside the workspace root.");
  }
  return candidate;
}

async function listFiles(args: unknown) {
  const input = typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
  const inputPath =
    typeof input.path === "string"
      ? input.path
      : ".";
  const target = resolveWorkspacePath(inputPath);
  const entries = await fs.readdir(target, { withFileTypes: true });
  const items = entries
    .sort((a, b) => a.name.localeCompare(b.name))
    .slice(0, MAX_LIST_ENTRIES)
    .map((entry) => ({
      name: entry.name,
      type: entry.isDirectory() ? "dir" : entry.isFile() ? "file" : "other",
    }));
  return JSON.stringify(
    {
      workspaceRoot: WORKSPACE_ROOT,
      path: path.relative(WORKSPACE_ROOT, target) || ".",
      entries: items,
      truncated: entries.length > items.length,
    },
    null,
    2,
  );
}

async function readFile(args: unknown) {
  if (typeof args !== "object" || args === null) throw new Error('Expected an object with "path".');
  const input = args as Record<string, unknown>;
  const target = resolveWorkspacePath(ensureString(input.path, "path"));
  const stat = await fs.stat(target);
  if (!stat.isFile()) throw new Error("Path is not a file.");
  if (stat.size > MAX_FILE_BYTES) throw new Error(`File is too large to read (${stat.size} bytes).`);
  return await fs.readFile(target, "utf8");
}

async function writeFile(args: unknown) {
  if (typeof args !== "object" || args === null) {
    throw new Error('Expected an object with "path" and "content".');
  }
  const input = args as Record<string, unknown>;
  const target = resolveWorkspacePath(ensureString(input.path, "path"));
  const content = ensureText(input.content, "content");
  const append = Boolean(input.append);
  const bytes = Buffer.byteLength(content, "utf8");
  if (bytes > MAX_FILE_BYTES) throw new Error(`Content is too large to write (${bytes} bytes).`);
  await fs.mkdir(path.dirname(target), { recursive: true });
  if (append) {
    await fs.appendFile(target, content, "utf8");
  } else {
    await fs.writeFile(target, content, "utf8");
  }
  return JSON.stringify(
    {
      ok: true,
      path: path.relative(WORKSPACE_ROOT, target),
      bytesWritten: bytes,
      mode: append ? "append" : "overwrite",
    },
    null,
    2,
  );
}

async function runCommand(args: unknown) {
  if (typeof args !== "object" || args === null) throw new Error('Expected an object with "command".');
  const input = args as Record<string, unknown>;
  const command = ensureString(input.command, "command");
  const isWindows = process.platform === "win32";
  const file = isWindows ? "powershell.exe" : "bash";
  const shellArgs = isWindows
    ? ["-NoProfile", "-NonInteractive", "-Command", command]
    : ["-lc", command];
  const { stdout, stderr } = await execFileAsync(file, shellArgs, {
    cwd: WORKSPACE_ROOT,
    timeout: COMMAND_TIMEOUT_MS,
    maxBuffer: 1024 * 1024,
  });
  return truncate([stdout, stderr].filter(Boolean).join(stderr && stdout ? "\n" : "") || "(no output)");
}

async function fetchUrl(args: unknown) {
  if (typeof args !== "object" || args === null) throw new Error('Expected an object with "url".');
  const input = args as Record<string, unknown>;
  const rawUrl = ensureString(input.url, "url");
  const url = new URL(rawUrl);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("URL must use http or https.");
  }
  const res = await fetch(url, {
    headers: {
      "user-agent": "imagination-ui/0.1",
      accept: "text/plain,text/markdown,text/html,application/json;q=0.9,*/*;q=0.8",
    },
  });
  const body = truncate(await res.text());
  return JSON.stringify(
    {
      url: url.toString(),
      status: res.status,
      statusText: res.statusText,
      body,
    },
    null,
    2,
  );
}

function decodeHtmlEntities(text: string) {
  return text
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

async function webSearch(args: unknown) {
  if (typeof args !== "object" || args === null) throw new Error('Expected an object with "query".');
  const input = args as Record<string, unknown>;
  const query = ensureString(input.query, "query");
  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
  const res = await fetch(url, {
    headers: {
      "user-agent": "imagination-ui/0.1",
      accept: "text/html,application/xhtml+xml",
    },
  });
  const html = await res.text();
  const results: Array<{ title: string; url: string }> = [];
  const pattern = /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(html)) && results.length < MAX_WEB_RESULTS) {
    const href = decodeHtmlEntities(match[1]);
    const title = decodeHtmlEntities(match[2].replace(/<[^>]+>/g, "").trim());
    if (href && title) results.push({ title, url: href });
  }
  return JSON.stringify(
    {
      query,
      results,
    },
    null,
    2,
  );
}

async function executeNamedTool(name: string, args: unknown) {
  switch (name) {
    case "list_files":
      return await listFiles(args);
    case "read_file":
      return await readFile(args);
    case "write_file":
      return await writeFile(args);
    case "run_command":
      return await runCommand(args);
    case "fetch_url":
      return await fetchUrl(args);
    case "web_search":
      return await webSearch(args);
    default:
      throw new Error(`Tool "${name}" is not supported.`);
  }
}

function parseToolCall(toolCall: ToolCall): ParsedToolCall {
  let args: unknown = {};
  if (toolCall.function.arguments.trim()) {
    try {
      args = JSON.parse(toolCall.function.arguments);
    } catch {
      throw new Error(`Tool "${toolCall.function.name}" arguments must be valid JSON.`);
    }
  }
  return {
    id: toolCall.id,
    name: toolCall.function.name,
    args,
  };
}

export async function executeToolCalls(toolCalls: ToolCall[]): Promise<ToolExecutionResult[]> {
  const results: ToolExecutionResult[] = [];
  for (const toolCall of toolCalls) {
    const parsed = parseToolCall(toolCall);
    try {
      const content = await executeNamedTool(parsed.name, parsed.args);
      results.push({ toolCallId: parsed.id, content });
    } catch (error) {
      results.push({
        toolCallId: parsed.id,
        content: `Tool "${parsed.name}" failed: ${error instanceof Error ? error.message : String(error)}`,
      });
    }
  }
  return results;
}
