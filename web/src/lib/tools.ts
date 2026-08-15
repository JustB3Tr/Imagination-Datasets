import { execFile } from "node:child_process";
import { lookup } from "node:dns/promises";
import { promises as fs } from "node:fs";
import { isIP } from "node:net";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import type { ToolCall } from "@/lib/types";

const execFileAsync = promisify(execFile);

const DEFAULT_WORKSPACE_ROOT = path.resolve(process.cwd(), "..");
const WORKSPACE_ROOT = path.resolve(process.env.IMAGINATION_WORKSPACE_ROOT || DEFAULT_WORKSPACE_ROOT);
const STORAGE_ROOT = path.resolve(process.env.IMAGINATION_TOOL_STORAGE_ROOT || path.join(os.homedir(), ".imagination-ui"));
const STORAGE_FILES_ROOT = path.join(STORAGE_ROOT, "files");
const COMMAND_SANDBOX_ROOT = path.join(STORAGE_ROOT, "sandbox");
const COMMAND_TIMEOUT_MS = 30_000;
const MAX_COMMAND_OUTPUT_CHARS = 12_000;
const MAX_FILE_BYTES = 128 * 1024;
const MAX_LIST_ENTRIES = 200;
const MAX_WEB_RESULTS = 8;
const SAFE_ENV_KEYS = ["PATH", "PATHEXT", "SystemRoot", "ComSpec", "WINDIR", "HOME", "USERPROFILE", "TMP", "TEMP"] as const;

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
          scope: {
            type: "string",
            enum: ["workspace", "storage"],
            description: "Read from the repository workspace or persistent user storage.",
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
          scope: {
            type: "string",
            enum: ["workspace", "storage"],
            description: "Read from the repository workspace or persistent user storage.",
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
            description: "Required relative path to the file inside persistent user storage.",
          },
          content: {
            type: "string",
            description: "Required full file contents to write.",
          },
          scope: {
            type: "string",
            enum: ["storage"],
            description: "Writes are always saved into persistent user storage.",
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

function isPrivateIpv4(host: string) {
  return (
    host === "0.0.0.0" ||
    host.startsWith("10.") ||
    host.startsWith("127.") ||
    host.startsWith("169.254.") ||
    host.startsWith("172.16.") ||
    host.startsWith("172.17.") ||
    host.startsWith("172.18.") ||
    host.startsWith("172.19.") ||
    host.startsWith("172.20.") ||
    host.startsWith("172.21.") ||
    host.startsWith("172.22.") ||
    host.startsWith("172.23.") ||
    host.startsWith("172.24.") ||
    host.startsWith("172.25.") ||
    host.startsWith("172.26.") ||
    host.startsWith("172.27.") ||
    host.startsWith("172.28.") ||
    host.startsWith("172.29.") ||
    host.startsWith("172.30.") ||
    host.startsWith("172.31.") ||
    host.startsWith("192.168.")
  );
}

function isPrivateIpv6(host: string) {
  const normalized = host.toLowerCase();
  return normalized === "::1" || normalized.startsWith("fc") || normalized.startsWith("fd") || normalized.startsWith("fe80:");
}

async function assertSafeRemoteUrl(url: URL) {
  const host = url.hostname.toLowerCase();
  if (host === "localhost" || host.endsWith(".localhost")) {
    throw new Error("Localhost URLs are not allowed.");
  }

  const ipVersion = isIP(host);
  if (ipVersion === 4 && isPrivateIpv4(host)) {
    throw new Error("Private IPv4 addresses are not allowed.");
  }
  if (ipVersion === 6 && isPrivateIpv6(host)) {
    throw new Error("Private IPv6 addresses are not allowed.");
  }

  if (ipVersion === 0) {
    const records = await lookup(host, { all: true });
    if (records.length === 0) throw new Error("Could not resolve hostname.");
    for (const record of records) {
      if (
        (record.family === 4 && isPrivateIpv4(record.address)) ||
        (record.family === 6 && isPrivateIpv6(record.address))
      ) {
        throw new Error("Resolved address is private and is not allowed.");
      }
    }
  }
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

type FileScope = "workspace" | "storage";

function resolveScopedPath(root: string, inputPath = ".") {
  const candidate = path.resolve(root, inputPath);
  const relative = path.relative(root, candidate);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("Path must stay inside the workspace root.");
  }
  return candidate;
}

function getScopeRoot(scope: FileScope) {
  return scope === "storage" ? STORAGE_FILES_ROOT : WORKSPACE_ROOT;
}

function getScope(args: Record<string, unknown>, defaultScope: FileScope): FileScope {
  return args.scope === "storage" ? "storage" : defaultScope;
}

async function ensureStorageRoots() {
  await fs.mkdir(STORAGE_FILES_ROOT, { recursive: true });
  await fs.mkdir(COMMAND_SANDBOX_ROOT, { recursive: true });
}

async function listFiles(args: unknown) {
  const input = typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
  const scope = getScope(input, "workspace");
  if (scope === "storage") await ensureStorageRoots();
  const root = getScopeRoot(scope);
  const inputPath = typeof input.path === "string" ? input.path : ".";
  const target = resolveScopedPath(root, inputPath);
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
      root,
      scope,
      path: path.relative(root, target) || ".",
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
  const scope = getScope(input, "workspace");
  if (scope === "storage") await ensureStorageRoots();
  const root = getScopeRoot(scope);
  const target = resolveScopedPath(root, ensureString(input.path, "path"));
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
  await ensureStorageRoots();
  const target = resolveScopedPath(STORAGE_FILES_ROOT, ensureString(input.path, "path"));
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
      scope: "storage",
      root: STORAGE_FILES_ROOT,
      path: path.relative(STORAGE_FILES_ROOT, target),
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
  await ensureStorageRoots();
  const isWindows = process.platform === "win32";
  const file = isWindows ? "powershell.exe" : "bash";
  const shellArgs = isWindows
    ? ["-NoProfile", "-NonInteractive", "-Command", command]
    : ["-lc", command];
  const env = Object.fromEntries(
    SAFE_ENV_KEYS.flatMap((key) => {
      if (key === "HOME" || key === "USERPROFILE") return [[key, COMMAND_SANDBOX_ROOT]];
      if ((key === "TMP" || key === "TEMP") && process.env[key]) return [[key, COMMAND_SANDBOX_ROOT]];
      return process.env[key] ? [[key, process.env[key]!]] : [];
    }),
  );
  const { stdout, stderr } = await execFileAsync(file, shellArgs, {
    cwd: COMMAND_SANDBOX_ROOT,
    env,
    timeout: COMMAND_TIMEOUT_MS,
    maxBuffer: 1024 * 1024,
  });
  return JSON.stringify(
    {
      sandboxRoot: COMMAND_SANDBOX_ROOT,
      output: truncate([stdout, stderr].filter(Boolean).join("\n") || "(no output)"),
    },
    null,
    2,
  );
}

async function fetchUrl(args: unknown) {
  if (typeof args !== "object" || args === null) throw new Error('Expected an object with "url".');
  const input = args as Record<string, unknown>;
  const rawUrl = ensureString(input.url, "url");
  const url = new URL(rawUrl);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("URL must use http or https.");
  }
  await assertSafeRemoteUrl(url);
  const res = await fetch(url, {
    redirect: "manual",
    headers: {
      "user-agent": "imagination-ui/0.1",
      accept: "text/plain,text/markdown,text/html,application/json;q=0.9,*/*;q=0.8",
    },
  });
  if (res.status >= 300 && res.status < 400) {
    throw new Error(`Redirects are blocked for safety. Received ${res.status} with Location: ${res.headers.get("location") ?? "(none)"}.`);
  }
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

async function webSearch(args: unknown) {
  if (typeof args !== "object" || args === null) throw new Error('Expected an object with "query".');
  const input = args as Record<string, unknown>;
  const query = ensureString(input.query, "query");
  const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
  const res = await fetch(url, {
    headers: {
      "user-agent": "imagination-ui/0.1",
      accept: "application/json",
    },
  });
  const data = (await res.json()) as {
    AbstractText?: string;
    AbstractURL?: string;
    RelatedTopics?: Array<{ Text?: string; FirstURL?: string; Topics?: Array<{ Text?: string; FirstURL?: string }> }>;
  };
  const results: Array<{ title: string; url: string }> = [];
  if (data.AbstractText && data.AbstractURL) {
    results.push({ title: data.AbstractText, url: data.AbstractURL });
  }
  for (const topic of data.RelatedTopics ?? []) {
    const candidates = "Topics" in topic && Array.isArray(topic.Topics) ? topic.Topics : [topic];
    for (const candidate of candidates) {
      if (candidate.Text && candidate.FirstURL) {
        results.push({ title: candidate.Text, url: candidate.FirstURL });
      }
      if (results.length >= MAX_WEB_RESULTS) break;
    }
    if (results.length >= MAX_WEB_RESULTS) break;
  }
  return JSON.stringify(
    {
      query,
      results: results.slice(0, MAX_WEB_RESULTS),
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
