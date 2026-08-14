# Imagination Web

A premium chat interface for **Imagination 2.1 Pro**, a custom QLoRA fine-tune
of Qwen3-Coder-30B-A3B-Instruct served locally via [Ollama](https://ollama.com).

Unlike a generic chat UI, this app owns the model's system-prompt contract:
Imagination 2.1 Pro's 5-tier "reasoning effort" system (low/medium/high/max,
with "ultra" planned) isn't a real API parameter — it's controlled entirely by
literal instruction text baked into the system prompt at training time. This
UI builds and injects that exact text on every request.

## How it connects to Imagination 2.1 Pro

Ollama exposes an OpenAI-compatible endpoint at
`http://<host>:11434/v1/chat/completions`. This app never calls that endpoint
from the browser — a Next.js API route (`src/app/api/chat/route.ts`) proxies
and streams the request server-side, so the base URL and any auth token stay
off the client and CORS is never an issue.

The base URL is **not** hardcoded. Because Ollama is commonly tunneled with
`cloudflared tunnel --url http://localhost:11434` (which mints a new random
`https://<random-words>.trycloudflare.com` URL on every restart), the base
URL, model tag, and auth token are all editable at runtime from the
**Settings** panel (gear icon) and persisted in `localStorage`. The
`OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_AUTH_TOKEN` env vars only supply
the defaults used before you've set anything in Settings.

## The system prompt contract

This is the single most important thing to understand before touching
`src/lib/prompt.ts`. Imagination 2.1 Pro was trained to recognize **exact**
instruction text and respond accordingly (e.g. emitting a structured
`<think>` block in a specific format at a given effort level). If the text
sent to the model deviates at all from what it was trained on, its behavior
silently degrades — it won't reliably produce the `<think>` structure, and
downstream parsing/rendering in this app will break too.

The full prompt is built as:

```
MODE_TEMPLATE.replace("{level_instruction}", LEVEL_INSTRUCTIONS[level])
```

and sent as the `system` role message on **every** request, rebuilt whenever
the effort level or mode changes. There are three modes (`direct`,
`orchestrator`, `subagent`) — only `direct` is enabled in the UI for v1, the
other two are stubbed in `src/lib/prompt.ts` and `MODES` (see Phase 2 below).

**Do not paraphrase, reformat, or "clean up" any string in `src/lib/prompt.ts`.**
If the training-side templates in the `Imagination-Datasets` repo's
`schema_templates.py` ever change, update this file to match them exactly.

Adding the 5th effort level ("ultra") once it's trained is a one-line change:
add its text to `LEVEL_INSTRUCTIONS` and flip `disabled: false` on its entry
in `EFFORT_LEVELS`.

## Features (v1)

- Streaming chat against any OpenAI-compatible endpoint (built for Ollama).
- Effort selector (Low/Medium/High/Max, Ultra stubbed) that rebuilds and
  injects the system prompt per the contract above.
- `<think>` blocks parsed out of the streamed response and rendered as a
  collapsible "Reasoning" panel, expanded by default while streaming.
- `tool_calls` and paired `tool` results rendered as their own cards in the
  transcript (no real tool dispatch in v1 — display only).
- Markdown rendering with syntax-highlighted code blocks (Shiki) and
  copy-to-clipboard.
- New conversation, stop/cancel an in-flight generation, dark/light theme
  (dark default), and a graceful "can't reach Imagination 2.1 Pro" state
  with retry when the endpoint is unreachable.
- Settings panel for base URL / model tag / auth token, persisted in
  `localStorage`, with a "Test connection" check against `/v1/models`.
- Conversation history persisted in `localStorage` across reloads.

## Setup

```bash
pnpm install
cp .env.example .env.local   # optional — defaults work for local Ollama
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). Point Settings at your
Ollama instance (defaults to `http://localhost:11434`) and pick a model tag
that's actually pulled (e.g. `ollama pull llama3.2` for a quick smoke test
against a non-Imagination model — the UI and streaming don't care which
model is behind the endpoint, only Imagination 2.1 Pro will actually follow
the `<think>` format the reasoning panel expects).

## Tech stack

- Next.js 16 (App Router) + TypeScript (strict) + Tailwind CSS v4
- Hand-rolled `fetch` + `ReadableStream` SSE client (`src/lib/stream-chat.ts`)
  rather than the Vercel AI SDK's chat hooks — the per-effort-level system
  prompt and literal `<think>`/`tool_calls` parsing needed full control over
  the raw stream, which the SDK's abstractions fought more than they helped.
- Zustand (+ `persist`) for app state: settings, effort/mode, conversation.
- Framer Motion for the effort selector's sliding indicator, message
  entrance, and reasoning-panel expand/collapse.
- `react-markdown` + `remark-gfm` + Shiki for chat content.
- `next-themes` for dark/light mode.
- Hand-built Tailwind component primitives (button, tooltip, sheet, input)
  rather than the shadcn/ui CLI scaffold, to keep the palette and feel
  intentional rather than default-template.

## Project structure

```
src/
  app/
    api/chat/route.ts     # streaming proxy to Ollama, injects system prompt
    api/health/route.ts   # connectivity check used by Settings "Test connection"
    page.tsx              # chat shell
  components/
    chat/                 # transcript, message, reasoning panel, tool cards, composer
    effort/                # the effort selector (segmented control)
    settings/              # settings panel
    theme/, ui/             # theme provider/toggle, shared primitives
  hooks/use-chat.ts        # streaming state machine (send/stop/retry)
  lib/
    prompt.ts              # the system prompt contract — read this first
    stream-chat.ts          # SSE client
    store.ts, types.ts      # app state
```

## Phase 2 (stubbed, not built)

- Mode selector UI (Direct/Orchestrator/Subagent) — the prompt builder
  already accepts a `mode` param, so this is a UI-only addition later.
- Multiple saved conversations (sidebar, rename/delete).
- "Ultra" effort level once trained.
- Real tool execution/dispatch — v1 only displays what the model emits.

## Linting

```bash
pnpm lint
pnpm build
```
