# Text Whiteboard Plan

## 1. Goal and scope

The desktop whiteboard gives **every conversation its own Markdown text
board**. The AI maintains one document per conversation by calling four
text-oriented tools. The active conversation's board is rendered as Markdown
in an **independent whiteboard window** that stays above the XTalk main
window, pops up the first time the AI writes to it, and switches content
automatically when the user changes conversations. A dock button to the right
of the start-conversation button shows or hides the window, and the visibility
choice survives conversation reloads.

Hard constraints:

- Every change stays under `app/`; `frontend/` and `src/` are untouched.
- Only the approved `xtalk.*` public APIs are imported (the existing
  `app/scripts/verify_boundaries.py` allowlist).
- The window is read-only for the user; editing stays model-driven.

## 2. Key design decisions

### 2.1 Per-conversation store

`app/backend/whiteboard_store.py` keeps one store per conversation:

- A registry maps a session id to its own `text`, `revision`, and
  `updated_at`, each persisted as `whiteboards/<session>.json` under the tool
  data directory, so boards survive sidecar restarts and never mix between
  conversations.
- `add_text` joins blocks with a single newline; `delete_text` and
  `update_text` remove or replace every exact match and normalize the
  resulting newlines.
- The tool UI wrapper stamps the currently bound conversation into each tool
  call's state, and the read-only sidecar endpoint serves the same
  session-keyed store, so the window never depends on the emit channel for
  content.

### 2.2 Independent window plus a dock toggle

- A Tauri `WebviewWindow` labelled `whiteboard` loads `whiteboard.html`, is
  created as a child of the main window so it stays above it, polls
  `GET /app/api/whiteboard?session_id=...` (launch token + Origin), and renders
  the returned Markdown with a small dependency-free renderer. The window
  reads the active conversation from the main window's persisted session key
  and follows conversation switches.
- The window has no in-page title or revision badge; the native window title
  follows `t("whiteboard.windowTitle")` through i18n.
- The main window shows a whiteboard button immediately to the right of the
  start-conversation button. Clicking it toggles the window; the visibility is
  persisted in `localStorage` and restored when a conversation is reloaded.
- Closing the window hides it (the webview survives), and Rust emits
  `whiteboard-window-hidden` so the dock button stays in sync.
- Switching to another conversation (including a brand-new one) collapses the
  window (and remembers that choice), so the previous board never follows
  into the newly opened chat; the first whiteboard tool emit there auto-opens
  it again.
- The first whiteboard tool emit auto-opens the window; later emits only keep
  an already-open window in place.

## 3. Tool contract

All four tools share one description, currently:

> 在需要教学、演示的场景考虑调用该工具。
> Consider using this tool when teaching or demonstrating.

### 3.1 Tool schemas

| tool | input | behavior |
| --- | --- | --- |
| `fetch_text` | - | return the full document text and revision |
| `add_text` | `text` | append one Markdown block to the end |
| `delete_text` | `text` | delete every exact match; error when absent |
| `update_text` | `from`, `to` | replace every exact `from` match with `to`; error when absent |

Limits: ≤ 50 000 characters per operation, and the emit payload must stay
≤ 256 KiB. Every call returns the full normalized snapshot
`{action, success, text, revision, message}`.

### 3.2 API snapshot

`GET /app/api/whiteboard?session_id=...` returns the same document as the
tools see for that conversation:

```json
{
  "version": 1,
  "text": "# 本周计划\n- 目标",
  "revision": 3,
  "updated_at": "…"
}
```

## 4. File-by-file changes

- `app/backend/whiteboard_store.py` (new) — global store, JSON persistence,
  singleton helpers.
- `app/backend/whiteboard_tool.py` — mirrored copy used by unit tests.
- `app/resources/tools/whiteboard/whiteboard_tool.py` — four `AsyncTool`
  classes sharing the description above; `structured_payload = True`.
- `app/resources/tools/whiteboard/ui/index.html` — minimal sandboxed fallback
  that renders the text snapshot (the window is the primary surface).
- `app/backend/runtime.py` — `GET /app/api/whiteboard`, protected by the
  launch token like the other `/app/api` routes; the `session_id` query
  parameter selects the conversation's board.
- `app/src-tauri/src/whiteboard.rs` (new) + `lib.rs` — show/hide/set/query
  commands, window creation (parented to the main window), close-request
  interception.
- `app/src-tauri/build.rs` + `capabilities/main.json` + new
  `capabilities/whiteboard.json` + `tauri.conf.json` — ACL permissions for the
  four window commands and the whiteboard window's backend polling.
- `app/backend/tool_ui.py` — stamps the bound conversation id into tool-call
  state so whiteboard tools address the right session store.
- `app/ui/whiteboard.html` + `whiteboard.ts` (new) — window page that polls the
  endpoint and renders Markdown (paused while hidden).
- `app/ui/markdown.ts` (new) — safe, dependency-free Markdown renderer.
- `app/ui/whiteboard-window.ts` (new) — Tauri invoke/event helpers and the
  visibility preference.
- `app/ui/index.html` + `main.ts` + `styles.css` — whiteboard dock button,
  first-emit auto-open, hidden-event sync, visibility restore.
- `app/ui/i18n.ts` — whiteboard labels (zh/en).
- `app/vite.config.ts` — second `whiteboard.html` entry.

## 5. Testing

- Rewritten `app/tests/unit/test_whiteboard_tool.py`: append/delete/update
  semantics, per-session isolation, JSON persistence across store restarts,
  `from` alias validation, limits, structured emit payload.
- `app/tests/unit/test_tool_registry.py` expects the four new tool names.
- `app/tests/unit/test_runtime.py` covers the authenticated whiteboard
  endpoint (401 without the launch token).
- Gates: `python -m pytest`, `npm run check`, `npm run build`, `cargo check`.
- Manual acceptance: ask the AI to add/update/delete text across turns; the
  window pops on the first call, the button toggles it, Markdown renders, and
  the visibility/content persist across conversation reloads.

## 6. Security and boundaries

- Everything stays under `app/`; only the approved `xtalk.models.agents.tools`
  API is used.
- The Markdown renderer escapes every raw character first and only reinserts
  pre-escaped placeholders; links are restricted to `http(s)` targets, so
  board text is never treated as HTML.
- The whiteboard endpoint is authenticated with the launch token and Origin
  checks; the window is a trusted App page, not an untrusted tool frame.
- Limits keep a single operation and the emit payload bounded.

## 7. Roadmap

- **Done in this plan:** global text store, four tools, API endpoint,
  independent window, dock toggle, Markdown rendering, persistence, and
  collapsing the window when switching conversations.
- **Future:** user editing of the board (needs a write endpoint and a
  security review), or per-session boards if a single global board is no
  longer sufficient.
