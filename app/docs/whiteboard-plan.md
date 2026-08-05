# Whiteboard Tool Plan

## 1. Goal and scope

During a conversation the AI often needs to keep a structured, evolving
artifact in front of the user: a plan, a brainstorm board, a checklist, or a
meeting outline. The existing tool UI channel (`timer`, `codex`) renders a
status card, not structured content the AI can update across multiple turns.

This plan adds a built-in **whiteboard** tool: the AI calls a tool during the
conversation, and each call pushes a normalized snapshot of sticky-note
content into a session-scoped whiteboard view that re-renders in real time.

Hard constraints:

- Every change stays under `app/`; `frontend/` and `src/` are untouched.
- Only the approved `xtalk.*` public APIs are imported (the existing
  `app/scripts/verify_boundaries.py` allowlist).
- In phase 1 the whiteboard is a read-only view for the user; user editing is
  deferred to a later phase.

## 2. Key design decision

### 2.1 Reuse the Tool UI observation channel (recommended, option A)

The current pipeline already does exactly the hard part:

AI calls an `AsyncTool` -> `wrap_tools_with_ui` -> `ToolUIBroker.publish_emit`
-> in-memory history / WebSocket -> UI polls `/app/api/tool-ui/events` ->
sandboxed `ToolUIFrame` receives `tool_ui.emit` via postMessage.

The whiteboard needs only two small, backward-compatible additions:

1. The emit event gains an optional structured `payload` field.
2. The whiteboard tool declares `structured_payload = True`; the wrapper then
   JSON-decodes the tool's emit content into `payload`.

Advantages: minimal surface, the security boundary is unchanged (frame stays
read-only, CSP unchanged, no new endpoints), session isolation and history
replay come for free.

Trade-offs: the frame height cap limits the board size, in-memory events do not
survive a sidecar restart, and users cannot edit.

### 2.2 Dedicated whiteboard service (option B, later phases)

Add `app/backend/whiteboard_store.py` (per-session persistence in
sqlite/JSON), authenticated `GET/POST /app/api/whiteboard/{session_id}`
endpoints (launch token + Origin), and a trusted UI panel that renders a large
canvas directly. This enables restart persistence, a bigger canvas, and user
editing at the cost of new endpoints and a new panel.

The roadmap is "A first, then B".

## 3. Data contract

### 3.1 Tool input (the schema the LLM sees)

`whiteboard_update` accepts a list of operations (v1 subset):

| op | fields | notes |
| --- | --- | --- |
| `set_title` | `title` | set the board title |
| `add_note` | `note {id?, text, color?}` | add a sticky note; `id` is auto-generated when absent |
| `update_note` | `id`, `text?`/`color?` | update an existing note |
| `remove_note` | `id` | remove a note |
| `clear` | - | clear the board |

Limits: ≤ 200 notes, ≤ 2000 characters per note text, ≤ 50 ops per call, and
the serialized payload must stay ≤ 256 KiB.

### 3.2 Emit content (idempotent snapshot)

Every emit carries the full normalized snapshot as its `message` and attaches
the same object as `payload`:

```json
{
  "version": 1,
  "title": "Weekly plan",
  "revision": 3,
  "notes": [{"id": "n1", "text": "…", "color": "yellow"}],
  "updated_at": "…"
}
```

The full-snapshot semantics keep the renderer stateless: any emit, replay, or
history frame can render completely on its own.

## 4. File-by-file changes

### 4.1 Tool bundle (new)

- `app/resources/tools/whiteboard/xtalk_tool.json` — display name
  `{zh: "白板", en: "Whiteboard"}`, entrypoint `whiteboard_tool:create_tools`,
  `ui: {entrypoint: "ui/index.html", update_every_s: -1}` (no status polling;
  content arrives as emits).
- `app/resources/tools/whiteboard/whiteboard_tool.py` — pydantic op/snapshot
  models, op application, revision counter, limits, and an `AsyncTool` with
  `name = "whiteboard_update"`, `subscribe_by_default = False`, and
  `structured_payload = True`. `emit_initial` applies ops and returns
  `Running(snapshot_json)`; `emit_updates` yields `Finished(snapshot_json)`.
- `app/resources/tools/whiteboard/ui/index.html` — self-contained sandboxed
  frame (see 4.3).
- `app/resources/tools/builtin_tools.json` — register `whiteboard`
  (`enabled_by_default: true`, `can_disable: true`).

Note: `AsyncTool` is required, not `SyncTool`, because the UI observer wraps
only `AsyncTool` subclasses.

### 4.2 Transport protocol (small, backward-compatible changes)

- `app/backend/tool_ui.py`
  - `ToolUIBroker.publish_emit(..., payload=None)` adds an optional
    keyword-only `dict` field.
  - `_wrap_async_tool` attaches `payload` when
    `getattr(original, "structured_payload", False)` and the content parses as
    a JSON object; otherwise it degrades to no payload (message unchanged).
  - Add `MAX_TOOL_UI_EMIT_PAYLOAD_BYTES = 256 * 1024`; oversized payloads are
    dropped while the message is kept.
  - History retention already copies the whole dict, so `payload` rides along
    for free.
- `app/ui/adapters/tool-ui-adapter.ts`
  - `ToolUIEmitEvent` gains `payload?: unknown`; `parseToolUIEvent` accepts
    `undefined` or a plain object and enforces the size limit.
- `app/ui/tool-ui-frame.ts`
  - Protocol unchanged; add a doc note that frames read structured content
    from `event.payload`.
  - Optional UX tweak: a taller live-frame cap for the whiteboard tool so the
    board can use the space (the frame itself scrolls otherwise).

### 4.3 Rendering (new frame)

`ui/index.html` is fully self-contained:

- Renders from `window.xtalkToolUI.emit` events; prefers `event.payload`,
  falls back to parsing `event.message`.
- v1 layout: title plus a sticky-note grid (CSS grid cards with colors), a
  revision/count badge, and an empty state.
- Safety: all note text is rendered with `textContent` (never `innerHTML`);
  CSP and sandbox remain unchanged; auto `reportHeight`.
- Built-in zh/en copy driven by `context.language`.

### 4.4 Registration and mounting

No `main.ts`/`index.html` changes are needed: whiteboard emits automatically
appear in the existing live tool panel and as timeline tool rows; history rows
render the final snapshot of that call because emit payloads are stateless.

## 5. Testing

- New `app/tests/unit/test_whiteboard_tool.py`: op-application matrix,
  revision increments, auto-generated ids, limits and invalid input, idempotent
  snapshots.
- Extend `app/tests/unit/test_tool_ui.py`: `structured_payload` parsing,
  payload retained in emits and history, invalid JSON degrades, legacy events
  without payload stay valid.
- Extend `app/tests/unit/test_runtime.py`: `/app/api/tool-ui/events` snapshots
  include `payload`.
- Gates: `python scripts/verify_boundaries.py`, `npm run check`,
  `python -m pytest`.
- Manual acceptance: ask the AI to add/update/remove notes across turns and
  confirm the live panel updates, sessions are isolated, and history rows
  render correct final snapshots.

## 6. Security and boundaries

- Everything stays under `app/`; only the approved `xtalk.models.agents.tools`
  API is used.
- The iframe keeps `allow-scripts` only, CSP `connect-src 'none'`, and click /
  submit blocking (read-only in v1).
- Payload and note-text limits; note text is rendered via `textContent`.
- Any new endpoint (phase 2) must be authenticated with the launch token and
  Origin checks.

## 7. Roadmap

- **Phase 1 (this plan):** tool + structured payload + frame, end to end.
- **Phase 2:** `whiteboard_store.py`, per-session persistence, authenticated
  `GET /app/api/whiteboard/{session_id}`, and a larger dedicated panel so the
  board survives restarts and uses more space.
- **Phase 3 (optional):** user editing. Recommended path: render the canvas in
  the trusted UI instead of the iframe; alternatively add an authenticated
  write endpoint and relax the sandbox for this one trusted built-in, which
  needs a separate security review.

## 8. Acceptance criteria

- All code and tests live under `app/`; `verify_boundaries.py` passes.
- The AI can update the same board incrementally across turns; different
  sessions never share boards.
- Note text is never treated as HTML.
- Live and history rendering are consistent.
