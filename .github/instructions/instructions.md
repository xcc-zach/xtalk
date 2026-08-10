---
applyTo: "**"
excludeAgent: "coding-agent"
---
Review checklist:
- Event-driven only: modules under src/xtalk/serving/modules talk via events, never direct cross-calls or WebSocket writes; add new events in serving/events.py and subscribe/publish with event bus in each module.
- Session scope: clone Pipeline per session; all managers take session_id + config; no shared mutable state across sessions.
- Audio contracts: ASR in = PCM s16le mono 16 kHz; TTS out = PCM s16le mono 48 kHz.
- Async hygiene: long work off the event loop; handle CancelledError; use `EventDispatchMode.WAIT_UNTIL_COMPLETE` for order-critical publishes; no task leaks on shutdown.
- Turn-taking single source of truth: let TurnTakingManager mediate ASR/TTS/VAD/caption/thought.
- Logging: English log lines
- Pipeline purpose: put all models for manager use in Pipeline (src/xtalk/pipelines/interfaces.py); managers should not load models themselves.
