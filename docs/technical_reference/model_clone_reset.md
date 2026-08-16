# Semantics of `clone()` and `reset()` on Model Objects

## Definitions

- `clone()`: Create an independent runtime instance for a new session. Static resources such as model weights and configuration may be shared, but streaming state must not be shared.
- `reset()`: Reset the runtime state of the current instance. It should clear mutable state such as caches, context, and connection state, but should not rebuild the model instance itself.

## Scope

In X-Talk, most models with session state should implement `clone()`. The main model types that currently have a unified `reset()` semantic are:

- `ASR`
- `VAD`
- `SpeechEnhancer`

`TTS`, `TurnDetector`, and `Agent` do not currently have a unified `reset()` interface. Session isolation for them is usually achieved through `clone()`.

## Implementation Constraints

- After `clone()`, the old and new instances must not share partial results, streaming caches, connection objects, or session context.
- After `reset()`, the current instance must not retain processing state from the previous turn, but it should retain weights, configuration, and shareable static resources.

## Practical Intuition

- `clone()`: create a new "independently runnable session instance"
- `reset()`: restore the "current instance" to its initial reusable state
