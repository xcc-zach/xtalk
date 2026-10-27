# Turn Detection Pipeline

A turn detector decides when the user has finished speaking and when the user is interrupting. X-Talk feeds it several signals through one `async_detect` method so that audio-driven and text-driven detectors share the same service-layer contract.

## Input combinations

`TurnDetectorManager` calls `async_detect` with different subsets of arguments depending on what just happened:

- Only `audio`: the pure audio-based detection path.
- Only `text` and `speech_pause`: the text-semantic detection path.
- Only `assistant_text`: update the detector's context with played assistant text.
- Only `speech_start=True`: notify the detector that the current speaking turn has started.

Two implementations in the repository represent the two main paths. `SoulxDuplug` is primarily audio-based and falls back on text pause signals; `LLMTurnDetector` is primarily text-semantic and relies on `text` and `speech_pause`. A detector may accept `assistant_text` without using it; assistant-aware detectors store the cumulative value as session state and use it in later decisions.

## Assistant response updates

`TurnDetectorManager` subscribes to `ResponseUpdate` and `ResponseFinish`. Each event causes one standalone call to `async_detect(assistant_text=event.text)`. The return value of this context-only call is ignored and never becomes `STOP_SPEAKING` or `START_GENERATION`.

`ResponseUpdate.text` is a cumulative prefix confirmed as played to the user; `ResponseFinish.text` is the final complete response. A detector that stores this context replaces its stored value rather than appending. The final `ResponseUpdate` and `ResponseFinish` may contain the same text, so updates stay idempotent.

## Listening state

The `listening` flag tells the detector which question it is answering. While the system waits for the user (`listening = True`), the detector decides when to emit `START_GENERATION`. While output is playing (`listening = False`), it decides whether user input is an interruption that warrants `STOP_SPEAKING`. `TurnDetectorManager` flips the flag at TTS playback start and restores it when playback finishes or is interrupted.

## Implementation suggestions

- Keep `action` and `semantic` consistent in every result.
- Session state inside a detector belongs to one instance; clones share no state.
- Treat `speech_pause` as a hint that a pause is occurring now; it carries no reset semantics after a turn ends.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Turn detector interface →](../reference/models/turn_detector.md)** · Reference

-   **[Register a new model →](../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[VAD Pipeline →](vad.md)** · Explanation

</div>
