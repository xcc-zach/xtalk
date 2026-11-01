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

## XTurnixLLM waiting window

`XTurnixLLM` keeps every XTurnix decision and adds a second chance to start a reply when XTurnix keeps listening. XTurnix only starts a reply after a VAD pause, so a VAD that keeps reporting speech, or a model that keeps predicting `<|keep|>`, can leave a finished user waiting indefinitely.

- **Opening.** Each accepted listening decision that does not start a reply opens a new window starting at that moment. This covers `<|keep|>` and a `<|start|>` held back because no pause was seen yet. VAD start and pause events neither open nor close a window.
- **Closing.** The window closes when the user transcript changes, XTurnix makes a new accepted decision, the assistant starts speaking, or a reply starts. Closing cancels all of its unfinished LLM requests and discards their late results.
- **Schedule.** The manager has no timer for detectors, so audio-frame calls drive the schedule. At each multiple of `llm_interval` after the window start, up to `llm_timeout`, the detector sends one background request with the full dialogue and the seconds elapsed. Missed multiples are not backfilled.
- **Concurrency.** New requests are sent whether or not earlier ones have returned. Whichever request in the open window answers `respond` first starts the reply on the next audio frame; the others are cancelled. A `wait`, an unparseable reply, or a failed request ends only that request.
- **Judgement.** The LLM answers `respond` only if the silence is long enough for the user to be waiting and the latest speech is directed at the assistant rather than someone else. With `llm_output_reason: false`, the prompt drops the `reason` field for LLMs that already reason internally.

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
