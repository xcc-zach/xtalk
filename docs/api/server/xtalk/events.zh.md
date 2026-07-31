<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.events

## Event

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class Event
```

Base dataclass for all Xtalk events.

### 参数

- `session_id` (`str`)
  Session identifier associated with the event.

### 属性

- `timestamp` (`float`)
  Unix timestamp recorded when the event instance is created.
- `session_id` (`str`)
  Session identifier associated with the event.
- `TYPE` (`str`)
  Stable event type string used by the event bus.

### 类字段

- `timestamp: float` = `field(init=False)`
- `session_id: str`
- `TYPE: ClassVar[str]` = `'base'`

### 方法

#### event_type

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
def event_type(self) -> str
```

## create_event_class

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
def create_event_class(*, name: str, fields: dict[str, Any] | None = None, type_name: str | None = None) -> Type[Event]
```

Create an ``Event`` subclass dynamically.

### 参数

- `name` (`str`)
  Dataclass name for the generated event type.
- `fields` (`dict[str, Any] | None, optional`)
  Mapping of field names to default values. Value types are inferred from
  the defaults.
- `type_name` (`str | None, optional`)
  Event bus type string. Defaults to ``name.lower()`` when omitted.

### 返回

- `Type[Event]`
  Generated dataclass type inheriting from ``Event``.

### 示例

```pycon
>>> CustomEvent = create_event_class(
...     name="CustomEvent",
...     fields={"text": ""},
... )
```

## WebSocketMessageReceived

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class WebSocketMessageReceived(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'websocket.message_received'`
- `message: str` = `''`

## AudioFrameReceived

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class AudioFrameReceived(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'audio.frame_received'`
- `audio_data: bytes`
- `sample_rate: int` = `16000`

## EnhancedAudioFrameReceived

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class EnhancedAudioFrameReceived(Event)
```

供下游 VAD、说话人识别和轮次检测使用的增强音频帧。

### 类字段

- `TYPE: ClassVar[str]` = `'audio.enhanced_frame_received'`
- `audio_data: bytes`
- `sample_rate: int` = `16000`

## VADSpeechStart

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class VADSpeechStart(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'vad.speech_start'`
- `origin: str` = `'client'`

## VADSpeechEnd

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class VADSpeechEnd(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'vad.speech_end'`
- `origin: str` = `'client'`

## ASRResultPartial

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ASRResultPartial(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'asr.result_partial'`
- `text: str` = `''`
- `display_text: str` = `''`
- `speech_pause: bool` = `False`

## ASRResultFinal

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ASRResultFinal(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'asr.result_final'`
- `text: str` = `''`
- `display_text: str` = `''`

## LLMFirstChunk

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class LLMFirstChunk(Event)
```

Event for first LLM chunk/tool call (measure first token latency).

### 类字段

- `TYPE: ClassVar[str]` = `'llm.first_chunk'`

## LLMFirstSentence

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class LLMFirstSentence(Event)
```

Event for first synthesizable sentence (measure sentence latency).

### 类字段

- `TYPE: ClassVar[str]` = `'llm.sentence_ready'`

## TTSStarted

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSStarted(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.started'`

## TTSStopped

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSStopped(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.stopped'`

## TTSPaused

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSPaused(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.paused'`

## TTSResumed

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSResumed(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.resumed'`

## TTSFinished

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSFinished(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.finished'`

## LLMAgentResponseUpdate

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class LLMAgentResponseUpdate(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'llm_agent.response_update'`
- `text: str` = `''`

## LLMAgentResponseFinish

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class LLMAgentResponseFinish(Event)
```

Final text emitted by the agent for one response.

### 属性

- `text` (`str`)
  Final response text.

### 类字段

- `TYPE: ClassVar[str]` = `'llm_agent.response_finish'`
- `text: str` = `''`

## ResponseUpdate

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ResponseUpdate(Event)
```

Text prefix whose corresponding TTS playback progress has been confirmed.

### 属性

- `text` (`str`)
  Text prefix that has been played to the user.

### 类字段

- `TYPE: ClassVar[str]` = `'response.update'`
- `text: str` = `''`

## ResponseFinish

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ResponseFinish(Event)
```

Final text whose corresponding TTS playback has finished.

### 属性

- `text` (`str`)
  Final response text whose playback completed.

### 类字段

- `TYPE: ClassVar[str]` = `'response.finish'`
- `text: str` = `''`

## TTSTextSynthesized

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSTextSynthesized(Event)
```

Text marker emitted after one synthesized text segment is fully produced.

### 属性

- `text` (`str`)
  Text segment that was synthesized.
- `audio_duration` (`float`)
  Estimated playback duration of the synthesized audio in milliseconds.

### 类字段

- `TYPE: ClassVar[str]` = `'tts.text_synthesized'`
- `text: str` = `''`
- `audio_duration: float` = `0.0`

## TTSVoiceChange

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSVoiceChange(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.reference_audio_changed'`
- `voice_name: str` = `''`

## TTSEmotionChange

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSEmotionChange(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.emotion_changed'`
- `emotion_name: str` = `''`
- `emotion_vector: list` = `None`

## TTSSpeedChange

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSSpeedChange(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.speed_changed'`
- `speed: float` = `1.0`

## TTSChunkReady

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSChunkReady(Event)
```

Indicates one TTS audio chunk is ready for sending. Not emitted when the chunk is generated.

### 类字段

- `TYPE: ClassVar[str]` = `'tts.chunk_ready'`
- `audio_chunk: bytes` = `b''`
- `sample_rate: int` = `48000`

## TTSChunkPlayed

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSChunkPlayed(Event)
```

Frontend confirmed playback completion for a TTS audio chunk.

InputGateway publishes this after receiving tts_chunk_played so downstream
listeners can observe frontend playback completion in FIFO order.

### 类字段

- `TYPE: ClassVar[str]` = `'tts.chunk_played_confirm'`

## TTSPlaybackFinished

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSPlaybackFinished(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'tts.playback_finished'`

## FullAudioFrameReady

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class FullAudioFrameReady(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'audio.full_frame_ready'`
- `audio_chunk: bytes` = `b''`
- `sample_rate: int` = `48000`
- `channels: int` = `2`
- `format: str` = `'pcm_s16le'`

## ErrorOccurred

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ErrorOccurred(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'error.occurred'`
- `error_type: str` = `''`
- `error_message: str` = `''`

## CaptionUpdated

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class CaptionUpdated(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'caption.updated'`
- `text: str` = `''`
- `is_final: bool` = `False`
- `reason: str` = `''`

## ToolCallOccurred

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ToolCallOccurred(Event)
```

LLM/Agent tool invocation notification.

### 类字段

- `TYPE: ClassVar[str]` = `'agent.tool_called'`
- `name: str` = `''`
- `args: Dict[str, Any]` = `field(default_factory=dict)`

## RetrievalUpdated

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class RetrievalUpdated(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'retrieval.updated'`
- `text: str` = `''`
- `is_final: bool` = `False`

## EmbeddingStatusUpdated

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class EmbeddingStatusUpdated(Event)
```

Embedding lifecycle update consumed by the LLM agent context manager.

### 类字段

- `TYPE: ClassVar[str]` = `'embedding.status_updated'`
- `status: str` = `''`
- `text: str | None` = `None`
- `vector_store_instance: Any` = `None`

## LLMAgentLoop

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class LLMAgentLoop(Event)
```

Request one agent-context accept loop iteration for the session.

### 类字段

- `TYPE: ClassVar[str]` = `'llm.agent_loop'`

## TextForEmbeddingReady

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TextForEmbeddingReady(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'embeddings.text_ready'`
- `text: str` = `''`

## LatencyMetricsUpdated

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class LatencyMetricsUpdated(Event)
```

Fine-grained backend latency metrics (milliseconds).

### 类字段

- `TYPE: ClassVar[str]` = `'metrics.latency_updated'`
- `network_latency_ms: int` = `0`
- `asr_latency_ms: int` = `0`
- `llm_first_token_ms: int` = `0`
- `llm_sentence_ms: int` = `0`
- `tts_first_chunk_ms: int` = `0`

## TurnTTSStartRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnTTSStartRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.tts_start_requested'`

## TurnTTSPauseRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnTTSPauseRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.tts_pause_requested'`

## TurnTTSResumeRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnTTSResumeRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.tts_resume_requested'`

## TurnTTSStopRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnTTSStopRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.tts_stop_requested'`
- `reason: str` = `''`

## TurnTTSFlushRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnTTSFlushRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.tts_flush_requested'`

## ConsumeLLMAgentGenerationRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ConsumeLLMAgentGenerationRequested(Event)
```

Request consumption of one LLM-agent output stream.

### 类字段

- `TYPE: ClassVar[str]` = `'llm_agent.consume_generation_requested'`
- `stream: AsyncIterator[AgentOutput]`

## TurnLLMAgentResumeRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnLLMAgentResumeRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.llm_agent_resume_requested'`

## TurnLLMAgentPauseRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnLLMAgentPauseRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.llm_agent_pause_requested'`

## TurnLLMAgentStopRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnLLMAgentStopRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.llm_agent_stop_requested'`
- `reason: str` = `''`

## TurnASRStartRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnASRStartRequested(Event)
```

### 类字段

- `TYPE: ClassVar[str]` = `'turn.asr_start_requested'`

## TurnASREndRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnASREndRequested(Event)
```

Indicates hard turn end. ASR model state is reset. Turn moves to next.

### 类字段

- `TYPE: ClassVar[str]` = `'turn.asr_end_requested'`

## TurnASRPauseRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnASRPauseRequested(Event)
```

Used when user indicates a wait, or pauses in the speech. Triggers recognition once. ASR model state is preserved; turn unchanged.

### 类字段

- `TYPE: ClassVar[str]` = `'turn.asr_pause_requested'`

## TurnTTSTextAppendRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnTTSTextAppendRequested(Event)
```

Request to append text into ongoing TTS stream (sim-trans).

### 类字段

- `TYPE: ClassVar[str]` = `'turn.tts_text_append_requested'`
- `text: str` = `''`

## SpeakerRecognized

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class SpeakerRecognized(Event)
```

Speaker-recognition result for frontend display.

### 类字段

- `TYPE: ClassVar[str]` = `'speaker.recognized'`
- `speaker_id: str | None` = `None`
- `reason: str` = `''`

## TTSModelSwitchRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TTSModelSwitchRequested(Event)
```

Request to switch the IndexTTS protocol version.

### 类字段

- `TYPE: ClassVar[str]` = `'tts.model_switch_requested'`
- `model_type: str` = `''`
- `config: Dict[str, Any]` = `field(default_factory=dict)`

## LLMModelSwitchRequested

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class LLMModelSwitchRequested(Event)
```

Request to switch LLM configuration (ChatOpenAI model/base_url).

### 类字段

- `TYPE: ClassVar[str]` = `'llm.model_switch_requested'`
- `model_name: str` = `''`
- `base_url: str` = `''`
- `api_key: str` = `''`
- `extra_body: dict | None` = `None`

## ClockSyncReceived

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class ClockSyncReceived(Event)
```

Clock-sync event for offset calculation.

### 类字段

- `TYPE: ClassVar[str]` = `'clock.sync_received'`
- `client_send_ts: float` = `0.0`
- `server_recv_ts: float` = `0.0`
- `client_recv_ts: float` = `0.0`

## SessionConfigReceived

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class SessionConfigReceived(Event)
```

Client sent per-session configuration (e.g., recording path).

### 类字段

- `TYPE: ClassVar[str]` = `'session.config_received'`
- `recording_path: str | None` = `None`

## TurnDetectorStopSpeaking

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnDetectorStopSpeaking(Event)
```

Turn detector determined ai should stop speaking.

### 类字段

- `TYPE: ClassVar[str]` = `'turn_detector.stop_speaking'`

## TurnDetectorStartGeneration

_定义于 [`xtalk.serving.events`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/events.py)。_

```python
@dataclass
class TurnDetectorStartGeneration(Event)
```

Turn detector determined ai should start generation.

### 类字段

- `TYPE: ClassVar[str]` = `'turn_detector.start_generation'`
