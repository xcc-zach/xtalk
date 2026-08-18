<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.serving.module_types

## OutputGateway

_定义于 [`xtalk.serving.modules.output_gateway`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/output_gateway.py)。_

```python
class OutputGateway(EventListenerMixin)
```

Forward backend events to the frontend WebSocket.

### 参数

- `event_bus` (`EventBus`)
  Event bus used to subscribe to session events.
- `session_id` (`str`)
  Session identifier sent back to the frontend.
- `websocket` (`WebSocket`)
  Live WebSocket connection used for outbound messages.
- `config` (`dict[str, Any] | None, optional`)
  Service configuration relevant to output behavior.
- `models` (`Models | None, optional`)
  Session models used to place ASR previews before the speaker-history
  barrier when the focus-only gate is active.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.output_gateway`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/output_gateway.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, websocket: WebSocket, config: dict[str, Any] | None = None, models: Models | None = None)
```

#### send_signal

_定义于 [`xtalk.serving.modules.output_gateway`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/output_gateway.py)。_

```python
async def send_signal(self, message: dict) -> None
```

Send a JSON payload to the frontend.

##### 参数

- `message` (`dict`)
  JSON-serializable payload to send over the WebSocket.

#### send_session_attached

_定义于 [`xtalk.serving.modules.output_gateway`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/output_gateway.py)。_

```python
async def send_session_attached(self) -> None
```

Send the attached session identifier to the frontend.

## ASRManager

_定义于 [`xtalk.serving.modules.asr_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/asr_manager.py)。_

```python
class ASRManager(Manager)
```

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.asr_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/asr_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None)
```

#### shutdown

_定义于 [`xtalk.serving.modules.asr_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/asr_manager.py)。_

```python
async def shutdown(self)
```

## DirectAudioManager

_定义于 [`xtalk.serving.modules.direct_audio_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/direct_audio_manager.py)。_

```python
class DirectAudioManager(Manager)
```

Forward ``direct_audio`` tool calls to the outbound audio stream.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.direct_audio_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/direct_audio_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None) -> None
```

Initialize the direct-audio manager.

##### 参数

- `event_bus` (`EventBus`)
  Session-scoped event bus.
- `session_id` (`str`)
  Active session identifier.
- `config` (`dict[str, Any] | None, optional`)
  Session configuration shared with managers.

#### shutdown

_定义于 [`xtalk.serving.modules.direct_audio_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/direct_audio_manager.py)。_

```python
async def shutdown(self) -> None
```

Shut down the manager.

## EmbeddingsManager

_定义于 [`xtalk.serving.modules.embeddings_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/embeddings_manager.py)。_

```python
class EmbeddingsManager(Manager)
```

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.embeddings_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/embeddings_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None)
```

#### shutdown

_定义于 [`xtalk.serving.modules.embeddings_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/embeddings_manager.py)。_

```python
async def shutdown(self) -> None
```

Remove per-session embedding directory on shutdown.

## EnhancerManager

_定义于 [`xtalk.serving.modules.enhancer_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/enhancer_manager.py)。_

```python
class EnhancerManager(Manager)
```

Backend speech enhancement manager.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.enhancer_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/enhancer_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: Optional[dict[str, Any]] = None) -> None
```

#### shutdown

_定义于 [`xtalk.serving.modules.enhancer_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/enhancer_manager.py)。_

```python
async def shutdown(self) -> None
```

Reset enhancer state on shutdown.

## LatencyManager

_定义于 [`xtalk.serving.modules.latency_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/latency_manager.py)。_

```python
class LatencyManager(EventListenerMixin)
```

Per-session latency tracker that listens to VAD/ASR/LLM/TTS events.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.latency_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/latency_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None) -> None
```

#### update_clock_offset

_定义于 [`xtalk.serving.modules.latency_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/latency_manager.py)。_

```python
def update_clock_offset(self, client_send_ts: float, server_recv_ts: float, client_recv_ts: float) -> None
```

Update the clock offset estimate using an NTP-style ping/pong exchange.

client_send_ts = T1, server_recv_ts = T2, client_recv_ts = T4.
Offset = T2 - (T1 + T4)/2 and we track a rolling median for stability.

#### shutdown

_定义于 [`xtalk.serving.modules.latency_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/latency_manager.py)。_

```python
async def shutdown(self)
```

## LLMAgentContextManager

_定义于 [`xtalk.serving.modules.llm_agent_context_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_context_manager.py)。_

```python
class LLMAgentContextManager(Manager)
```

Forward session context events into the configured LLM agent.

### 参数

- `event_bus` (`EventBus`)
  Shared event bus for the current session.
- `session_id` (`str`)
  Current session identifier.
- `models` (`Models`)
  Session model container that owns the LLM agent.
- `config` (`dict[str, Any] | None, optional`)
  Unused manager config kept for interface consistency.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.llm_agent_context_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_context_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None) -> None
```

#### shutdown

_定义于 [`xtalk.serving.modules.llm_agent_context_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_context_manager.py)。_

```python
async def shutdown(self) -> None
```

Release manager resources.

## LLMAgentConsumptionManager

_定义于 [`xtalk.serving.modules.llm_agent_generation_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_generation_manager.py)。_

```python
class LLMAgentConsumptionManager(Manager)
```

Consume one or more agent streams and forward their output downstream.

### 说明

Multiple agent streams may be active concurrently. Each stream owns an
independent response so asynchronous tool reports cannot reset or append to
a response that is already playing.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.llm_agent_generation_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_generation_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None) -> None
```

#### shutdown

_定义于 [`xtalk.serving.modules.llm_agent_generation_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/llm_agent_generation_manager.py)。_

```python
async def shutdown(self) -> None
```

Cancel all active streams during service shutdown.

## SpeakerManager

_定义于 [`xtalk.serving.modules.speaker_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/speaker_manager.py)。_

```python
class SpeakerManager(Manager)
```

Session-scoped speaker identification manager.

Responsibilities:
- Collect enhanced audio frames per turn and extract embeddings.
- Compare against previously registered speakers.
- Recognize an existing speaker or register a new one.
- Emit ``SpeakerRecognized`` events for downstream consumers.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.speaker_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/speaker_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None)
```

Initialize the speaker manager.

Args:
    event_bus: shared event bus
    session_id: unique session identifier
    models: model container providing a speaker encoder
    config: optional parameters
        - similarity_threshold: cosine threshold (default 0.4)
        - min_audio_length_sec: minimum audio length (default 0.5s)
        - embedding_update_alpha: EMA rate for embeddings (default 0.05)

#### shutdown

_定义于 [`xtalk.serving.modules.speaker_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/speaker_manager.py)。_

```python
async def shutdown(self) -> None
```

Stop buffering audio and persist debug summaries.

## TTSPlaybackManager

_定义于 [`xtalk.serving.modules.tts_playback_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_playback_manager.py)。_

```python
class TTSPlaybackManager(Manager)
```

Project confirmed TTS playback progress back onto response text.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.tts_playback_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_playback_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models | None = None, config: dict[str, Any] | None = None) -> None
```

#### shutdown

_定义于 [`xtalk.serving.modules.tts_playback_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_playback_manager.py)。_

```python
async def shutdown(self) -> None
```

## TTSManager

_定义于 [`xtalk.serving.modules.tts_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_manager.py)。_

```python
class TTSManager(Manager)
```

Event-driven TTS manager handling streaming synthesis and control.

### 类字段

- `SENTENCE_DELIMITERS` = `{'。', '，', '！', '!', '？', '?', '.', ',', '：', ':'}`
- `TTS_CHUNK_MS` = `100`
- `MAX_OUTSTANDING_MS` = `300`

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.tts_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None)
```

Initialize TTS manager.

Args:
    event_bus: shared event bus
    session_id: unique session identifier
    models: model container providing TTS models/controllers

#### reset_tts

_定义于 [`xtalk.serving.modules.tts_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_manager.py)。_

```python
async def reset_tts(self) -> None
```

Reset all TTS state and cancel consumers.

#### shutdown

_定义于 [`xtalk.serving.modules.tts_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_manager.py)。_

```python
async def shutdown(self) -> None
```

Shut down TTS manager and reset state.

## TTSResponseCoordinator

_定义于 [`xtalk.serving.modules.tts_response_coordinator`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_response_coordinator.py)。_

```python
class TTSResponseCoordinator(Manager)
```

Gate all response delivery through one session-scoped state machine.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.tts_response_coordinator`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_response_coordinator.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None) -> None
```

Initialize response-delivery state for one session.

##### 参数

- `event_bus` (`EventBus`)
  Session event bus.
- `session_id` (`str`)
  Session identifier.
- `config` (`dict[str, Any] | None, optional`)
  Shared service configuration.

#### shutdown

_定义于 [`xtalk.serving.modules.tts_response_coordinator`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/tts_response_coordinator.py)。_

```python
async def shutdown(self) -> None
```

Clear coordinator state during session shutdown.

## TurnTakingManager

_定义于 [`xtalk.serving.modules.turn_taking_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/turn_taking_manager.py)。_

```python
class TurnTakingManager(Manager)
```

Coordinate VAD boundaries with ASR and response interruption.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.turn_taking_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/turn_taking_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None)
```

#### shutdown

_定义于 [`xtalk.serving.modules.turn_taking_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/turn_taking_manager.py)。_

```python
async def shutdown(self)
```

## VADManager

_定义于 [`xtalk.serving.modules.vad_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/vad_manager.py)。_

```python
class VADManager(Manager)
```

Backend VAD manager.

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.vad_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/vad_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: Optional[dict[str, Any]] = None) -> None
```

#### shutdown

_定义于 [`xtalk.serving.modules.vad_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/vad_manager.py)。_

```python
async def shutdown(self) -> None
```

Reset VAD state and release any remote session resources.

## MultiSpeakerTurnContextManager

_定义于 [`xtalk.serving.modules.multi_speaker_turn_context_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/multi_speaker_turn_context_manager.py)。_

```python
class MultiSpeakerTurnContextManager(Manager)
```

Schedule generic diarization and join its turn results with ASR.

### 类字段

- `BYTES_PER_SAMPLE` = `2`
- `SAMPLE_RATE` = `16000`

### 方法

#### __init__

_定义于 [`xtalk.serving.modules.multi_speaker_turn_context_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/multi_speaker_turn_context_manager.py)。_

```python
def __init__(self, event_bus: EventBus, session_id: str, models: Models, config: dict[str, Any] | None = None) -> None
```

#### shutdown

_定义于 [`xtalk.serving.modules.multi_speaker_turn_context_manager`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/serving/modules/multi_speaker_turn_context_manager.py)。_

```python
async def shutdown(self) -> None
```

Cancel session tasks and release the diarization model clone.
