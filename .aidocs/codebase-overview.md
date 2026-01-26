*AI GENERATED*

## Project Summary

**X-Talk** is an open-source, full-duplex cascaded spoken dialogue system framework written in Python. It enables low-latency, interruptible, human-like speech interaction with the following key features:

- **Low-latency speech-to-speech pipeline** (sub-second end-to-end latency)
- **Natural user interruption support** during conversations
- **Paralinguistic information encoding** (environment noise, emotion)
- **Event-driven, loosely-coupled architecture** with async WebSocket support
- **Multiple ASR/TTS/LLM provider support** (local or cloud-based)
- **Researcher-friendly and production-ready** with pure Python backend

The system follows a cascaded pipeline architecture: User Speech → ASR → LLM Agent → TTS → Audio Output, with an event bus coordinating all components.

## How to Read Through the Codebase

1. **Start with the high-level API**: `src/xtalk/api.py` - This is the main entry point that shows how the system is configured and connected.

2. **Understand the pipeline abstraction**: `src/xtalk/pipelines/` - Read `interfaces.py` first, then `default.py` to see how models are organized.

3. **Learn the event-driven architecture**: `src/xtalk/serving/`
   - `event_bus.py` - Central pub/sub system
   - `events.py` - All event definitions
   - `service.py` - Per-session orchestrator

4. **Explore the managers**: `src/xtalk/serving/modules/` - Each manager handles a specific responsibility (ASR, TTS, LLM, VAD, etc.)

5. **Study speech components**: `src/xtalk/speech/` - ASR, TTS, VAD implementations with various provider options.

6. **Review the LLM agent layer**: `src/xtalk/llm_agent/` - Conversational AI with tool support.

7. **Check examples**: `examples/sample_app/` - Working server implementations demonstrating usage patterns.

## Individual File Descriptions

- `src/xtalk/api.py`: High-level Xtalk API class that handles configuration parsing, model initialization, session management, tool registration, and WebSocket connection handling. Main entry point for users.

- `src/xtalk/pipelines/interfaces.py`: Abstract base class defining the Pipeline interface with getter methods for all models (ASR, TTS, LLM agent, VAD, captioner, etc.) and context management.

- `src/xtalk/pipelines/default.py`: DefaultPipeline implementation containing core models (ASR, LLM agent, TTS) and optional models (captioner, VAD, enhancer, embeddings, rewriters). Handles model cloning for session isolation.

- `src/xtalk/pipelines/context.py`: Pipeline context/state sharing utilities for maintaining conversation state across components.

- `src/xtalk/serving/event_bus.py`: Central event routing and publishing system implementing pub/sub pattern with priority-based handler ordering, async task management, and error handling.

- `src/xtalk/serving/events.py`: Defines all event types (AudioFrameReceived, VADSpeechStart/End, ASRResultPartial/Final, TTSChunkReady, LLMAgentResponseFinish, etc.) used for inter-component communication.

- `src/xtalk/serving/interfaces.py`: Abstract interfaces for Manager and Service classes defining lifecycle methods and event handler patterns.

- `src/xtalk/serving/service.py`: Service class that orchestrates per-session lifecycle, manages all managers, handles event subscriptions, and coordinates input/output gateways.

- `src/xtalk/serving/service_manager.py`: WebSocket connection manager handling concurrent session management and connection lifecycle.

- `src/xtalk/serving/session_limiter.py`: Concurrency limiter for controlling maximum simultaneous sessions.

- `src/xtalk/serving/modules/input_gateway.py`: Translates WebSocket input messages (audio chunks, VAD signals) into internal events for the event bus.

- `src/xtalk/serving/modules/output_gateway.py`: Converts internal events into WebSocket output messages (audio, text, metadata) for the client.

- `src/xtalk/serving/modules/asr_manager.py`: Manages ASR pipeline including audio buffering, streaming/batch recognition modes, and transcription refinement.

- `src/xtalk/serving/modules/tts_manager.py`: Orchestrates TTS synthesis including text chunking, streaming audio generation, and voice/emotion/speed control.

- `src/xtalk/serving/modules/llm_agent_manager.py`: Coordinates LLM inference, tool execution (web search, voice change, etc.), and context management for conversational responses.

- `src/xtalk/serving/modules/vad_manager.py`: Voice activity detection manager processing audio frames to detect speech start/end events.

- `src/xtalk/serving/modules/enhancer_manager.py`: Audio enhancement/denoising manager for improving input audio quality.

- `src/xtalk/serving/modules/captioner_manager.py`: Audio captioning manager generating descriptions of audio content (environment sounds, speaker characteristics).

- `src/xtalk/serving/modules/speaker_manager.py`: Speaker recognition/tracking manager for identifying and maintaining speaker identity across turns.

- `src/xtalk/serving/modules/embeddings_manager.py`: Text embedding and retrieval manager for semantic search and document similarity.

- `src/xtalk/serving/modules/thought_manager.py`: LLM reasoning/thought display manager for exposing internal model reasoning.

- `src/xtalk/serving/modules/turn_taking_manager.py`: Dialogue turn management for natural conversation flow and interruption handling.

- `src/xtalk/serving/modules/latency_manager.py`: Performance metrics tracking for monitoring end-to-end latency.

- `src/xtalk/serving/modules/recording_manager.py`: Session audio recording manager for logging and debugging.

- `src/xtalk/speech/interfaces.py`: Abstract base classes for all speech components (ASR, TTS, VAD, Captioner, SpeechEnhancer, SpeakerEncoder, SpeechSpeedController, PuntRestorer).

- `src/xtalk/speech/asr/sherpa_onnx_asr.py`: Recommended local ASR implementation using SherpaOnnx with WebSocket streaming support.

- `src/xtalk/speech/asr/qwen3_asr_flash_realtime.py`: AliCloud Qwen3 ASR implementation with real-time streaming recognition.

- `src/xtalk/speech/asr/paraformer_local.py`: Local FunASR Paraformer model implementation for speech recognition.

- `src/xtalk/speech/asr/sensevoice_small_local.py`: Local SenseVoice small model ASR implementation.

- `src/xtalk/speech/asr/zipformer_local.py`: ONNX-based Zipformer ASR implementation.

- `src/xtalk/speech/asr/elevenlabs.py`: ElevenLabs cloud ASR API integration.

- `src/xtalk/speech/tts/index_tts.py`: IndexTTS neural vocoder-based text-to-speech implementation.

- `src/xtalk/speech/tts/index_tts2.py`: IndexTTS2 improved neural TTS implementation.

- `src/xtalk/speech/tts/cosyvoice.py`: AliCloud CosyVoice TTS integration with voice cloning support.

- `src/xtalk/speech/tts/gpt_sovits.py`: GPT-SoVITS voice cloning TTS implementation.

- `src/xtalk/speech/tts/f5_tts.py`: F5-TTS flow matching based text-to-speech.

- `src/xtalk/speech/tts/bark.py`: Bark PyTorch-based TTS with voice presets.

- `src/xtalk/speech/tts/elevenlabs.py`: ElevenLabs cloud TTS API integration.

- `src/xtalk/speech/tts/edge_tts.py`: Microsoft Edge TTS integration.

- `src/xtalk/speech/vad/silero_vad.py`: Silero VAD model implementation for voice activity detection.

- `src/xtalk/speech/captioner/qwen3_omni_captioner.py`: Qwen3 Omni model-based audio captioning implementation.

- `src/xtalk/speech/speech_enhancer/speech_enhancer.py`: FastEnhancer ONNX-based audio denoising implementation.

- `src/xtalk/speech/speaker_encoder/pyannote_embedding.py`: Pyannote.audio speaker embedding extraction for speaker recognition.

- `src/xtalk/speech/punt_restorer/ct_punt.py`: CT-Transformer punctuation restoration implementation.

- `src/xtalk/llm_agent/interfaces.py`: Abstract Agent base class defining generate/generate_stream methods with sync/async variants.

- `src/xtalk/llm_agent/default.py`: DefaultAgent LangChain-based implementation with system prompts, tool usage (voice/emotion/speed control, web search, retrieval), and multi-turn conversation support.

- `src/xtalk/llm_agent/tools/pipeline_control.py`: Built-in tools for voice switching, emotion control, and speech rate adjustment.

- `src/xtalk/llm_agent/tools/retrievers.py`: Tools for local document retrieval and web search integration.

- `src/xtalk/rewriter/interfaces.py`: Abstract Rewriter base class for text transformation pipelines.

- `src/xtalk/rewriter/simple.py`: Default caption and thought rewriter implementations using LLM-based text refinement.

- `src/xtalk/events.py`: Top-level event definitions and event class creation utilities.

- `src/xtalk/model_types.py`: Model interface exports and type definitions for external use.

- `src/xtalk/log_utils.py`: Logging configuration and utilities for consistent log formatting.

- `frontend/src/index.ts`: TypeScript entry point for the browser client library with WebSocket connection, audio capture, and VAD processing.

- `frontend/src/vad-processor.worklet.js`: Audio worklet for client-side voice activity detection processing.

- `examples/sample_app/configurable_server.py`: Generic FastAPI server demonstrating configuration-based deployment with WebSocket endpoint.

- `examples/sample_app/custom_service.py`: Example showing how to create custom services with additional managers and event handlers.

- `examples/sample_app/custom_model.py`: Example demonstrating custom model registration and usage.

- `examples/sample_app/mental_consultant_server.py`: Specialized demo server configured as a mental health counselor application.

- `scripts/gen_event_graph.py`: Utility script to generate event flow graphs for documentation and debugging.
