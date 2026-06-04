# Overall Architecture

## Frontend / Backend Separation

The frontend API lives under `frontend`, and the backend lives under `src/xtalk`.

The frontend is responsible for sending and receiving audio and control signals, as well as lightweight speech processing such as VAD and speech enhancement. The backend carries the core logic of X-Talk.

## Backend Architecture

Backend components are organized into two layers: models and `Manager`s.

Models are adapters for external models. For example, if you want to use IndexTTS in X-Talk, you need to implement an adapter that follows the `TTS` interface definition, then start IndexTTS externally and connect it through configuration.

`Manager`s are model schedulers. For example, `ASRManager` controls when the ASR model starts, pauses, resumes, and ends recognition.

Communication between `Manager`s follows the observer pattern in order to keep the capabilities of different models decoupled and move cross-model interactions into event-based communication.

Model interfaces are defined in `src/xtalk/llm_agent/interfaces.py` and `src/xtalk/speech/interfaces.py`.

The former defines the LLM adapter interface: the LLM is the intelligence core of X-Talk and is responsible for producing output by combining information from various speech models.

The latter contains adapter interfaces for speech modules such as ASR, TTS, and VAD, as well as some experimental interfaces such as `TurnDetector` and `SpeakerEncoder`.
