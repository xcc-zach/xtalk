import { createWebSocket } from "./websocket";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "./bases/audio-session";
import { createInputAudioSession, createOutputAudioSession } from "./audio-session";
import { Conversation } from "./conversation";
import { ActionHandler } from "./action-handler";

export { createSession };

interface SessionConfig {
    inputConfig?: Partial<InputAudioSessionConfig>,
    outputConfig?: Partial<OutputAudioSessionConfig>
}
function createSession(
    websocketURL: string | URL,
    {
        inputConfig = {},
        outputConfig = {},
    }: SessionConfig = {},
) {
    const resolvedInputConfig: InputAudioSessionConfig = {
        sampleRate: 16000,
        ...inputConfig,
    };
    const resolvedOutputConfig: OutputAudioSessionConfig = {
        sampleRate: 48000,
        ...outputConfig,
    };
    const conversation = new Conversation();
    const actionHandler = new ActionHandler();
    let websocket: ReturnType<typeof createWebSocket>;
    let inputAudioSession: ReturnType<typeof createInputAudioSession>;
    let outputAudioSession: ReturnType<typeof createOutputAudioSession>;

    let inputAudioChunkCallback: ((pcmChunkInt16: ArrayBuffer, sampleRate: number) => void) = (_chunk, _sr) => { };
    let outputAudioChunkCallback: ((pcmChunkInt16: ArrayBuffer, sampleRate: number) => void) = (_chunk, _sr) => { };

    function initialize() {
        websocket = createWebSocket(websocketURL);
        inputAudioSession = createInputAudioSession(resolvedInputConfig);
        outputAudioSession = createOutputAudioSession(resolvedOutputConfig);

        // Subscribe actions and audio chunks
        websocket.addEventListener("message", async (event: { data: string | ArrayBuffer }) => {
            if (typeof event.data === "string") {
                const message: { action: string, data: any } = JSON.parse(event.data);
                try {
                    await actionHandler.handleAction(message.action, message.data, websocket, conversation, outputAudioSession);
                } catch (error) {
                    //TODO: Handle unknown action error
                }
            } else if (event.data instanceof ArrayBuffer) {
                await outputAudioSession.pushAudioChunk(event.data);
            }
        });

        // Bind audio input handling
        inputAudioSession.onFrame(async (audioChunk) => {
            inputAudioChunkCallback(audioChunk, resolvedInputConfig.sampleRate);
            websocket.sendAudioChunk(audioChunk);
        });
        inputAudioSession.onSpeechStart(async () => {
            await actionHandler.handleAction("client_speech_start", null, websocket, conversation, outputAudioSession);
        });
        inputAudioSession.onSpeechEnd(async () => {
            await actionHandler.handleAction("client_speech_end", null, websocket, conversation, outputAudioSession);
        });

        // Bind audio output handling
        outputAudioSession.onChunkStarted(async (audioChunk) => {
            outputAudioChunkCallback(audioChunk, resolvedOutputConfig.sampleRate);
            await actionHandler.handleAction("client_audio_chunk_started", null, websocket, conversation, outputAudioSession);
        });
        outputAudioSession.onChunkPlayed(async (_audioChunk) => {
            await actionHandler.handleAction("client_audio_chunk_played", null, websocket, conversation, outputAudioSession);
        });
        outputAudioSession.onAllChunksPlayed(async () => {
            await actionHandler.handleAction("client_audio_playback_finished", null, websocket, conversation, outputAudioSession);
        });
    }


    // Create API for external use
    const session = {
        open: async () => {
            initialize();
            await inputAudioSession.open();
            await outputAudioSession.open();
        },
        close: async () => {
            await inputAudioSession.close();
            await outputAudioSession.close();
            websocket.close();
        },
        onStateChange: (callback: (state: Conversation["state"]) => void) => {
            conversation.onStateChange(callback);
        },
        get state() {
            return conversation.state;
        },
        onInputAudioChunk: (callback: (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void) => {
            inputAudioChunkCallback = callback;
        },
        onOutputAudioChunk: (callback: (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void) => {
            outputAudioChunkCallback = callback;
        },
        onFullAudioChunk: (callback: (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void) => {
            conversation.onFullAudioChunk(callback);
        },
        get muted() {
            return inputAudioSession.muted;
        },
        set muted(value: boolean) {
            inputAudioSession.muted = value;
        },
        async changeVoice(voiceName: string) {
            await actionHandler.handleAction("client_change_voice", { voiceName }, websocket, conversation, outputAudioSession)
        },
        async uploadFile(file: Blob, endpoint: string | URL = "./api/upload") {
            await actionHandler.handleAction("client_upload_file", { file, endpoint }, websocket, conversation, outputAudioSession);
        }
    }

    return session;
}
