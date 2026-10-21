/** A Realtime network event with extension-specific payload fields. */
export interface RealtimeEvent {
    type: string;
    event_id?: string;
    [key: string]: unknown;
}

/** Xtalk's revision semantics on the standard transcription event prefix. */
export interface TranscriptionDelta extends RealtimeEvent {
    type: "conversation.item.input_audio_transcription.delta";
    item_id: string;
    content_index: number;
    delta: string;
    xtalk: {mode: "append" | "replace"; revision: number};
}

/** Content identity shared by generated audio, text, and transcript events. */
export interface OutputContentEvent extends RealtimeEvent {
    response_id: string;
    item_id: string;
    output_index: number;
    content_index: number;
}

/** Original network audio block; acknowledge once after this block is played. */
export interface AudioDelta extends OutputContentEvent {
    type: "response.output_audio.delta";
    delta: string;
}

/** Complete playback-confirmed text snapshot, independent of generated text. */
export interface PlaybackText extends RealtimeEvent {
    type: "response.output_audio.playback.updated" | "response.output_audio.playback.done";
    response_id: string;
    text: string;
}

/** Explicitly distinguish real-time user text from context-only injection. */
export interface ConversationItemCreate extends RealtimeEvent {
    type: "conversation.item.create";
    item: {
        id?: string;
        type: "message";
        role: "user" | "assistant" | "system";
        content: Array<{type: "input_text" | "text"; text: string}>;
    };
    xtalk?: {generate_response: boolean};
}

/** Negotiated encoding for server output audio; input encoding is independent. */
export type OutputAudioEncoding = "base64" | "binary_pcm";

/** Versioned binary audio header preceding raw PCM bytes. */
export interface BinaryAudioHeader extends RealtimeEvent {
    version: 1;
    type: "xtalk.output_audio.delta" | "xtalk.audio.mix.delta";
    event_id: string;
}
