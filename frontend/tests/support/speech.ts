/**
 * Test speech generation shared with `tests/support/speech.py`.
 *
 * Sentence `id` (0..63) is encoded as a sine tone at `400 + 60 * id` Hz with
 * amplitude 0.5; silence is all zeros.
 */

const BASE_FREQUENCY_HZ = 400;
const FREQUENCY_STEP_HZ = 60;
const AMPLITUDE = 0.5;

const FIXED_TEXTS: Record<number, string> = {
    0: "你好",
    1: "请讲一个很长的故事",
    2: "帮我查一下天气",
    3: "现在报错",
    4: "你刚才说了什么",
    5: "慢慢说",
    6: "让合成失败",
    7: "识别失败",
};

/** Return the fixed text mapped to one sentence id. */
export function textFor(sentenceId: number): string {
    if (sentenceId < 0 || sentenceId > 63) {
        throw new Error("sentence id must be in 0..63");
    }
    return FIXED_TEXTS[sentenceId] ?? `第${sentenceId}句话`;
}

/** Return the tone frequency encoding one sentence id. */
export function frequencyFor(sentenceId: number): number {
    return BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * sentenceId;
}

/** Generate PCM16 mono test speech followed by optional silence. */
export function encode(
    sentenceId: number,
    options: { voicedMs: number; trailingSilenceMs?: number; sampleRate: number },
): ArrayBuffer {
    const frequency = frequencyFor(sentenceId);
    const voicedSamples = Math.round((options.sampleRate * options.voicedMs) / 1000);
    const silentSamples = Math.round((options.sampleRate * (options.trailingSilenceMs ?? 0)) / 1000);
    const pcm = new Int16Array(voicedSamples + silentSamples);
    for (let i = 0; i < voicedSamples; i++) {
        pcm[i] = Math.round(Math.sin((2 * Math.PI * frequency * i) / options.sampleRate) * AMPLITUDE * 32767);
    }
    return pcm.buffer;
}

/** Generate PCM16 silence. */
export function silence(ms: number, sampleRate: number): ArrayBuffer {
    return new Int16Array(Math.round((sampleRate * ms) / 1000)).buffer;
}
