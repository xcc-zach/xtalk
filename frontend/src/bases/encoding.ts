export { BaseEncoding };

abstract class BaseEncoding {
    /** Encodes PCM audio after conversion to the wire sample rate. */
    abstract encodePCM(data: ArrayBuffer, sampleRate: number, targetRate: number): string;
    /** Encodes PCM bytes for JSON transport. */
    abstract encodeBase64(data: ArrayBuffer): string;
    abstract decodeBase64(base64: string): ArrayBuffer;
}
