import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioTTS

actor ModelRuntime {
    let service: ManagedModelService

    private let senseVoice: SenseVoiceModel?
    private let mossTTS: MossTTSNanoModel?

    init(service: ManagedModelService, modelRoot: URL) async throws {
        self.service = service
        switch service {
        case .senseVoice:
            senseVoice = try SenseVoiceModel.fromDirectory(modelRoot)
            mossTTS = nil
        case .mossTTSNano:
            senseVoice = nil
            mossTTS = try await MossTTSNanoModel.fromModelDirectory(modelRoot)
        }
    }

    func transcribe(_ packet: OfflineAudioPacket) throws -> String {
        guard let senseVoice else {
            throw ModelRuntimeError.wrongService
        }
        let samples: [Float]
        if packet.sampleRate == ManagedModelService.senseVoice.sampleRate {
            samples = packet.samples
        } else {
            samples = try resampleAudio(
                packet.samples,
                from: packet.sampleRate,
                to: ManagedModelService.senseVoice.sampleRate
            )
        }
        let output = senseVoice.generate(
            audio: MLXArray(samples),
            language: "auto",
            useITN: true
        )
        return output.text
    }

    func synthesize(
        text: String,
        promptAudio: Data,
        filename: String?
    ) async throws -> Data {
        guard let mossTTS else {
            throw ModelRuntimeError.wrongService
        }
        let normalizedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedText.isEmpty else {
            throw ModelRuntimeError.emptyText
        }
        guard promptAudio.count <= 32 * 1024 * 1024 else {
            throw ModelRuntimeError.promptAudioTooLarge
        }

        let fileExtension = sanitizedAudioExtension(filename)
        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("xtalk-mlx-\(UUID().uuidString)")
            .appendingPathExtension(fileExtension)
        try promptAudio.write(to: temporaryURL, options: .atomic)
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
        }
        let (_, referenceAudio) = try loadAudioArray(
            from: temporaryURL,
            sampleRate: ManagedModelService.mossTTSNano.sampleRate
        )
        let generationChunks: [String]
        if let tokenizer = mossTTS.tokenizer {
            generationChunks = try mossSplitTextIntoBestSentences(
                tokenizer: tokenizer,
                text: mossLightweightNormalizeText(normalizedText),
                maxTokens: 75
            )
        } else {
            generationChunks = [normalizedText]
        }
        var generationParameters = mossTTS.defaultGenerationParameters
        generationParameters.maxTokens = mossTTSFrameLimit(for: generationChunks)
        var samples: [Float] = []
        for _ in 0 ..< 2 {
            let output = try await mossTTS.generate(
                text: normalizedText,
                voice: nil,
                refAudio: referenceAudio,
                refText: nil,
                language: nil,
                generationParameters: generationParameters
            )
            let channelCount = output.ndim > 1 ? output.dim(output.ndim - 1) : 1
            let monoSamples = downmixInterleavedAudio(
                output.asArray(Float.self),
                channelCount: channelCount
            )
            samples = trimTrailingSilence(
                monoSamples,
                sampleRate: ManagedModelService.mossTTSNano.sampleRate
            )
            if !samples.isEmpty {
                break
            }
        }
        guard !samples.isEmpty else {
            throw ModelRuntimeError.emptyAudio
        }
        return encodePCM16Wave(
            samples: samples,
            sampleRate: ManagedModelService.mossTTSNano.sampleRate
        )
    }
}

/// Estimate a bounded MOSS generation budget for one already-split TTS chunk.
func mossTTSFrameLimit(for text: String) -> Int {
    let meaningfulCharacters = text.unicodeScalars.reduce(into: 0) { count, scalar in
        if !CharacterSet.whitespacesAndNewlines.contains(scalar) {
            count += 1
        }
    }
    // The official service allows 375 frames per <=75-token chunk. The MLX
    // model does not always emit EOS, so retain that ceiling while bounding
    // short requests by character count. Twelve frames per character plus a
    // 48-frame margin covers the slower observed Chinese samples without
    // forcing every missing-EOS request to generate the full 30 seconds.
    return min(375, max(64, meaningfulCharacters * 12 + 48))
}

/// Return the per-chunk frame limit required by the longest official text chunk.
func mossTTSFrameLimit(for chunks: [String]) -> Int {
    chunks.map(mossTTSFrameLimit(for:)).max() ?? 64
}

enum ModelRuntimeError: Error, LocalizedError {
    case wrongService
    case emptyText
    case emptyAudio
    case promptAudioTooLarge

    var errorDescription: String? {
        switch self {
        case .wrongService:
            "Requested operation is unavailable for this MLX service"
        case .emptyText:
            "text must not be empty"
        case .emptyAudio:
            "MOSS generation returned no audio after two attempts"
        case .promptAudioTooLarge:
            "prompt_audio exceeds 32 MiB"
        }
    }
}

private func sanitizedAudioExtension(_ filename: String?) -> String {
    let candidate = filename
        .map(URL.init(fileURLWithPath:))
        .map { $0.pathExtension.lowercased() }
    switch candidate {
    case "wav", "wave":
        return "wav"
    case "aif", "aiff":
        return "aiff"
    case "m4a":
        return "m4a"
    default:
        return "wav"
    }
}
