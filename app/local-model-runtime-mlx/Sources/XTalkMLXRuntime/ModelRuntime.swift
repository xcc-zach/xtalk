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
        filename: String?,
        seed: UInt64
    ) async throws -> MossSynthesisResult {
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
            generationChunks = try mossTTSGenerationChunks(
                tokenizer: tokenizer,
                text: normalizedText
            )
        } else {
            generationChunks = mossTTSClauseChunks(for: normalizedText)
                .map(mossTTSClosingClauseBoundary)
        }
        var samples: [Float] = []
        for (chunkIndex, chunk) in generationChunks.enumerated() {
            var generationParameters = mossTTS.defaultGenerationParameters
            generationParameters.maxTokens = mossTTSFrameLimit(for: chunk)
            var chunkSamples: [Float] = []
            let chunkSeed = mossTTSSeed(for: chunk, requestedSeed: seed)
            for attempt in 0 ..< 2 {
                MLXRandom.seed(chunkSeed &+ UInt64(attempt))
                let output = try await mossTTS.generate(
                    text: chunk,
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
                chunkSamples = trimTrailingSilence(
                    monoSamples,
                    sampleRate: ManagedModelService.mossTTSNano.sampleRate
                )
                if !chunkSamples.isEmpty {
                    break
                }
            }
            guard !chunkSamples.isEmpty else {
                throw ModelRuntimeError.emptyAudio
            }
            samples.append(contentsOf: chunkSamples)
            if chunkIndex < generationChunks.count - 1 {
                samples.append(contentsOf: repeatElement(
                    Float.zero,
                    count: mossTTSInterChunkPauseSamples(
                        sampleRate: ManagedModelService.mossTTSNano.sampleRate
                    )
                ))
            }
        }
        guard !samples.isEmpty else {
            throw ModelRuntimeError.emptyAudio
        }
        return MossSynthesisResult(
            wave: encodePCM16Wave(
                samples: samples,
                sampleRate: ManagedModelService.mossTTSNano.sampleRate
            ),
            textChunks: generationChunks
        )
    }
}

struct MossSynthesisResult: Sendable {
    let wave: Data
    let textChunks: [String]
}

private let mossTTSChunkPunctuation = Set(".!?。！？；;，,、：:")
private let mossTTSOpenClausePunctuation = Set("；;，,、：:")
private let mossTTSShortChunkCharacterLimit = 4
private let mossTTSShortChunkSeed: UInt64 = 21
private let mossTTSInterChunkPauseMilliseconds = 400

/// Split one request at natural sentence and clause boundaries.
func mossTTSClauseChunks(for text: String) -> [String] {
    let normalized = mossLightweightNormalizeText(text)
    let punctuationChunks = mossSplitTextByPunctuation(
        normalized,
        punctuation: mossTTSChunkPunctuation
    )
    var chunks: [String] = []
    var pending = ""
    for punctuationChunk in punctuationChunks {
        pending = mossJoinSentenceParts(pending, punctuationChunk)
        if let last = punctuationChunk.last,
           mossTTSOpenClausePunctuation.contains(last),
           mossTTSMeaningfulCharacterCount(pending)
               <= mossTTSShortChunkCharacterLimit
        {
            continue
        }
        chunks.append(pending)
        pending = ""
    }
    if !pending.isEmpty {
        chunks.append(pending)
    }
    return chunks.isEmpty ? [normalized] : chunks
}

/// Split oversized clauses with the official token budget and close open punctuation.
func mossTTSGenerationChunks(
    tokenizer: MossTextTokenizing,
    text: String
) throws -> [String] {
    var chunks: [String] = []
    for clause in mossTTSClauseChunks(for: text) {
        let tokenChunks = try mossSplitTextIntoBestSentences(
            tokenizer: tokenizer,
            text: clause,
            maxTokens: 75
        )
        chunks.append(contentsOf: tokenChunks.map(mossTTSClosingClauseBoundary))
    }
    return chunks
}

/// Replace an open clause delimiter with a closed sentence delimiter for inference.
func mossTTSClosingClauseBoundary(_ text: String) -> String {
    var normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let last = normalized.last,
          mossTTSOpenClausePunctuation.contains(last)
    else {
        return normalized
    }
    normalized.removeLast()
    normalized.append(mossContainsCJK(normalized) ? "。" : ".")
    return normalized
}

/// Select the stable short-phrase seed while retaining the general default seed.
func mossTTSSeed(for text: String, requestedSeed: UInt64) -> UInt64 {
    mossTTSSpokenCharacterCount(text) <= mossTTSShortChunkCharacterLimit
        ? mossTTSShortChunkSeed
        : requestedSeed
}

/// Return the official inter-chunk pause in samples for one output rate.
func mossTTSInterChunkPauseSamples(sampleRate: Int) -> Int {
    sampleRate * mossTTSInterChunkPauseMilliseconds / 1_000
}

/// Count non-whitespace Unicode scalars used by generation heuristics.
func mossTTSMeaningfulCharacterCount(_ text: String) -> Int {
    text.unicodeScalars.reduce(into: 0) { count, scalar in
        if !CharacterSet.whitespacesAndNewlines.contains(scalar) {
            count += 1
        }
    }
}

/// Count spoken Unicode scalars while excluding whitespace and punctuation.
func mossTTSSpokenCharacterCount(_ text: String) -> Int {
    text.unicodeScalars.reduce(into: 0) { count, scalar in
        if !CharacterSet.whitespacesAndNewlines.contains(scalar),
           !CharacterSet.punctuationCharacters.contains(scalar)
        {
            count += 1
        }
    }
}

/// Estimate a bounded MOSS generation budget for one already-split TTS chunk.
func mossTTSFrameLimit(for text: String) -> Int {
    let meaningfulCharacters = mossTTSMeaningfulCharacterCount(text)
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
