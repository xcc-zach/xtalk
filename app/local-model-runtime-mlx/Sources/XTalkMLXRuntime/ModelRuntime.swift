import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioTTS
import MLXLLM
import MLXLMCommon
import Tokenizers

actor ModelRuntime {
    let service: ManagedModelService

    private let senseVoice: SenseVoiceModel?
    private let mossTTS: MossTTSNanoModel?
    private let refiner: ModelContainer?

    init(service: ManagedModelService, modelRoot: URL) async throws {
        self.service = service
        switch service {
        case .senseVoice:
            senseVoice = try SenseVoiceModel.fromDirectory(modelRoot)
            mossTTS = nil
            refiner = nil
        case .mossTTSNano:
            senseVoice = nil
            mossTTS = try await MossTTSNanoModel.fromModelDirectory(modelRoot)
            refiner = nil
        case .agenticASRRefiner:
            senseVoice = nil
            mossTTS = nil
            refiner = try await loadModelContainer(
                from: modelRoot,
                using: RefinerTokenizerLoader()
            )
        }
    }

    /// Refine one OpenAI-compatible chat request using greedy no-thinking decoding.
    func refine(
        messages: [RefinerMessage],
        maxTokens: Int
    ) async throws -> RefinerGenerationResult {
        guard let refiner else {
            throw ModelRuntimeError.wrongService
        }
        let inputMessages: [Message] = messages.map { message in
            ["role": message.role, "content": message.content]
        }
        let input = UserInput(
            messages: inputMessages,
            additionalContext: ["enable_thinking": false]
        )
        let prepared = try await refiner.prepare(input: input)
        let stream = try await refiner.generate(
            input: prepared,
            parameters: GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0
            )
        )
        var text = ""
        var promptTokenCount = 0
        var generationTokenCount = 0
        var finishReason = "length"
        for await generation in stream {
            switch generation {
            case .chunk(let chunk):
                text.append(chunk)
            case .info(let info):
                promptTokenCount = info.promptTokenCount
                generationTokenCount = info.generationTokenCount
                finishReason = switch info.stopReason {
                case .stop: "stop"
                case .length, .cancelled: "length"
                }
            case .toolCall:
                continue
            }
        }
        return RefinerGenerationResult(
            text: cleanRefinerOutput(text),
            promptTokenCount: promptTokenCount,
            generationTokenCount: generationTokenCount,
            finishReason: finishReason
        )
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
        let normalizedText = mossTTSSpeechText(text)
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
        }
        var samples: [Float] = []
        for (chunkIndex, chunk) in generationChunks.enumerated() {
            var generationParameters = mossTTS.defaultGenerationParameters
            generationParameters.maxTokens = mossTTSFrameLimit(for: chunk)
            generationParameters.temperature = 0.8
            generationParameters.topP = 0.95
            generationParameters.topK = 25
            generationParameters.repetitionPenalty = 1.2
            var candidates: [[Float]] = []
            let targetDuration = mossTTSTargetDurationSeconds(for: chunk)
            for candidateSeed in mossTTSCandidateSeeds(
                for: chunk,
                requestedSeed: seed
            ) {
                MLXRandom.seed(candidateSeed)
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
                let chunkSamples = trimTrailingSilence(
                    monoSamples,
                    sampleRate: ManagedModelService.mossTTSNano.sampleRate
                )
                guard !chunkSamples.isEmpty else {
                    continue
                }
                candidates.append(chunkSamples)
                if mossTTSDurationIsPreferred(
                    sampleCount: chunkSamples.count,
                    sampleRate: ManagedModelService.mossTTSNano.sampleRate,
                    targetDuration: targetDuration
                ) {
                    break
                }
            }
            let chunkSamples = candidates.min { lhs, rhs in
                mossTTSDurationScore(
                    sampleCount: lhs.count,
                    sampleRate: ManagedModelService.mossTTSNano.sampleRate,
                    targetDuration: targetDuration
                ) < mossTTSDurationScore(
                    sampleCount: rhs.count,
                    sampleRate: ManagedModelService.mossTTSNano.sampleRate,
                    targetDuration: targetDuration
                )
            } ?? []
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

/// One chat message accepted by the local Refiner service.
struct RefinerMessage: Codable, Sendable {
    let role: String
    let content: String
}

/// Generated Refiner text plus OpenAI-compatible token metadata.
struct RefinerGenerationResult: Sendable {
    let text: String
    let promptTokenCount: Int
    let generationTokenCount: Int
    let finishReason: String
}

/// Strip any chat terminators emitted before the configured stop token is observed.
func cleanRefinerOutput(_ text: String) -> String {
    text.replacingOccurrences(of: "<|im_end|>", with: "")
        .replacingOccurrences(of: "</s>", with: "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

private struct RefinerTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await Tokenizers.AutoTokenizer.from(
            modelFolder: directory
        )
        return RefinerTokenizerBridge(tokenizer)
    }
}

private struct RefinerTokenizerBridge: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(
            tokens: tokenIds,
            skipSpecialTokens: skipSpecialTokens
        )
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

struct MossSynthesisResult: Sendable {
    let wave: Data
    let textChunks: [String]
}

private let mossTTSChunkPunctuation = Set(".!?。！？")
private let mossTTSOpenClausePunctuation = Set("；;，,、：:")
private let mossTTSShortChunkCharacterLimit = 4
private let mossTTSShortChunkSeed: UInt64 = 21
private let mossTTSInterChunkPauseMilliseconds = 400
private let mossTTSCJKScalarRanges: [ClosedRange<UInt32>] = [
    0x3400 ... 0x4DBF,
    0x4E00 ... 0x9FFF,
    0x3040 ... 0x30FF,
    0xAC00 ... 0xD7AF,
]
private let mossTTSAbsolutePathExpression = try! NSRegularExpression(
    pattern: #"(^|[\s，。！？；：、“”])((?:/[A-Za-z0-9._+~-]+){2,})"#
)
private let mossTTSCamelBoundaryExpressions = [
    try! NSRegularExpression(pattern: #"([a-z0-9])([A-Z])"#),
    try! NSRegularExpression(pattern: #"([A-Z])([A-Z][a-z])"#),
]
private let mossTTSLatinWordExpression = try! NSRegularExpression(
    pattern: #"[A-Za-z0-9]+"#
)

/// Normalize mixed-language model input while preserving the displayed text.
func mossTTSSpeechText(_ text: String) -> String {
    var normalized = mossLightweightNormalizeText(
        text.precomposedStringWithCanonicalMapping
    )
    let matches = mossTTSAbsolutePathExpression.matches(
        in: normalized,
        range: NSRange(normalized.startIndex..., in: normalized)
    )
    for match in matches.reversed() {
        guard let pathRange = Range(match.range(at: 2), in: normalized) else {
            continue
        }
        let spokenPath = mossTTSPronouncePath(String(normalized[pathRange]))
        normalized.replaceSubrange(pathRange, with: spokenPath + ".")
    }
    normalized = normalized
        .replacingOccurrences(of: "，路径是 slash", with: "。路径如下。slash")
        .replacingOccurrences(of: ",路径是 slash", with: "。路径如下。slash")
        .replacingOccurrences(of: "路径是 slash", with: "路径如下。slash")
        .replacingOccurrences(of: "。，", with: "。")
        .replacingOccurrences(of: "。,", with: "。")
        .replacingOccurrences(of: ".，", with: ". ")
        .replacingOccurrences(of: ".,", with: ". ")
        .replacingOccurrences(of: "。。", with: "。")
        .replacingOccurrences(of: "..", with: ".")
    return mossTTSAddScriptBoundarySpacing(normalized)
}

/// Convert an absolute filesystem path into words the multilingual model can read.
func mossTTSPronouncePath(_ path: String) -> String {
    let components = path.split(separator: "/").map { component in
        mossTTSPronouncePathComponent(String(component))
    }
    return components.map { "slash \($0)" }.joined(separator: ", ")
}

/// Expand camel case and filename punctuation inside one spoken path component.
func mossTTSPronouncePathComponent(_ component: String) -> String {
    var spoken = component
    for expression in mossTTSCamelBoundaryExpressions {
        spoken = expression.stringByReplacingMatches(
            in: spoken,
            range: NSRange(spoken.startIndex..., in: spoken),
            withTemplate: "$1 $2"
        )
    }
    spoken = spoken.replacingOccurrences(of: "-", with: " ")
        .replacingOccurrences(of: "_", with: " ")
    spoken = spoken.replacingOccurrences(of: ".", with: " dot ")
    spoken = spoken.replacingOccurrences(
        of: #"\bxtalk\b"#,
        with: "X Talk",
        options: [.regularExpression, .caseInsensitive]
    )
    return spoken.split(whereSeparator: \Character.isWhitespace)
        .map { token in
            let value = String(token)
            if value.count >= 2,
               value.count <= 6,
               value.unicodeScalars.allSatisfy({
                   (0x41 ... 0x5A).contains($0.value)
               })
            {
                return value.map(String.init).joined(separator: " ")
            }
            return value
        }
        .joined(separator: " ")
}

/// Add one space where CJK and Latin scripts meet without changing punctuation.
func mossTTSAddScriptBoundarySpacing(_ text: String) -> String {
    let characters = Array(text)
    guard characters.count > 1 else {
        return text
    }
    var result = ""
    for index in characters.indices {
        let character = characters[index]
        result.append(character)
        guard index < characters.index(before: characters.endIndex),
              !character.isWhitespace
        else {
            continue
        }
        let next = characters[characters.index(after: index)]
        guard !next.isWhitespace else {
            continue
        }
        if (mossTTSIsCJK(character) && mossTTSIsLatinOrDigit(next))
            || (mossTTSIsLatinOrDigit(character) && mossTTSIsCJK(next))
        {
            result.append(" ")
        }
    }
    return result
}

/// Split one request at natural sentence and clause boundaries.
func mossTTSClauseChunks(for text: String) -> [String] {
    let normalized = mossTTSSpeechText(text)
    let punctuationChunks = mossSplitTextByPunctuation(
        normalized,
        punctuation: mossTTSChunkPunctuation
    )
    var chunks: [String] = []
    for punctuationChunk in punctuationChunks {
        let chunk = punctuationChunk.trimmingCharacters(in: .whitespacesAndNewlines)
        if !chunk.isEmpty {
            chunks.append(chunk)
        }
    }
    return chunks.isEmpty ? [normalized] : chunks
}

/// Split oversized clauses with the official token budget and close open punctuation.
func mossTTSGenerationChunks(
    tokenizer: MossTextTokenizing,
    text: String
) throws -> [String] {
    var chunks: [String] = []
    for sentence in mossTTSClauseChunks(for: text) {
        if let mixedChunks = mossTTSLeadingMixedClauseChunks(sentence) {
            chunks.append(contentsOf: mixedChunks)
            continue
        }
        let tokenCount = mossEncodeText(tokenizer, sentence).count
        let clauseBoundaryCount = sentence.filter {
            mossTTSOpenClausePunctuation.contains($0)
        }.count
        let shouldSplitLongCJK = mossContainsCJK(sentence)
            && tokenCount > 20
            && clauseBoundaryCount >= 2
        let maxTokens = shouldSplitLongCJK ? 12 : 75
        let sentenceChunks = try mossSplitTextIntoBestSentences(
            tokenizer: tokenizer,
            text: sentence,
            maxTokens: maxTokens
        )
        chunks.append(contentsOf: sentenceChunks.map(mossTTSClosingClauseBoundary))
    }
    return chunks
}

/// Isolate a short Chinese discourse opener before a Latin-heavy remainder.
func mossTTSLeadingMixedClauseChunks(_ sentence: String) -> [String]? {
    guard let boundary = sentence.firstIndex(where: {
        mossTTSOpenClausePunctuation.contains($0)
    }) else {
        return nil
    }
    let opener = String(sentence[..<boundary])
        .trimmingCharacters(in: .whitespacesAndNewlines)
    let remainderStart = sentence.index(after: boundary)
    let remainder = String(sentence[remainderStart...])
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !opener.isEmpty,
          !remainder.isEmpty,
          mossTTSSpokenCharacterCount(opener) <= mossTTSShortChunkCharacterLimit,
          remainder.unicodeScalars.contains(where: {
              (0x41 ... 0x5A).contains($0.value)
                  || (0x61 ... 0x7A).contains($0.value)
          })
    else {
        return nil
    }
    return [opener, remainder].map(mossTTSClosingClauseBoundary)
}

/// Remove split-boundary punctuation and close every inference chunk as a sentence.
func mossTTSClosingClauseBoundary(_ text: String) -> String {
    var normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
    while let first = normalized.first,
          mossTTSOpenClausePunctuation.contains(first)
    {
        normalized.removeFirst()
        normalized = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    guard !normalized.isEmpty else {
        return normalized
    }
    if let last = normalized.last, ".!?。！？".contains(last) {
        return normalized
    }
    if let last = normalized.last,
       mossTTSOpenClausePunctuation.contains(last)
    {
        normalized.removeLast()
    }
    normalized.append(mossContainsCJK(normalized) ? "。" : ".")
    return normalized
}

/// Select the stable short-phrase seed while retaining the general default seed.
func mossTTSSeed(for text: String, requestedSeed: UInt64) -> UInt64 {
    mossTTSSpokenCharacterCount(text) <= mossTTSShortChunkCharacterLimit
        ? mossTTSShortChunkSeed
        : requestedSeed
}

/// Return deterministic fallback seeds used when the model exits implausibly early or late.
func mossTTSCandidateSeeds(
    for text: String,
    requestedSeed: UInt64
) -> [UInt64] {
    let preferredSeed = mossTTSSeed(for: text, requestedSeed: requestedSeed)
    var seeds: [UInt64] = []
    for candidate in [preferredSeed, requestedSeed, 7, 1, 84] where !seeds.contains(candidate) {
        seeds.append(candidate)
    }
    return Array(seeds.prefix(4))
}

/// Estimate natural speech duration for candidate selection, not audio truncation.
func mossTTSTargetDurationSeconds(for text: String) -> Double {
    let cjkCount = text.reduce(into: 0) { count, character in
        if mossTTSIsCJK(character) {
            count += 1
        }
    }
    let fullRange = NSRange(text.startIndex..., in: text)
    let latinWordCount = mossTTSLatinWordExpression.numberOfMatches(
        in: text,
        range: fullRange
    )
    let latinCharacterCount = text.unicodeScalars.reduce(into: 0) { count, scalar in
        if (0x30 ... 0x39).contains(scalar.value)
            || (0x41 ... 0x5A).contains(scalar.value)
            || (0x61 ... 0x7A).contains(scalar.value)
        {
            count += 1
        }
    }
    let sentencePauseCount = text.filter { ".!?。！？".contains($0) }.count
    let clausePauseCount = text.filter { ",，、;；:：".contains($0) }.count
    return max(
        0.45,
        Double(cjkCount) * 0.22
            + Double(latinCharacterCount) * 0.075
            + Double(latinWordCount) * 0.12
            + Double(sentencePauseCount) * 0.12
            + Double(clausePauseCount) * 0.06
    )
}

/// Check whether generated duration is close enough to accept without another inference.
func mossTTSDurationIsPreferred(
    sampleCount: Int,
    sampleRate: Int,
    targetDuration: Double
) -> Bool {
    let duration = Double(sampleCount) / Double(max(1, sampleRate))
    return duration >= targetDuration * 0.65
        && duration <= targetDuration * 1.45 + 0.2
}

/// Score duration candidates symmetrically so early EOS and hallucinated tails both lose.
func mossTTSDurationScore(
    sampleCount: Int,
    sampleRate: Int,
    targetDuration: Double
) -> Double {
    let duration = max(0.001, Double(sampleCount) / Double(max(1, sampleRate)))
    return abs(log(duration / max(0.001, targetDuration)))
}

private func mossTTSIsCJK(_ character: Character) -> Bool {
    character.unicodeScalars.contains { scalar in
        mossTTSCJKScalarRanges.contains { $0.contains(scalar.value) }
    }
}

private func mossTTSIsLatinOrDigit(_ character: Character) -> Bool {
    character.unicodeScalars.allSatisfy { scalar in
        (0x30 ... 0x39).contains(scalar.value)
            || (0x41 ... 0x5A).contains(scalar.value)
            || (0x61 ... 0x7A).contains(scalar.value)
    }
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
            "MOSS generation returned no audio after candidate retries"
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
