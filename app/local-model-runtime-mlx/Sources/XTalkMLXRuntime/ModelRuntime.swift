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
        let output = try await mossTTS.generate(
            text: normalizedText,
            voice: nil,
            refAudio: referenceAudio,
            refText: nil,
            language: nil
        )
        return encodePCM16Wave(
            samples: output.asArray(Float.self),
            sampleRate: ManagedModelService.mossTTSNano.sampleRate
        )
    }
}

enum ModelRuntimeError: Error, LocalizedError {
    case wrongService
    case emptyText
    case promptAudioTooLarge

    var errorDescription: String? {
        switch self {
        case .wrongService:
            "Requested operation is unavailable for this MLX service"
        case .emptyText:
            "text must not be empty"
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
