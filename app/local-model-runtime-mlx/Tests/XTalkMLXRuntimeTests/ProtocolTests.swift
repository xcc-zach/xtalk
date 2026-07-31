import Foundation
import Testing
@testable import XTalkMLXRuntime

@Suite("MLX runtime protocol helpers")
struct ProtocolTests {
    @Test
    func parsesManagedRuntimeOptions() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        defer {
            try? FileManager.default.removeItem(at: root)
        }

        let options = try RuntimeOptions.parse([
            "--service", "sensevoice-small",
            "--model-root", root.path,
            "--host=127.0.0.1",
            "--port", "0",
        ])
        #expect(options.service == .senseVoice)
        #expect(options.port == 0)
    }

    @Test
    func decodesChunkedSherpaOfflinePacket() throws {
        let samples: [Float] = [0.25, -0.5, 1.0]
        var payload = Data()
        payload.appendLittleEndian(UInt32(16_000))
        payload.appendLittleEndian(UInt32(samples.count * 4))
        samples.withUnsafeBytes { payload.append(contentsOf: $0) }

        #expect(try OfflineAudioPacket.decode(payload.prefix(9)) == nil)
        let decoded = try #require(try OfflineAudioPacket.decode(payload))
        #expect(decoded.sampleRate == 16_000)
        #expect(decoded.samples == samples)
    }

    @Test
    func encodesFortyEightKilohertzPCM16Wave() {
        let wave = encodePCM16Wave(
            samples: [-1, 0, 1],
            sampleRate: 48_000
        )
        #expect(String(data: wave[0 ..< 4], encoding: .ascii) == "RIFF")
        #expect(wave.readLittleEndianUInt32(at: 24) == 48_000)
        #expect(wave.count == 50)
    }

    @Test
    func boundsMossGenerationFramesToSentenceLength() {
        #expect(mossTTSFrameLimit(for: "嘿，") == 72)
        #expect(mossTTSFrameLimit(for: "最近过得怎么样？") == 144)
        #expect(mossTTSFrameLimit(for: "嘿，你好呀。今天过得怎么样？") == 216)
        #expect(
            mossTTSFrameLimit(for: "那就好。有什么想聊的，或者需要我帮忙的吗？") == 300
        )
        #expect(mossTTSFrameLimit(for: String(repeating: "长", count: 100)) == 375)
        #expect(
            mossTTSFrameLimit(for: [
                "这是较短的一句。",
                "这是按官方文本预算切出的最长一块。",
            ]) == 252
        )
    }

    @Test
    func trimsCodecTrailingSilenceWithNaturalPadding() {
        let samples = [Float](repeating: 0, count: 10)
            + [0.02]
            + [Float](repeating: 0, count: 1_000)
        let trimmed = trimTrailingSilence(samples, sampleRate: 1_000)
        #expect(trimmed.count == 131)
        #expect(trimTrailingSilence([0, 0.001], sampleRate: 48_000).isEmpty)
    }

    @Test
    func downmixesInterleavedStereoBeforeMonoWaveEncoding() {
        let mono = downmixInterleavedAudio(
            [1, -1, 0.5, 0.25],
            channelCount: 2
        )
        #expect(mono == [0, 0.375])
        #expect(downmixInterleavedAudio([0.25], channelCount: 1) == [0.25])
    }

    @Test
    func parsesBinaryMultipartAudioWithoutTextConversion() throws {
        let boundary = "xtalk-test-boundary"
        let binary = Data([0x00, 0x80, 0xFF, 0x0A])
        var body = Data()
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(Data("Content-Disposition: form-data; name=\"text\"\r\n\r\n你好\r\n".utf8))
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(Data("Content-Disposition: form-data; name=\"prompt_audio\"; filename=\"voice.wav\"\r\n".utf8))
        body.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
        body.append(binary)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))

        let parts = try parseMultipartForm(
            contentType: "multipart/form-data; boundary=\(boundary)",
            body: body
        )
        #expect(parts.first(where: { $0.name == "text" })?.body == Data("你好".utf8))
        #expect(parts.first(where: { $0.name == "prompt_audio" })?.body == binary)
    }
}
