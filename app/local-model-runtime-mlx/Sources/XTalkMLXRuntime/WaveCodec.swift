import Foundation

enum WaveCodecError: Error, LocalizedError {
    case invalidFloatPayload
    case unsupportedSampleRate(UInt32)

    var errorDescription: String? {
        switch self {
        case .invalidFloatPayload:
            "Offline ASR payload is not aligned float32 audio"
        case .unsupportedSampleRate(let sampleRate):
            "Offline ASR sample rate is invalid: \(sampleRate)"
        }
    }
}

struct OfflineAudioPacket: Sendable {
    let sampleRate: Int
    let samples: [Float]

    static func decode(_ payload: Data) throws -> OfflineAudioPacket? {
        guard payload.count >= 8 else {
            return nil
        }
        let sampleRate = payload.readLittleEndianUInt32(at: 0)
        let byteCount = payload.readLittleEndianUInt32(at: 4)
        guard sampleRate > 0 else {
            throw WaveCodecError.unsupportedSampleRate(sampleRate)
        }
        guard byteCount <= 160 * 1024 * 1024 else {
            throw WaveCodecError.invalidFloatPayload
        }
        let expectedLength = 8 + Int(byteCount)
        guard payload.count >= expectedLength else {
            return nil
        }
        guard byteCount % 4 == 0 else {
            throw WaveCodecError.invalidFloatPayload
        }

        let audio = payload.subdata(in: 8 ..< expectedLength)
        var samples = Array(repeating: Float.zero, count: Int(byteCount) / 4)
        _ = samples.withUnsafeMutableBytes { destination in
            audio.copyBytes(to: destination)
        }
        if CFByteOrderGetCurrent() == CFByteOrderBigEndian.rawValue {
            for index in samples.indices {
                samples[index] = Float(
                    bitPattern: samples[index].bitPattern.byteSwapped
                )
            }
        }
        return OfflineAudioPacket(
            sampleRate: Int(sampleRate),
            samples: samples
        )
    }
}

func encodePCM16Wave(samples: [Float], sampleRate: Int) -> Data {
    var pcm = Data(capacity: samples.count * 2)
    for sample in samples {
        let scaled = Int(
            (max(-1, min(1, sample)) * Float(Int16.max)).rounded()
        )
        pcm.appendLittleEndian(Int16(clamping: scaled))
    }

    var wave = Data(capacity: 44 + pcm.count)
    wave.append(contentsOf: "RIFF".utf8)
    wave.appendLittleEndian(UInt32(36 + pcm.count))
    wave.append(contentsOf: "WAVE".utf8)
    wave.append(contentsOf: "fmt ".utf8)
    wave.appendLittleEndian(UInt32(16))
    wave.appendLittleEndian(UInt16(1))
    wave.appendLittleEndian(UInt16(1))
    wave.appendLittleEndian(UInt32(sampleRate))
    wave.appendLittleEndian(UInt32(sampleRate * 2))
    wave.appendLittleEndian(UInt16(2))
    wave.appendLittleEndian(UInt16(16))
    wave.append(contentsOf: "data".utf8)
    wave.appendLittleEndian(UInt32(pcm.count))
    wave.append(pcm)
    return wave
}

/// Downmix frame-interleaved audio channels to mono samples.
func downmixInterleavedAudio(
    _ samples: [Float],
    channelCount: Int
) -> [Float] {
    guard channelCount > 1 else {
        return samples
    }
    let frameCount = samples.count / channelCount
    return (0 ..< frameCount).map { frameIndex in
        let offset = frameIndex * channelCount
        let sum = samples[offset ..< offset + channelCount].reduce(0, +)
        return sum / Float(channelCount)
    }
}

/// Remove codec-generated trailing silence while retaining a short natural tail.
func trimTrailingSilence(
    _ samples: [Float],
    sampleRate: Int,
    amplitudeThreshold: Float = 0.009,
    tailMilliseconds: Int = 120
) -> [Float] {
    guard !samples.isEmpty, sampleRate > 0 else {
        return []
    }
    guard let lastAudibleIndex = samples.lastIndex(where: {
        abs($0) >= amplitudeThreshold
    }) else {
        return []
    }
    let tailSamples = sampleRate * max(0, tailMilliseconds) / 1_000
    let endIndex = min(samples.count, lastAudibleIndex + 1 + tailSamples)
    return Array(samples[..<endIndex])
}

extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }

    func readLittleEndianUInt32(at offset: Int) -> UInt32 {
        let range = offset ..< offset + MemoryLayout<UInt32>.size
        return self[range].withUnsafeBytes { bytes in
            UInt32(littleEndian: bytes.loadUnaligned(as: UInt32.self))
        }
    }
}
