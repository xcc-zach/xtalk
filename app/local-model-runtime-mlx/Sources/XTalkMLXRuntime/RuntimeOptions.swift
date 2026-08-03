import Foundation

enum ManagedModelService: String, Sendable {
    case senseVoice = "sensevoice-small"
    case mossTTSNano = "moss-tts-nano"

    var engineName: String {
        "\(rawValue)-mlx"
    }

    var sampleRate: Int {
        switch self {
        case .senseVoice:
            16_000
        case .mossTTSNano:
            48_000
        }
    }
}

struct RuntimeOptions: Sendable {
    let service: ManagedModelService
    let modelRoot: URL
    let host: String
    let port: Int

    static func parse(_ arguments: [String]) throws -> RuntimeOptions {
        var values: [String: String] = [:]
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            guard argument.hasPrefix("--") else {
                throw RuntimeOptionError.invalidArgument(argument)
            }
            let option = String(argument.dropFirst(2))
            if let separator = option.firstIndex(of: "=") {
                let name = String(option[..<separator])
                let value = String(option[option.index(after: separator)...])
                values[name] = value
                index += 1
                continue
            }
            guard index + 1 < arguments.count else {
                throw RuntimeOptionError.missingValue(option)
            }
            values[option] = arguments[index + 1]
            index += 2
        }

        guard let rawService = values["service"],
              let service = ManagedModelService(rawValue: rawService)
        else {
            throw RuntimeOptionError.invalidService(values["service"])
        }
        guard let rawRoot = values["model-root"], !rawRoot.isEmpty else {
            throw RuntimeOptionError.missingValue("model-root")
        }
        let modelRoot = URL(fileURLWithPath: rawRoot, isDirectory: true)
            .standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: modelRoot.path,
            isDirectory: &isDirectory
        ), isDirectory.boolValue
        else {
            throw RuntimeOptionError.invalidModelRoot(modelRoot.path)
        }

        let host = values["host"] ?? "127.0.0.1"
        guard host == "127.0.0.1" else {
            throw RuntimeOptionError.nonLoopbackHost(host)
        }
        let rawPort = values["port"] ?? "0"
        guard let port = Int(rawPort), (0 ... 65_535).contains(port) else {
            throw RuntimeOptionError.invalidPort(rawPort)
        }
        return RuntimeOptions(
            service: service,
            modelRoot: modelRoot,
            host: host,
            port: port
        )
    }
}

enum RuntimeOptionError: Error, LocalizedError {
    case invalidArgument(String)
    case missingValue(String)
    case invalidService(String?)
    case invalidModelRoot(String)
    case nonLoopbackHost(String)
    case invalidPort(String)

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let argument):
            "Unexpected positional argument: \(argument)"
        case .missingValue(let option):
            "Missing value for --\(option)"
        case .invalidService(let service):
            "Unsupported managed MLX service: \(service ?? "<missing>")"
        case .invalidModelRoot(let path):
            "MLX model root is not a directory: \(path)"
        case .nonLoopbackHost(let host):
            "MLX runtime only accepts the loopback host, got \(host)"
        case .invalidPort(let port):
            "Invalid MLX runtime port: \(port)"
        }
    }
}
