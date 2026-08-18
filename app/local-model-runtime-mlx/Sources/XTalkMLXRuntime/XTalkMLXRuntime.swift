import Foundation
import Darwin

@main
struct XTalkMLXRuntime {
    static func main() async {
        do {
            let options = try RuntimeOptions.parse(
                Array(CommandLine.arguments.dropFirst())
            )
            let runtime = try await ModelRuntime(
                service: options.service,
                modelRoot: options.modelRoot
            )
            let server = try await RuntimeServer.start(
                options: options,
                runtime: runtime
            )
            guard let port = server.port else {
                throw RuntimeServerError.invalidBoundAddress
            }
            try writeReadyMessage(port: port)
            try await server.waitUntilClosed()
        } catch {
            fputs("xtalk MLX runtime failed: \(error.localizedDescription)\n", stderr)
            exit(EXIT_FAILURE)
        }
    }

    private static func writeReadyMessage(port: Int) throws {
        let payload = try JSONSerialization.data(withJSONObject: [
            "status": "ready",
            "protocol_version": 1,
            "port": port,
        ])
        FileHandle.standardOutput.write(payload)
        FileHandle.standardOutput.write(Data([0x0A]))
    }
}
