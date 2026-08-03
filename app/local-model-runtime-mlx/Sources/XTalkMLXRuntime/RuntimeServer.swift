import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import NIOWebSocket

struct HTTPResult: Sendable {
    let status: HTTPResponseStatus
    let contentType: String
    let body: Data

    static func json(
        status: HTTPResponseStatus = .ok,
        _ value: [String: any Sendable]
    ) -> HTTPResult {
        let body: Data
        do {
            body = try JSONSerialization.data(withJSONObject: value)
        } catch {
            body = Data("{\"error\":\"response encoding failed\"}".utf8)
        }
        return HTTPResult(
            status: status,
            contentType: "application/json; charset=utf-8",
            body: body
        )
    }
}

final class RuntimeHTTPHandler:
    ChannelInboundHandler,
    RemovableChannelHandler,
    @unchecked Sendable
{
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let runtime: ModelRuntime
    private let service: ManagedModelService
    private var requestHead: HTTPRequestHead?
    private var requestBody = Data()

    init(runtime: ModelRuntime, service: ManagedModelService) {
        self.runtime = runtime
        self.service = service
    }

    func channelRead(
        context: ChannelHandlerContext,
        data: NIOAny
    ) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            requestHead = head
            requestBody.removeAll(keepingCapacity: true)
        case .body(var buffer):
            guard requestBody.count + buffer.readableBytes <= 40 * 1024 * 1024 else {
                write(
                    .json(status: .payloadTooLarge, ["error": "request body is too large"]),
                    keepAlive: false,
                    context: context
                )
                return
            }
            if let bytes = buffer.readBytes(length: buffer.readableBytes) {
                requestBody.append(contentsOf: bytes)
            }
        case .end:
            guard let head = requestHead else {
                write(
                    .json(status: .badRequest, ["error": "request head is missing"]),
                    keepAlive: false,
                    context: context
                )
                return
            }
            handle(
                head: head,
                body: requestBody,
                context: context
            )
            requestHead = nil
            requestBody.removeAll(keepingCapacity: true)
        }
    }

    private func handle(
        head: HTTPRequestHead,
        body: Data,
        context: ChannelHandlerContext
    ) {
        let keepAlive = head.isKeepAlive
        if head.method == .GET, head.uri == "/health" {
            write(
                .json([
                    "status": "ok",
                    "protocol_version": 1,
                    "engine": service.engineName,
                    "sample_rate": service.sampleRate,
                ]),
                keepAlive: keepAlive,
                context: context
            )
            return
        }
        if head.method == .GET, head.uri == "/" {
            write(
                .json([
                    "service": "xtalk-mlx-model-runtime",
                    "engine": service.engineName,
                ]),
                keepAlive: keepAlive,
                context: context
            )
            return
        }
        guard service == .mossTTSNano,
              head.method == .POST,
              head.uri == "/api/generate"
        else {
            write(
                .json(status: .notFound, ["error": "endpoint not found"]),
                keepAlive: keepAlive,
                context: context
            )
            return
        }

        let contentType = head.headers.first(name: "content-type") ?? ""
        let runtime = runtime
        let boundContext = NIOLoopBound(
            context,
            eventLoop: context.eventLoop
        )
        context.eventLoop.makeFutureWithTask {
            let parts = try parseMultipartForm(
                contentType: contentType,
                body: body
            )
            let text = parts
                .first(where: { $0.name == "text" })
                .flatMap { String(data: $0.body, encoding: .utf8) }
                ?? ""
            guard let prompt = parts.first(where: { $0.name == "prompt_audio" }) else {
                throw RuntimeServerError.missingPromptAudio
            }
            let rawSeed = parts
                .first(where: { $0.name == "seed" })
                .flatMap { String(data: $0.body, encoding: .utf8) }
            let seed = try parseMossSeed(rawSeed)
            let synthesis = try await runtime.synthesize(
                text: text,
                promptAudio: prompt.body,
                filename: prompt.filename,
                seed: seed
            )
            return HTTPResult.json([
                "audio_base64": synthesis.wave.base64EncodedString(),
                "sample_rate": ManagedModelService.mossTTSNano.sampleRate,
                "run_status": "MOSS MLX generation complete: chunks=\(synthesis.textChunks.count)",
                "prompt_audio_path": prompt.filename ?? "prompt_audio",
                "warmup_status_text": "Ready.",
                "text_normalization_status_text": "Ready.",
                "text_chunks": synthesis.textChunks,
                "normalized_text": text,
                "normalization_method": "mlx-audio-swift",
                "text_normalization_language": "auto",
            ])
        }.whenComplete { result in
            let context = boundContext.value
            switch result {
            case .success(let response):
                self.write(
                    response,
                    keepAlive: keepAlive,
                    context: context
                )
            case .failure(let error):
                self.write(
                    .json(
                        status: .badRequest,
                        ["error": error.localizedDescription]
                    ),
                    keepAlive: keepAlive,
                    context: context
                )
            }
        }
    }

    private func write(
        _ result: HTTPResult,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: result.contentType)
        headers.add(name: "content-length", value: String(result.body.count))
        if keepAlive {
            headers.add(name: "connection", value: "keep-alive")
        } else {
            headers.add(name: "connection", value: "close")
        }
        let head = HTTPResponseHead(
            version: .http1_1,
            status: result.status,
            headers: headers
        )
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        var buffer = context.channel.allocator.buffer(
            capacity: result.body.count
        )
        buffer.writeBytes(result.body)
        context.write(wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        let boundContext = NIOLoopBound(
            context,
            eventLoop: context.eventLoop
        )
        context.writeAndFlush(wrapOutboundOut(.end(nil))).whenComplete { _ in
            if !keepAlive {
                boundContext.value.close(promise: nil)
            }
        }
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        write(
            .json(status: .internalServerError, ["error": error.localizedDescription]),
            keepAlive: false,
            context: context
        )
    }
}

final class OfflineASRWebSocketHandler:
    ChannelInboundHandler,
    @unchecked Sendable
{
    typealias InboundIn = WebSocketFrame
    typealias OutboundOut = WebSocketFrame

    private let runtime: ModelRuntime
    private var payload = Data()
    private var inferenceStarted = false

    init(runtime: ModelRuntime) {
        self.runtime = runtime
    }

    func channelRead(
        context: ChannelHandlerContext,
        data: NIOAny
    ) {
        let frame = unwrapInboundIn(data)
        switch frame.opcode {
        case .binary, .continuation:
            guard !inferenceStarted else {
                return
            }
            var frameData = frame.unmaskedData
            if let bytes = frameData.readBytes(length: frameData.readableBytes) {
                payload.append(contentsOf: bytes)
            }
            do {
                guard let packet = try OfflineAudioPacket.decode(payload) else {
                    return
                }
                inferenceStarted = true
                let runtime = runtime
                let boundContext = NIOLoopBound(
                    context,
                    eventLoop: context.eventLoop
                )
                context.eventLoop.makeFutureWithTask {
                    try await runtime.transcribe(packet)
                }.whenComplete { result in
                    let context = boundContext.value
                    switch result {
                    case .success(let text):
                        self.sendText(text, context: context)
                    case .failure(let error):
                        self.sendText("", context: context)
                        fputs("MLX ASR inference failed: \(error)\n", stderr)
                    }
                }
            } catch {
                sendText("", context: context)
            }
        case .text:
            let text = frame.unmaskedData.getString(
                at: frame.unmaskedData.readerIndex,
                length: frame.unmaskedData.readableBytes
            )
            if text == "Done" {
                context.close(promise: nil)
            }
        case .connectionClose:
            context.close(promise: nil)
        case .ping:
            let data = frame.unmaskedData
            context.writeAndFlush(
                wrapOutboundOut(
                    WebSocketFrame(fin: true, opcode: .pong, data: data)
                ),
                promise: nil
            )
        default:
            break
        }
    }

    private func sendText(
        _ text: String,
        context: ChannelHandlerContext
    ) {
        var buffer = context.channel.allocator.buffer(
            capacity: text.utf8.count
        )
        buffer.writeString(text)
        context.writeAndFlush(
            wrapOutboundOut(
                WebSocketFrame(fin: true, opcode: .text, data: buffer)
            ),
            promise: nil
        )
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        fputs("MLX WebSocket error: \(error)\n", stderr)
        context.close(promise: nil)
    }
}

enum RuntimeServerError: Error, LocalizedError {
    case missingPromptAudio
    case invalidSeed
    case invalidBoundAddress

    var errorDescription: String? {
        switch self {
        case .missingPromptAudio:
            "prompt_audio is required"
        case .invalidSeed:
            "seed must be an integer"
        case .invalidBoundAddress:
            "MLX runtime did not bind a TCP port"
        }
    }
}

/// Stable default seed selected for the MLX sampling implementation.
let mossDefaultSeed: UInt64 = 42

/// Parse the optional MOSS multipart seed into a deterministic MLX seed.
func parseMossSeed(_ rawValue: String?) throws -> UInt64 {
    guard let rawValue else {
        return mossDefaultSeed
    }
    let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    if value.isEmpty || value == "0" {
        return mossDefaultSeed
    }
    guard let seed = UInt64(value) else {
        throw RuntimeServerError.invalidSeed
    }
    return seed
}

struct RuntimeServer {
    let channel: Channel
    let group: MultiThreadedEventLoopGroup

    static func start(
        options: RuntimeOptions,
        runtime: ModelRuntime
    ) async throws -> RuntimeServer {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 2)
        do {
            let bootstrap = ServerBootstrap(group: group)
                .serverChannelOption(
                    ChannelOptions.socketOption(.so_reuseaddr),
                    value: 1
                )
                .childChannelInitializer { channel in
                    let httpHandler = RuntimeHTTPHandler(
                        runtime: runtime,
                        service: options.service
                    )
                    if options.service == .senseVoice {
                        let upgrader = NIOWebSocketServerUpgrader(
                            maxFrameSize: 16 * 1024 * 1024,
                            automaticErrorHandling: true,
                            shouldUpgrade: { channel, _ in
                                channel.eventLoop.makeSucceededFuture(
                                    HTTPHeaders()
                                )
                            },
                            upgradePipelineHandler: { channel, _ in
                                channel.pipeline.addHandler(
                                    OfflineASRWebSocketHandler(runtime: runtime)
                                )
                            }
                        )
                        return channel.pipeline.configureHTTPServerPipeline(
                            withServerUpgrade: (
                                upgraders: [upgrader],
                                completionHandler: { context in
                                    context.pipeline.removeHandler(
                                        httpHandler,
                                        promise: nil
                                    )
                                }
                            )
                        ).flatMap {
                            channel.pipeline.addHandler(httpHandler)
                        }
                    }
                    return channel.pipeline.configureHTTPServerPipeline()
                        .flatMap {
                            channel.pipeline.addHandler(httpHandler)
                        }
                }
                .childChannelOption(
                    ChannelOptions.socketOption(.tcp_nodelay),
                    value: 1
                )

            let channel = try await bootstrap.bind(
                host: options.host,
                port: options.port
            ).get()
            return RuntimeServer(channel: channel, group: group)
        } catch {
            try await group.shutdownGracefully()
            throw error
        }
    }

    var port: Int? {
        channel.localAddress?.port
    }

    func waitUntilClosed() async throws {
        try await channel.closeFuture.get()
        try await group.shutdownGracefully()
    }
}
