import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import NIOWebSocket

private let refinerModelID = "agentic-asr-refiner"
private let refinerMaximumTokens = 512
private let xturnixModelID = "xturnix"

private struct RefinerChatRequest: Decodable, Sendable {
    let model: String
    let messages: [RefinerMessage]
    let maxTokens: Int?
    let temperature: Float?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens = "max_tokens"
        case temperature
    }
}

struct XTurnixChatTemplateOptions: Decodable, Sendable {
    let enableThinking: Bool?

    enum CodingKeys: String, CodingKey {
        case enableThinking = "enable_thinking"
    }
}

struct XTurnixTokenizeRequest: Decodable, Sendable {
    let model: String
    let prompt: String?
    let messages: [XTurnixMessage]?
    let addSpecialTokens: Bool?
    let addGenerationPrompt: Bool?
    let chatTemplateOptions: XTurnixChatTemplateOptions?

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case messages
        case addSpecialTokens = "add_special_tokens"
        case addGenerationPrompt = "add_generation_prompt"
        case chatTemplateOptions = "chat_template_kwargs"
    }
}

struct XTurnixChatRequest: Decodable, Sendable {
    let model: String
    let messages: [XTurnixMessage]
    let temperature: Float?
    let maxTokens: Int?
    let allowedTokenIDs: [Int]?
    let chatTemplateOptions: XTurnixChatTemplateOptions?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case temperature
        case maxTokens = "max_tokens"
        case allowedTokenIDs = "allowed_token_ids"
        case chatTemplateOptions = "chat_template_kwargs"
    }
}

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
            var response: [String: any Sendable] = [
                "status": "ok",
                "protocol_version": 1,
                "engine": service.engineName,
            ]
            if service.sampleRate > 0 {
                response["sample_rate"] = service.sampleRate
            }
            write(
                .json(response),
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
        if service == .agenticASRRefiner {
            if head.method == .GET,
               head.uri == "/v1/models" || head.uri == "/models"
            {
                write(
                    .json([
                        "object": "list",
                        "data": [[
                            "id": refinerModelID,
                            "object": "model",
                            "owned_by": "xtalk",
                        ]],
                    ]),
                    keepAlive: keepAlive,
                    context: context
                )
                return
            }
            if head.method == .POST,
               head.uri == "/v1/chat/completions"
                    || head.uri == "/chat/completions"
            {
                handleRefinerChat(
                    body: body,
                    keepAlive: keepAlive,
                    context: context
                )
                return
            }
            write(
                .json(status: .notFound, ["error": "endpoint not found"]),
                keepAlive: keepAlive,
                context: context
            )
            return
        }
        if service == .xturnixZHBase {
            if head.method == .GET,
               head.uri == "/v1/models" || head.uri == "/models"
            {
                write(
                    .json([
                        "object": "list",
                        "data": [[
                            "id": xturnixModelID,
                            "object": "model",
                            "owned_by": "xtalk",
                        ]],
                    ]),
                    keepAlive: keepAlive,
                    context: context
                )
                return
            }
            if head.method == .POST, head.uri == "/tokenize" {
                handleXTurnixTokenize(
                    body: body,
                    keepAlive: keepAlive,
                    context: context
                )
                return
            }
            if head.method == .POST,
               head.uri == "/v1/chat/completions"
                    || head.uri == "/chat/completions"
            {
                handleXTurnixChat(
                    body: body,
                    keepAlive: keepAlive,
                    context: context
                )
                return
            }
            write(
                .json(status: .notFound, ["error": "endpoint not found"]),
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

    private func handleRefinerChat(
        body: Data,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        let runtime = runtime
        let boundContext = NIOLoopBound(
            context,
            eventLoop: context.eventLoop
        )
        context.eventLoop.makeFutureWithTask {
            let request = try JSONDecoder().decode(
                RefinerChatRequest.self,
                from: body
            )
            guard request.model == refinerModelID else {
                throw RuntimeServerError.invalidRefinerModel(request.model)
            }
            guard !request.messages.isEmpty else {
                throw RuntimeServerError.emptyRefinerMessages
            }
            guard request.messages.allSatisfy({
                ["system", "user", "assistant"].contains($0.role)
            }) else {
                throw RuntimeServerError.invalidRefinerRole
            }
            let maxTokens = request.maxTokens ?? refinerMaximumTokens
            guard (1 ... refinerMaximumTokens).contains(maxTokens) else {
                throw RuntimeServerError.invalidRefinerTokenLimit
            }
            let temperature = request.temperature ?? 0
            guard temperature.isFinite, temperature >= 0 else {
                throw RuntimeServerError.invalidRefinerTemperature
            }
            let generation = try await runtime.refine(
                messages: request.messages,
                maxTokens: maxTokens
            )
            let totalTokens = generation.promptTokenCount
                + generation.generationTokenCount
            let assistantMessage: [String: any Sendable] = [
                "role": "assistant",
                "content": generation.text,
            ]
            let choice: [String: any Sendable] = [
                "index": 0,
                "message": assistantMessage,
                "finish_reason": generation.finishReason,
            ]
            let usage: [String: any Sendable] = [
                "prompt_tokens": generation.promptTokenCount,
                "completion_tokens": generation.generationTokenCount,
                "total_tokens": totalTokens,
            ]
            return HTTPResult.json([
                "id": "chatcmpl-\(UUID().uuidString)",
                "object": "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": refinerModelID,
                "choices": [choice],
                "usage": usage,
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

    private func handleXTurnixTokenize(
        body: Data,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        let runtime = runtime
        let boundContext = NIOLoopBound(
            context,
            eventLoop: context.eventLoop
        )
        context.eventLoop.makeFutureWithTask {
            let request = try JSONDecoder().decode(
                XTurnixTokenizeRequest.self,
                from: body
            )
            try validateXTurnixModel(request.model)
            guard (request.prompt == nil) != (request.messages == nil) else {
                throw RuntimeServerError.invalidXTurnixTokenizeInput
            }
            let tokens = try await runtime.tokenizeXTurnix(
                prompt: request.prompt,
                messages: request.messages,
                addSpecialTokens: request.addSpecialTokens ?? true,
                addGenerationPrompt: request.addGenerationPrompt ?? false,
                enableThinking: request.chatTemplateOptions?.enableThinking ?? false
            )
            return HTTPResult.json([
                "count": tokens.count,
                "tokens": tokens,
            ])
        }.whenComplete { result in
            self.completeXTurnixRequest(
                result,
                keepAlive: keepAlive,
                context: boundContext.value
            )
        }
    }

    private func handleXTurnixChat(
        body: Data,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        let runtime = runtime
        let boundContext = NIOLoopBound(
            context,
            eventLoop: context.eventLoop
        )
        context.eventLoop.makeFutureWithTask {
            let request = try JSONDecoder().decode(
                XTurnixChatRequest.self,
                from: body
            )
            try validateXTurnixChatRequest(request)
            let content = try await runtime.predictXTurnixAction(
                messages: request.messages,
                allowedTokenIDs: request.allowedTokenIDs ?? [],
                enableThinking: request.chatTemplateOptions?.enableThinking ?? false
            )
            let assistantMessage: [String: any Sendable] = [
                "role": "assistant",
                "content": content,
            ]
            let choice: [String: any Sendable] = [
                "index": 0,
                "message": assistantMessage,
                "finish_reason": "length",
            ]
            return HTTPResult.json([
                "id": "chatcmpl-\(UUID().uuidString)",
                "object": "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": xturnixModelID,
                "choices": [choice],
            ])
        }.whenComplete { result in
            self.completeXTurnixRequest(
                result,
                keepAlive: keepAlive,
                context: boundContext.value
            )
        }
    }

    private func completeXTurnixRequest(
        _ result: Result<HTTPResult, Error>,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        switch result {
        case .success(let response):
            write(response, keepAlive: keepAlive, context: context)
        case .failure(let error):
            write(
                .json(status: .badRequest, ["error": error.localizedDescription]),
                keepAlive: keepAlive,
                context: context
            )
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
    case invalidRefinerModel(String)
    case emptyRefinerMessages
    case invalidRefinerRole
    case invalidRefinerTokenLimit
    case invalidRefinerTemperature
    case invalidXTurnixModel(String)
    case invalidXTurnixTokenizeInput
    case emptyXTurnixMessages
    case invalidXTurnixRole
    case invalidXTurnixTemperature
    case invalidXTurnixTokenLimit
    case invalidXTurnixAllowedTokens

    var errorDescription: String? {
        switch self {
        case .missingPromptAudio:
            "prompt_audio is required"
        case .invalidSeed:
            "seed must be an integer"
        case .invalidBoundAddress:
            "MLX runtime did not bind a TCP port"
        case .invalidRefinerModel(let model):
            "Unknown Refiner model \(model); expected \(refinerModelID)"
        case .emptyRefinerMessages:
            "messages must not be empty"
        case .invalidRefinerRole:
            "messages contain an unsupported role"
        case .invalidRefinerTokenLimit:
            "max_tokens must be between 1 and \(refinerMaximumTokens)"
        case .invalidRefinerTemperature:
            "temperature must be a non-negative finite number"
        case .invalidXTurnixModel(let model):
            "Unknown XTurnix model \(model); expected \(xturnixModelID)"
        case .invalidXTurnixTokenizeInput:
            "exactly one of prompt or messages is required"
        case .emptyXTurnixMessages:
            "messages must not be empty"
        case .invalidXTurnixRole:
            "messages contain an unsupported role"
        case .invalidXTurnixTemperature:
            "temperature must be zero"
        case .invalidXTurnixTokenLimit:
            "max_tokens must be one"
        case .invalidXTurnixAllowedTokens:
            "allowed_token_ids must contain exactly two distinct non-negative IDs"
        }
    }
}

func validateXTurnixModel(_ model: String) throws {
    guard model == xturnixModelID else {
        throw RuntimeServerError.invalidXTurnixModel(model)
    }
}

func validateXTurnixChatRequest(_ request: XTurnixChatRequest) throws {
    try validateXTurnixModel(request.model)
    guard !request.messages.isEmpty else {
        throw RuntimeServerError.emptyXTurnixMessages
    }
    guard request.messages.allSatisfy({
        ["system", "user", "assistant"].contains($0.role)
    }) else {
        throw RuntimeServerError.invalidXTurnixRole
    }
    guard request.temperature == nil || request.temperature == 0 else {
        throw RuntimeServerError.invalidXTurnixTemperature
    }
    guard request.maxTokens == nil || request.maxTokens == 1 else {
        throw RuntimeServerError.invalidXTurnixTokenLimit
    }
    guard let allowedTokenIDs = request.allowedTokenIDs,
          allowedTokenIDs.count == 2,
          Set(allowedTokenIDs).count == 2,
          allowedTokenIDs.allSatisfy({ $0 >= 0 })
    else {
        throw RuntimeServerError.invalidXTurnixAllowedTokens
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
