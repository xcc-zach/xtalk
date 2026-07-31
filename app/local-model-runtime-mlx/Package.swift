// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "XTalkMLXRuntime",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "xtalk-mlx-model-runtime",
            targets: ["XTalkMLXRuntime"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/Blaizzy/mlx-audio-swift.git",
            exact: "0.1.3"
        ),
        .package(
            url: "https://github.com/apple/swift-nio.git",
            exact: "2.99.0"
        ),
    ],
    targets: [
        .executableTarget(
            name: "XTalkMLXRuntime",
            dependencies: [
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOWebSocket", package: "swift-nio"),
            ]
        ),
        .testTarget(
            name: "XTalkMLXRuntimeTests",
            dependencies: ["XTalkMLXRuntime"]
        ),
    ]
)
