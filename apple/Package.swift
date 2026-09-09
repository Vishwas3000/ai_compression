// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "JPEGAIApple",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "JPEGAI", targets: ["JPEGAI"]),
        .executable(name: "jpegai-info", targets: ["JPEGAIInfo"]),
    ],
    targets: [
        .target(name: "CJPEGAIEntropy", publicHeadersPath: "include"),
        .target(name: "JPEGAI", dependencies: ["CJPEGAIEntropy"]),
        .executableTarget(name: "JPEGAIInfo", dependencies: ["JPEGAI"]),
        .testTarget(name: "JPEGAITests", dependencies: ["JPEGAI"]),
    ]
)
