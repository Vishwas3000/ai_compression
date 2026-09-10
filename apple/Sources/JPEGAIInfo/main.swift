import Foundation
import CryptoKit
import CoreML
import JPEGAI

private let cliVersion = "0.1.0"
private let cliUsage = """
usage: jpeg-ai encode INPUT.png OUTPUT.bits [--preset 12|25|50|75|100] [--visualizations DIR]
       jpeg-ai decode INPUT.bits OUTPUT.png
       jpeg-ai --version

The default encode preset is 75. Set JPEG_AI_RESOURCES to override the bundled
Tables and Models directory.
"""

private enum CLIError: LocalizedError {
    case invalidArguments(String)
    case resourcesNotFound

    var errorDescription: String? {
        switch self {
        case let .invalidArguments(message): message
        case .resourcesNotFound:
            "Codec models were not found. Reinstall with Homebrew or set JPEG_AI_RESOURCES."
        }
    }
}

private struct CLIResources {
    let tables: URL
    let models: URL

    static func locate() throws -> Self {
        let manager = FileManager.default
        let executable = URL(fileURLWithPath: CommandLine.arguments[0])
            .resolvingSymlinksInPath()
        let prefix = executable.deletingLastPathComponent().deletingLastPathComponent()
        let current = URL(fileURLWithPath: manager.currentDirectoryPath, isDirectory: true)
        var candidates = [
            prefix.appendingPathComponent("share/jpeg-ai", isDirectory: true),
            current.appendingPathComponent("Models", isDirectory: true),
            current.appendingPathComponent("apple/Models", isDirectory: true),
        ]
        if let override = ProcessInfo.processInfo.environment["JPEG_AI_RESOURCES"] {
            candidates.insert(URL(fileURLWithPath: override, isDirectory: true), at: 0)
        }

        for root in candidates {
            let packaged = Self(
                tables: root.appendingPathComponent("Tables", isDirectory: true),
                models: root.appendingPathComponent("Models", isDirectory: true)
            )
            if packaged.exists(using: manager) { return packaged }

            let source = Self(
                tables: root,
                models: root.appendingPathComponent("apple-coreml-simple", isDirectory: true)
            )
            if source.exists(using: manager) { return source }
        }
        throw CLIError.resourcesNotFound
    }

    private func exists(using manager: FileManager) -> Bool {
        manager.fileExists(
            atPath: tables.appendingPathComponent("unique_z_distributions.csv").path
        ) && manager.fileExists(atPath: models.appendingPathComponent("tools_0").path)
    }
}

private func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
}

private func runFriendlyCLI(_ arguments: [String]) async throws -> Bool {
    guard let command = arguments.first else {
        print(cliUsage)
        return true
    }
    if command == "--help" || command == "-h" {
        print(cliUsage)
        return true
    }
    if command == "--version" {
        print("jpeg-ai \(cliVersion)")
        return true
    }
    if command == "encode" {
        guard arguments.count >= 3 else { throw CLIError.invalidArguments(cliUsage) }
        let presets = [12: (0, 0), 25: (1, 0), 50: (2, -184), 75: (2, 0), 100: (3, 0)]
        var preset = 75
        var visualizations: URL?
        var index = 3
        while index < arguments.count {
            guard index + 1 < arguments.count else {
                throw CLIError.invalidArguments("Missing value for \(arguments[index]).")
            }
            switch arguments[index] {
            case "--preset", "-p":
                guard let value = Int(arguments[index + 1]), presets[value] != nil else {
                    throw CLIError.invalidArguments("Preset must be 12, 25, 50, 75, or 100.")
                }
                preset = value
            case "--visualizations":
                visualizations = URL(fileURLWithPath: arguments[index + 1], isDirectory: true)
            default:
                throw CLIError.invalidArguments("Unknown option: \(arguments[index])")
            }
            index += 2
        }

        let resources = try CLIResources.locate()
        let input = URL(fileURLWithPath: arguments[1])
        let output = URL(fileURLWithPath: arguments[2])
        let image = try JPEGAIDecodedImage.readPNG(from: input)
        let models = JPEGAICoreMLModelSet(directory: resources.models)
        let (model, beta) = presets[preset]!
        let started = ContinuousClock.now
        let encoded = try await image.encodeJPEGAI(
            tablesDirectory: resources.tables, models: models, model: model, beta: beta,
            includeDiagnostics: visualizations != nil
        )
        let elapsed = seconds(started.duration(to: .now))
        try encoded.write(to: output)
        if let visualizations {
            try encoded.diagnostics?.write(to: visualizations)
        }
        print(String(
            format: "Encoded %dx%d to %d bytes in %.3f s (preset %d)\n%@",
            image.width, image.height, encoded.data.count, elapsed, preset, output.path
        ))
        return true
    }
    if command == "decode" {
        guard arguments.count == 3 else { throw CLIError.invalidArguments(cliUsage) }
        let resources = try CLIResources.locate()
        let input = URL(fileURLWithPath: arguments[1])
        let output = URL(fileURLWithPath: arguments[2])
        let stream = try JPEGAIBitstream(contentsOf: input)
        let models = JPEGAICoreMLModelSet(directory: resources.models)
        let started = ContinuousClock.now
        let image = try await stream.decodeImage(
            tablesDirectory: resources.tables, models: models
        )
        let elapsed = seconds(started.duration(to: .now))
        try image.writePNG(to: output)
        print(String(
            format: "Decoded %dx%d in %.3f s\n%@",
            image.width, image.height, elapsed, output.path
        ))
        return true
    }
    return false
}

do {
    if try await runFriendlyCLI(Array(CommandLine.arguments.dropFirst())) { exit(0) }
} catch let error as CLIError {
    FileHandle.standardError.write(Data("jpeg-ai: \(error.localizedDescription)\n".utf8))
    exit(2)
} catch {
    FileHandle.standardError.write(Data("jpeg-ai: \(error.localizedDescription)\n".utf8))
    exit(1)
}

if (8 ... 9).contains(CommandLine.arguments.count), CommandLine.arguments[1] == "--encode" {
    do {
        guard let model = Int(CommandLine.arguments[6]),
              let beta = Int(CommandLine.arguments[7]) else {
            throw JPEGAIEncodeError.invalidModel
        }
        let input = URL(fileURLWithPath: CommandLine.arguments[2])
        let tables = URL(fileURLWithPath: CommandLine.arguments[3])
        let models = JPEGAICoreMLModelSet(
            directory: URL(fileURLWithPath: CommandLine.arguments[4])
        )
        let output = URL(fileURLWithPath: CommandLine.arguments[5])
        let image = try JPEGAIDecodedImage.readPNG(from: input)
        let started = ContinuousClock.now
        let encoded = try await image.encodeJPEGAI(
            tablesDirectory: tables, models: models, model: model, beta: beta,
            includeDiagnostics: CommandLine.arguments.count == 9
        )
        let elapsed = started.duration(to: .now)
        try encoded.write(to: output)
        if CommandLine.arguments.count == 9 {
            let diagnostics = URL(fileURLWithPath: CommandLine.arguments[8], isDirectory: true)
            try encoded.diagnostics?.write(to: diagnostics)
            print("Wrote visualizations to \(diagnostics.path)")
        }
        let seconds = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        print(String(
            format: "Encoded %dx%d to %d bytes in %.3f s (model %d, beta %d)",
            image.width, image.height, encoded.data.count, seconds, model, beta
        ))
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("jpegai-info: \(error.localizedDescription)\n".utf8))
        exit(1)
    }
}

guard (2 ... 5).contains(CommandLine.arguments.count) else {
    FileHandle.standardError.write(Data("""
    usage: jpegai-info INPUT.bits [TABLES_DIR [MODELS_DIR [OUTPUT.png]]]
           jpegai-info --encode INPUT.png TABLES_DIR MODELS_DIR OUTPUT.bits MODEL BETA [VISUALIZATIONS_DIR]
    """.utf8))
    exit(2)
}

do {
    let stream = try JPEGAIBitstream(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    let header = try stream.pictureHeader
    print("\(header.displayWidth)x\(header.displayHeight), \(header.bitDepth)-bit, model \(header.model), beta \(header.betaDisplacementY)")
    for substream in stream.substreams {
        print(String(format: "0x%04X  %8d bytes", substream.marker.rawValue, substream.payload.count))
    }
    if CommandLine.arguments.count >= 3 {
        let tables = try JPEGAIZTables(
            directory: URL(fileURLWithPath: CommandLine.arguments[2]), model: header.model
        )
        let latents = try stream.decodeHyperLatents(using: tables)
        func md5<T>(_ values: [T]) -> String {
            let data = values.withUnsafeBufferPointer {
                Data(bytes: $0.baseAddress!, count: $0.count * MemoryLayout<T>.stride)
            }
            return Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        print("z Y  [1, \(header.channelsY), \(latents.height), \(latents.width)] md5 \(md5(latents.ySymbols))")
        print("z UV [1, \(header.channelsUV), \(latents.height), \(latents.width)] md5 \(md5(latents.uvSymbols))")
        if CommandLine.arguments.count >= 4 {
            let models = JPEGAICoreMLModelSet(
                directory: URL(fileURLWithPath: CommandLine.arguments[3])
            )
            let residualTables = try JPEGAIResidualTables(
                directory: URL(fileURLWithPath: CommandLine.arguments[2]), model: header.model
            )
            let residuals = try await stream.decodeResidualLatents(
                hyper: latents, tables: residualTables, models: models
            )
            func report(_ name: String, _ component: JPEGAIResidualComponent) {
                let masked = zip(component.quantized, component.mask).compactMap {
                    $0.1 == 1 ? $0.0 : nil
                }
                print("scale \(name) md5 \(md5(component.scaleLog))")
                print("mask  \(name) md5 \(md5(component.mask.map(Int32.init)))")
                print("resi  \(name) md5 \(md5(masked))")
            }
            report("Y ", residuals.y)
            report("UV", residuals.uv)
            let image = try await stream.reconstructImage(
                hyper: latents, residuals: residuals, tables: residualTables, models: models
            )
            print("Core ML synthesis -> \(image.width)x\(image.height) RGB")
            if CommandLine.arguments.count == 5 {
                let output = URL(fileURLWithPath: CommandLine.arguments[4])
                try image.writePNG(to: output)
                print("Wrote \(output.path)")
            }
        }
    }
} catch {
    FileHandle.standardError.write(Data("jpegai-info: \(error)\n".utf8))
    exit(1)
}
