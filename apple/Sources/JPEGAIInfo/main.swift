import Foundation
import CryptoKit
import CoreML
import JPEGAI

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
