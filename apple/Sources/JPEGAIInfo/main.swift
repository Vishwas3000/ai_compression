import Foundation
import CryptoKit
import CoreML
import CoreGraphics
import ImageIO
import JPEGAI
import UniformTypeIdentifiers

guard (2 ... 5).contains(CommandLine.arguments.count) else {
    FileHandle.standardError.write(Data("usage: jpegai-info INPUT.bits [TABLES_DIR [MODELS_DIR [OUTPUT.png]]]\n".utf8))
    exit(2)
}

func writePNG(_ image: JPEGAIDecodedImage, to url: URL) throws {
    guard let provider = CGDataProvider(data: Data(image.rgb) as CFData),
          let cgImage = CGImage(
            width: image.width, height: image.height,
            bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: image.width * 3,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
          ),
          let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
          ) else {
        throw CocoaError(.fileWriteUnknown)
    }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw CocoaError(.fileWriteUnknown)
    }
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
                try writePNG(image, to: output)
                print("Wrote \(output.path)")
            }
        }
    }
} catch {
    FileHandle.standardError.write(Data("jpegai-info: \(error)\n".utf8))
    exit(1)
}
