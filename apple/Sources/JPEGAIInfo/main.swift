import Foundation
import CryptoKit
import CoreML
import JPEGAI

guard (2 ... 4).contains(CommandLine.arguments.count) else {
    FileHandle.standardError.write(Data("usage: jpegai-info INPUT.bits [TABLES_DIR [MODELS_DIR]]\n".utf8))
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
        func md5(_ bytes: [UInt8]) -> String {
            Insecure.MD5.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
        }
        print("z Y  [1, \(header.channelsY), \(latents.height), \(latents.width)] md5 \(md5(latents.ySymbols))")
        print("z UV [1, \(header.channelsUV), \(latents.height), \(latents.width)] md5 \(md5(latents.uvSymbols))")
        if CommandLine.arguments.count == 4 {
            let models = JPEGAICoreMLModelSet(
                directory: URL(fileURLWithPath: CommandLine.arguments[3])
            )
            let yInput = try JPEGAICoreMLModelSet.floatArray(
                latents.y, channels: header.channelsY, height: latents.height, width: latents.width
            )
            let uvInput = try JPEGAICoreMLModelSet.floatArray(
                latents.uv, channels: header.channelsUV, height: latents.height, width: latents.width
            )
            let y = try await models.predict(
                tool: header.model, component: "model_y",
                path: "common_modules/hyper_decoder", inputs: [yInput]
            )
            let uv = try await models.predict(
                tool: header.model, component: "model_uv",
                path: "common_modules/hyper_decoder", inputs: [uvInput]
            )
            print("Core ML hyper decoder Y  -> \(y.shape)")
            print("Core ML hyper decoder UV -> \(uv.shape)")
        }
    }
} catch {
    FileHandle.standardError.write(Data("jpegai-info: \(error)\n".utf8))
    exit(1)
}
