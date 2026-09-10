#if canImport(CoreML)
import CoreML
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct JPEGAIResidualTables: Sendable {
    let transitions: [UInt32]
    let bounds: [UInt8]
    let gainY: [Int32]
    let gainUV: [Int32]

    public init(directory: URL, model: Int) throws {
        guard jpegAIModelNames.indices.contains(model) else {
            throw JPEGAIEntropyError.invalidTables
        }
        let transitionRows = try readJPEGAICSV(
            directory.appendingPathComponent("residual_transitions.csv")
        )
        let boundRows = try readJPEGAICSV(
            directory.appendingPathComponent("residual_bounds.csv")
        )
        guard !transitionRows.isEmpty, transitionRows.allSatisfy({ $0.count == 256 }) else {
            throw JPEGAIEntropyError.invalidTables
        }
        transitions = try transitionRows.flatMap { row in
            try row.map {
                guard let value = UInt32(exactly: $0) else {
                    throw JPEGAIEntropyError.invalidTables
                }
                return value
            }
        }
        bounds = try boundRows.flatMap { row in
            try row.map {
                guard let value = UInt8(exactly: $0) else {
                    throw JPEGAIEntropyError.invalidTables
                }
                return value
            }
        }
        guard bounds.count == transitionRows.count else {
            throw JPEGAIEntropyError.invalidTables
        }
        let name = jpegAIModelNames[model]
        gainY = try Self.gains(directory.appendingPathComponent("Y_\(name)_gain.csv"))
        gainUV = try Self.gains(directory.appendingPathComponent("UV_\(name)_gain.csv"))
    }

    private static func gains(_ url: URL) throws -> [Int32] {
        try readJPEGAICSV(url).flatMap { row in
            try row.map {
                guard let value = Int32(exactly: $0) else {
                    throw JPEGAIEntropyError.invalidTables
                }
                return value
            }
        }
    }
}

public struct JPEGAIResidualComponent: Sendable {
    public let scaleLog: [Int32]
    public let mask: [UInt8]
    public let quantized: [Int16]
}

public struct JPEGAIResidualLatents: Sendable {
    public let y: JPEGAIResidualComponent
    public let uv: JPEGAIResidualComponent
    public let height: Int
    public let width: Int
}

public struct JPEGAIDecodedImage: Sendable {
    public let width: Int
    public let height: Int
    public let rgb: [UInt8]

    public func writePNG(to url: URL) throws {
        guard let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
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
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw CocoaError(.fileWriteUnknown)
        }
    }
}

@available(macOS 13, iOS 16, *)
public extension JPEGAIBitstream {
    func decodeImage(
        tablesDirectory: URL, models: isolated JPEGAICoreMLModelSet
    ) async throws -> JPEGAIDecodedImage {
        let header = try pictureHeader
        let hyper = try decodeHyperLatents(
            using: JPEGAIZTables(directory: tablesDirectory, model: header.model)
        )
        let residualTables = try JPEGAIResidualTables(
            directory: tablesDirectory, model: header.model
        )
        let residuals = try await decodeResidualLatents(
            hyper: hyper, tables: residualTables, models: models
        )
        return try await reconstructImage(
            hyper: hyper, residuals: residuals, tables: residualTables, models: models
        )
    }

    func decodeResidualLatents(
        hyper: JPEGAIHyperLatents,
        tables: JPEGAIResidualTables,
        models: isolated JPEGAICoreMLModelSet
    ) async throws -> JPEGAIResidualLatents {
        let header = try pictureHeader
        let height = (header.codedHeight + 15) / 16
        let width = (header.codedWidth + 15) / 16

        func decode(
            component: String,
            marker: JPEGAIMarker,
            z: [Int8],
            channels: Int,
            gain: [Int32],
            beta: Int
        ) async throws -> JPEGAIResidualComponent {
            guard gain.count == channels,
                  let payload = substreams.first(where: { $0.marker == marker })?.payload else {
                throw JPEGAIEntropyError.invalidTables
            }
            let input = try JPEGAICoreMLModelSet.int32Array(
                z, channels: channels, height: hyper.height, width: hyper.width
            )
            let output = try await models.predict(
                tool: header.model, component: component,
                path: "common_modules/hyper_scale_decoder", inputs: [input]
            )
            var scale = try JPEGAICoreMLModelSet.int32Values(
                output, channels: channels, height: height, width: width
            )
            let plane = height * width
            for index in scale.indices {
                scale[index] += gain[index / plane] + Int32(beta)
            }
            let mask = scale.map { UInt8($0 > 382 ? 1 : 0) }
            let lastDistribution = Int32(tables.bounds.count - 1)
            let indexes = scale.map {
                UInt8(clamping: max(0, min(($0 + 64) >> 7, lastDistribution)))
            }
            let decoder = try JPEGAIEntropyDecoder(stream: payload)
            try decoder.setSGMTables(transitions: tables.transitions, bounds: tables.bounds)
            return JPEGAIResidualComponent(
                scaleLog: scale,
                mask: mask,
                quantized: try decoder.decodeSGM(indexes: indexes, masks: mask)
            )
        }

        let y = try await decode(
            component: "model_y", marker: .primaryResidual, z: hyper.y,
            channels: header.channelsY, gain: tables.gainY, beta: header.betaDisplacementY
        )
        let uv = try await decode(
            component: "model_uv", marker: .secondaryResidual, z: hyper.uv,
            channels: header.channelsUV, gain: tables.gainUV, beta: header.betaDisplacementUV
        )
        return JPEGAIResidualLatents(y: y, uv: uv, height: height, width: width)
    }

    func reconstructImage(
        hyper: JPEGAIHyperLatents,
        residuals: JPEGAIResidualLatents,
        tables: JPEGAIResidualTables,
        models: isolated JPEGAICoreMLModelSet
    ) async throws -> JPEGAIDecodedImage {
        let header = try pictureHeader
        guard header.codedChromaSubsampling == (1, 1),
              header.sourceChromaSubsampling == (1, 1) else {
            throw JPEGAIBitstreamError.unsupportedFeature("subsampled chroma")
        }
        guard substreams.first(where: { $0.marker == .toolHeader })?
            .payload.allSatisfy({ $0 == 0 }) != false else {
            throw JPEGAIBitstreamError.unsupportedFeature("optional decoder tools")
        }
        let height = residuals.height
        let width = residuals.width
        let stageHeight = (height + 1) / 2
        let stageWidth = (width + 1) / 2

        func hyperParameters(_ component: String, _ values: [Int8], _ channels: Int) async throws -> [Float32] {
            let input = try JPEGAICoreMLModelSet.floatArray(
                values, channels: channels, height: hyper.height, width: hyper.width
            )
            let output = try await models.predict(
                tool: header.model, component: component,
                path: "common_modules/hyper_decoder", inputs: [input]
            )
            return try JPEGAICoreMLModelSet.float32Values(
                output, channels: channels * 4, height: stageHeight, width: stageWidth
            )
        }

        let psiY = try await hyperParameters("model_y", hyper.y, header.channelsY)
        let psiUV = try await hyperParameters("model_uv", hyper.uv, header.channelsUV)
        let residualY = Self.dequantize(
            residuals.y.quantized, gain: tables.gainY, beta: header.betaDisplacementY,
            height: height, width: width
        )
        let residualUV = Self.dequantize(
            residuals.uv.quantized, gain: tables.gainUV, beta: header.betaDisplacementUV,
            height: height, width: width
        )

        let yParts = Self.downShuffle(
            residualY, channels: header.channelsY, height: height, width: width
        )
        let psiYParts = Self.channelChunks(psiY, count: 4)
        var reconstructedYParts = [[Float32]]()
        for stage in 0 ..< 4 {
            var inputs = [try JPEGAICoreMLModelSet.floatArray(
                psiYParts[stage], channels: header.channelsY,
                height: stageHeight, width: stageWidth
            )]
            if !reconstructedYParts.isEmpty {
                inputs.append(try JPEGAICoreMLModelSet.floatArray(
                    reconstructedYParts.flatMap { $0 },
                    channels: header.channelsY * reconstructedYParts.count,
                    height: stageHeight, width: stageWidth
                ))
            }
            let prediction = try await models.predict(
                tool: header.model, component: "model_y",
                path: "common_modules/MCM/stage\(stage)", inputs: inputs
            )
            let mean = try JPEGAICoreMLModelSet.float32Values(
                prediction, channels: header.channelsY,
                height: stageHeight, width: stageWidth
            )
            reconstructedYParts.append(zip(mean, yParts[stage]).map(+))
        }
        let yHat = Self.upShuffle(
            reconstructedYParts, channels: header.channelsY,
            height: height, width: width
        )
        let uvMean = Self.upShuffle(
            Self.channelChunks(psiUV, count: 4), channels: header.channelsUV,
            height: height, width: width
        )
        let uvHat = zip(uvMean, residualUV).map(+)

        let yOutput = try await models.predict(
            tool: header.model, component: "model_y", path: "synthesis",
            inputs: [try JPEGAICoreMLModelSet.floatArray(
                yHat, channels: header.channelsY, height: height, width: width
            )]
        )
        let uvOutput = try await models.predict(
            tool: header.model, component: "model_uv", path: "synthesis",
            inputs: [try JPEGAICoreMLModelSet.floatArray(
                yHat + uvHat, channels: header.channelsY + header.channelsUV,
                height: height, width: width
            )]
        )
        let pixelHeight = header.displayHeight
        let pixelWidth = header.displayWidth
        let y = try JPEGAICoreMLModelSet.float32Values(
            yOutput, channels: 1, height: pixelHeight, width: pixelWidth
        )
        let uv = try JPEGAICoreMLModelSet.float32Values(
            uvOutput, channels: 2, height: pixelHeight, width: pixelWidth
        )
        return JPEGAIDecodedImage(
            width: pixelWidth, height: pixelHeight,
            rgb: Self.rgb(y: y, uv: uv, pixels: pixelHeight * pixelWidth)
        )
    }

    private static func dequantize(
        _ values: [Int16], gain: [Int32], beta: Int, height: Int, width: Int
    ) -> [Float32] {
        let plane = height * width
        let logK = Float32((log(100.0) - log(0.11)) / 34.0)
        var result = [Float32](repeating: 0, count: values.count)
        for channel in gain.indices {
            let scaler = (
                exp(Float32(gain[channel] + Int32(beta)) * logK / 128) * 1024
            ).rounded() / 1024
            let offset = channel * plane
            for index in 0 ..< plane {
                result[offset + index] = Float32(values[offset + index]) / (scaler + 1e-9)
            }
        }
        return result
    }

    private static func channelChunks(_ values: [Float32], count: Int) -> [[Float32]] {
        let size = values.count / count
        return (0 ..< count).map { Array(values[$0 * size ..< ($0 + 1) * size]) }
    }

    internal static func downShuffle(
        _ values: [Float32], channels: Int, height: Int, width: Int
    ) -> [[Float32]] {
        let outputHeight = (height + 1) / 2
        let outputWidth = (width + 1) / 2
        let inputPlane = height * width
        let outputPlane = outputHeight * outputWidth
        var parts = Array(repeating: [Float32](repeating: 0, count: channels * outputPlane), count: 4)
        let stageForPosition = [[0, 2], [3, 1]]
        for channel in 0 ..< channels {
            for row in 0 ..< height {
                for column in 0 ..< width {
                    let stage = stageForPosition[row % 2][column % 2]
                    parts[stage][channel * outputPlane + row / 2 * outputWidth + column / 2] =
                        values[channel * inputPlane + row * width + column]
                }
            }
        }
        return parts
    }

    internal static func upShuffle(
        _ parts: [[Float32]], channels: Int, height: Int, width: Int
    ) -> [Float32] {
        let inputHeight = (height + 1) / 2
        let inputWidth = (width + 1) / 2
        let inputPlane = inputHeight * inputWidth
        let outputPlane = height * width
        let stageForPosition = [[0, 2], [3, 1]]
        var result = [Float32](repeating: 0, count: channels * outputPlane)
        for channel in 0 ..< channels {
            for row in 0 ..< height {
                for column in 0 ..< width {
                    let stage = stageForPosition[row % 2][column % 2]
                    result[channel * outputPlane + row * width + column] =
                        parts[stage][channel * inputPlane + row / 2 * inputWidth + column / 2]
                }
            }
        }
        return result
    }

    private static func rgb(y: [Float32], uv: [Float32], pixels: Int) -> [UInt8] {
        let kr: Float32 = 0.2126
        let kg: Float32 = 0.7152
        let kb: Float32 = 0.0722
        let kby: Float32 = 1.8556
        let kry: Float32 = 1.5748
        func byte(_ value: Float32) -> UInt8 {
            UInt8(max(0, min(255, value)).rounded(.toNearestOrEven))
        }
        var result = [UInt8](repeating: 0, count: pixels * 3)
        for index in 0 ..< pixels {
            let luma = y[index]
            let u = uv[index] - 127.5
            let v = uv[pixels + index] - 127.5
            result[index * 3] = byte(luma + kry * v)
            result[index * 3 + 1] = byte(luma - kb * kby / kg * u - kr * kry / kg * v)
            result[index * 3 + 2] = byte(luma + kby * u)
        }
        return result
    }
}
#endif
