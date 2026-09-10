#if canImport(CoreML)
import CoreGraphics
import CoreML
import Foundation
import ImageIO

public struct JPEGAIEncodedImage: Sendable {
    public let data: Data
    public let width: Int
    public let height: Int
    public let model: Int
    public let beta: Int
    public let diagnostics: JPEGAIEncodingDiagnostics?

    public func write(to url: URL) throws { try data.write(to: url, options: .atomic) }
}

public enum JPEGAIEncodeError: LocalizedError {
    case invalidPNG
    case unsupportedDimensions(width: Int, height: Int)
    case invalidModel
    case invalidBeta

    public var errorDescription: String? {
        switch self {
        case .invalidPNG:
            "The selected file is not a readable PNG image."
        case let .unsupportedDimensions(width, height):
            "The native encoder currently supports images from 64×64 through 4096×4096; got \(width)×\(height)."
        case .invalidModel:
            "The JPEG AI model index must be between 0 and 3."
        case .invalidBeta:
            "The JPEG AI beta displacement must be between -2048 and 2047."
        }
    }
}

public extension JPEGAIDecodedImage {
    static func readPNG(from url: URL) throws -> Self {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.width > 0, image.height > 0 else {
            throw JPEGAIEncodeError.invalidPNG
        }
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let colourSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: &rgba, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: colourSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            throw JPEGAIEncodeError.invalidPNG
        }
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< width * height {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return Self(width: width, height: height, rgb: rgb)
    }

    func encodeJPEGAI(
        tablesDirectory: URL,
        models: isolated JPEGAICoreMLModelSet,
        model: Int,
        beta: Int,
        includeDiagnostics: Bool = false
    ) async throws -> JPEGAIEncodedImage {
        guard (0 ..< 4).contains(model) else { throw JPEGAIEncodeError.invalidModel }
        guard (-2048 ... 2047).contains(beta) else { throw JPEGAIEncodeError.invalidBeta }
        let codedWidth = width + width % 2
        let codedHeight = height + height % 2
        // ponytail: converted Core ML graphs cap each side at 4096; widen their flexible shapes when larger images matter.
        guard width >= 64, height >= 64, codedWidth <= 4096, codedHeight <= 4096 else {
            throw JPEGAIEncodeError.unsupportedDimensions(width: width, height: height)
        }

        let tables = try EncoderTables(directory: tablesDirectory, model: model)
        let planes = Self.yuvPlanes(
            rgb: rgb, width: width, height: height,
            codedWidth: codedWidth, codedHeight: codedHeight
        )
        let latentHeight = (codedHeight + 15) / 16
        let latentWidth = (codedWidth + 15) / 16
        let hyperHeight = (codedHeight + 63) / 64
        let hyperWidth = (codedWidth + 63) / 64
        let uvInput = Self.pixelUnshuffle(
            planes.y, channels: 1, height: codedHeight, width: codedWidth
        ) + Self.pixelUnshuffle(
            planes.u + planes.v, channels: 2, height: codedHeight, width: codedWidth
        )

        func transform(
            component: String, input: [Float32], inputChannels: Int,
            inputHeight: Int, inputWidth: Int, channels: Int, gain: [Int32],
            contextual: Bool
        ) async throws -> EncodedComponent {
            let analysis = try await models.predict(
                tool: model, component: component, path: "analysis",
                inputs: [try JPEGAICoreMLModelSet.floatArray(
                    input, channels: inputChannels, height: inputHeight, width: inputWidth
                )]
            )
            let y = try JPEGAICoreMLModelSet.float32Values(
                analysis, channels: channels, height: latentHeight, width: latentWidth
            )
            let hyper = try await models.predict(
                tool: model, component: component, path: "common_modules/hyper_encoder",
                inputs: [try JPEGAICoreMLModelSet.floatArray(
                    y, channels: channels, height: latentHeight, width: latentWidth
                )]
            )
            let z = try JPEGAICoreMLModelSet.float32Values(
                hyper, channels: channels, height: hyperHeight, width: hyperWidth
            ).map { value -> Int8 in
                Int8(max(-31, min(31, Int(value.rounded(.toNearestOrEven)))))
            }
            let zInput = try JPEGAICoreMLModelSet.int32Array(
                z, channels: channels, height: hyperHeight, width: hyperWidth
            )
            let scaleOutput = try await models.predict(
                tool: model, component: component,
                path: "common_modules/hyper_scale_decoder", inputs: [zInput]
            )
            var scaleLog = try JPEGAICoreMLModelSet.int32Values(
                scaleOutput, channels: channels, height: latentHeight, width: latentWidth
            )
            let plane = latentHeight * latentWidth
            guard gain.count == channels else { throw JPEGAIEntropyError.invalidTables }
            for index in scaleLog.indices {
                scaleLog[index] += gain[index / plane] + Int32(beta)
            }
            let masks = scaleLog.map { UInt8($0 > 382 ? 1 : 0) }
            let lastDistribution = Int32(tables.residual.bounds.count - 1)
            let indexes = scaleLog.map {
                UInt8(clamping: max(0, min(($0 + 64) >> 7, lastDistribution)))
            }
            let psiOutput = try await models.predict(
                tool: model, component: component,
                path: "common_modules/hyper_decoder", inputs: [
                    try JPEGAICoreMLModelSet.floatArray(
                        z, channels: channels, height: hyperHeight, width: hyperWidth
                    ),
                ]
            )
            let stageHeight = (latentHeight + 1) / 2
            let stageWidth = (latentWidth + 1) / 2
            let psi = try JPEGAICoreMLModelSet.float32Values(
                psiOutput, channels: channels * 4, height: stageHeight, width: stageWidth
            )
            let scalers = Self.scalers(gain: gain, beta: beta)
            let quantized: [Int16]
            if contextual {
                let yParts = JPEGAIBitstream.downShuffle(
                    y, channels: channels, height: latentHeight, width: latentWidth
                )
                let maskParts = JPEGAIBitstream.downShuffle(
                    masks.map(Float32.init), channels: channels,
                    height: latentHeight, width: latentWidth
                )
                let psiParts = Self.channelChunks(psi, count: 4)
                let stagePlane = stageHeight * stageWidth
                var reconstructed = [[Float32]]()
                var residualParts = [[Float32]]()
                for stage in 0 ..< 4 {
                    var inputs = [try JPEGAICoreMLModelSet.floatArray(
                        psiParts[stage], channels: channels,
                        height: stageHeight, width: stageWidth
                    )]
                    if !reconstructed.isEmpty {
                        inputs.append(try JPEGAICoreMLModelSet.floatArray(
                            reconstructed.flatMap { $0 },
                            channels: channels * reconstructed.count,
                            height: stageHeight, width: stageWidth
                        ))
                    }
                    let prediction = try await models.predict(
                        tool: model, component: component,
                        path: "common_modules/MCM/stage\(stage)", inputs: inputs
                    )
                    let mean = try JPEGAICoreMLModelSet.float32Values(
                        prediction, channels: channels,
                        height: stageHeight, width: stageWidth
                    )
                    var residual = [Float32](repeating: 0, count: mean.count)
                    var yHat = mean
                    for index in mean.indices where maskParts[stage][index] != 0 {
                        let channel = index / stagePlane
                        let value = Self.quantize(yParts[stage][index] - mean[index], scalers[channel])
                        residual[index] = Float32(value)
                        yHat[index] += Float32(value) / (scalers[channel] + 1e-9)
                    }
                    residualParts.append(residual)
                    reconstructed.append(yHat)
                }
                quantized = JPEGAIBitstream.upShuffle(
                    residualParts, channels: channels,
                    height: latentHeight, width: latentWidth
                ).map { Int16($0) }
            } else {
                let mean = JPEGAIBitstream.upShuffle(
                    Self.channelChunks(psi, count: 4), channels: channels,
                    height: latentHeight, width: latentWidth
                )
                quantized = y.indices.map { index in
                    guard masks[index] != 0 else { return 0 }
                    return Self.quantize(y[index] - mean[index], scalers[index / plane])
                }
            }
            return EncodedComponent(
                latent: y, z: z, scaleLog: scaleLog,
                zSymbols: z.map { UInt8(Int($0) + 31) },
                indexes: indexes, masks: masks, quantized: quantized
            )
        }

        let y = try await transform(
            component: "model_y", input: planes.y, inputChannels: 1,
            inputHeight: codedHeight, inputWidth: codedWidth,
            channels: 160, gain: tables.residual.gainY, contextual: true
        )
        let uv = try await transform(
            component: "model_uv", input: uvInput, inputChannels: 12,
            inputHeight: codedHeight / 2, inputWidth: codedWidth / 2,
            channels: 96, gain: tables.residual.gainUV, contextual: false
        )

        let hyperEncoder = try JPEGAIEntropyEncoder(
            capacity: (y.zSymbols.count + uv.zSymbols.count) * 4 + 64
        )
        try hyperEncoder.encodeFactorized(
            cdfs: tables.z.uv, values: uv.zSymbols,
            channels: 96, valuesPerChannel: hyperHeight * hyperWidth
        )
        try hyperEncoder.encodeFactorized(
            cdfs: tables.z.y, values: y.zSymbols,
            channels: 160, valuesPerChannel: hyperHeight * hyperWidth
        )
        let hyperPayload = try hyperEncoder.finish()

        func residualPayload(_ component: EncodedComponent) throws -> Data {
            let encoder = try JPEGAIEntropyEncoder(capacity: component.quantized.count * 4 + 64)
            try encoder.setSGMTables(
                transitions: tables.transitions,
                bounds: tables.residual.bounds,
                stateMaps: tables.stateMaps
            )
            try encoder.encodeSGM(
                indexes: component.indexes,
                values: component.quantized,
                masks: component.masks
            )
            return try encoder.finish()
        }

        let data = try JPEGAIBitstream.simpleProfileData(
            codedWidth: codedWidth, codedHeight: codedHeight,
            displayWidth: width, displayHeight: height,
            model: model, beta: beta,
            primaryResidual: residualPayload(y),
            secondaryResidual: residualPayload(uv),
            hyperLatent: hyperPayload
        )
        return JPEGAIEncodedImage(
            data: data, width: width, height: height, model: model, beta: beta,
            diagnostics: includeDiagnostics ? JPEGAIEncodingDiagnostics(
                luma: planes.y, pixelWidth: codedWidth, pixelHeight: codedHeight,
                y: y.latent, yChannels: 160,
                z: y.z.map(Float32.init), zChannels: 160,
                latentWidth: latentWidth, latentHeight: latentHeight,
                hyperWidth: hyperWidth, hyperHeight: hyperHeight,
                mask: y.masks
            ) : nil
        )
    }

    private static func yuvPlanes(
        rgb: [UInt8], width: Int, height: Int, codedWidth: Int, codedHeight: Int
    ) -> (y: [Float32], u: [Float32], v: [Float32]) {
        let count = codedWidth * codedHeight
        var y = [Float32](repeating: 0, count: count)
        var u = y
        var v = y
        for row in 0 ..< codedHeight {
            for column in 0 ..< codedWidth {
                let source = (min(row, height - 1) * width + min(column, width - 1)) * 3
                let destination = row * codedWidth + column
                let red = Float32(rgb[source])
                let green = Float32(rgb[source + 1])
                let blue = Float32(rgb[source + 2])
                let luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                y[destination] = luma
                u[destination] = (blue - luma) / 1.8556 + 127.5
                v[destination] = (red - luma) / 1.5748 + 127.5
            }
        }
        return (y, u, v)
    }

    private static func pixelUnshuffle(
        _ values: [Float32], channels: Int, height: Int, width: Int
    ) -> [Float32] {
        let outputHeight = height / 2
        let outputWidth = width / 2
        let inputPlane = height * width
        let outputPlane = outputHeight * outputWidth
        var result = [Float32](repeating: 0, count: channels * 4 * outputPlane)
        for channel in 0 ..< channels {
            for row in 0 ..< outputHeight {
                for column in 0 ..< outputWidth {
                    for offset in 0 ..< 4 {
                        let inputRow = row * 2 + offset / 2
                        let inputColumn = column * 2 + offset % 2
                        result[(channel * 4 + offset) * outputPlane + row * outputWidth + column] =
                            values[channel * inputPlane + inputRow * width + inputColumn]
                    }
                }
            }
        }
        return result
    }

    private static func channelChunks(_ values: [Float32], count: Int) -> [[Float32]] {
        let size = values.count / count
        return (0 ..< count).map { Array(values[$0 * size ..< ($0 + 1) * size]) }
    }

    private static func scalers(gain: [Int32], beta: Int) -> [Float32] {
        let logK = Float32((log(100.0) - log(0.11)) / 34.0)
        return gain.map {
            (exp(Float32($0 + Int32(beta)) * logK / 128) * 1024).rounded() / 1024
        }
    }

    private static func quantize(_ value: Float32, _ scaler: Float32) -> Int16 {
        let rounded = (value * scaler).rounded(.toNearestOrEven)
        return Int16(max(Float32(Int16.min), min(Float32(Int16.max), rounded)))
    }
}

private struct EncoderTables {
    let z: JPEGAIZTables
    let residual: JPEGAIResidualTables
    let transitions: [UInt32]
    let stateMaps: [UInt8]

    init(directory: URL, model: Int) throws {
        z = try JPEGAIZTables(directory: directory, model: model)
        residual = try JPEGAIResidualTables(directory: directory, model: model)
        transitions = try Self.values(
            directory.appendingPathComponent("residual_encode_transitions.csv"), as: UInt32.self
        )
        stateMaps = try Self.values(
            directory.appendingPathComponent("residual_state_maps.csv"), as: UInt8.self
        )
        guard transitions.count == residual.bounds.count * 256,
              stateMaps.count == transitions.count else {
            throw JPEGAIEntropyError.invalidTables
        }
    }

    private static func values<T: FixedWidthInteger>(
        _ url: URL, as type: T.Type
    ) throws -> [T] {
        try readJPEGAICSV(url).flatMap { row in
            try row.map {
                guard let value = T(exactly: $0) else { throw JPEGAIEntropyError.invalidTables }
                return value
            }
        }
    }
}

private struct EncodedComponent {
    let latent: [Float32]
    let z: [Int8]
    let scaleLog: [Int32]
    let zSymbols: [UInt8]
    let indexes: [UInt8]
    let masks: [UInt8]
    let quantized: [Int16]
}

public struct JPEGAIEncodingDiagnostics: Sendable {
    fileprivate let luma: [Float32]
    fileprivate let pixelWidth: Int
    fileprivate let pixelHeight: Int
    fileprivate let y: [Float32]
    fileprivate let yChannels: Int
    fileprivate let z: [Float32]
    fileprivate let zChannels: Int
    fileprivate let latentWidth: Int
    fileprivate let latentHeight: Int
    fileprivate let hyperWidth: Int
    fileprivate let hyperHeight: Int
    fileprivate let mask: [UInt8]

    public func write(to directory: URL) throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        try Self.writePNG(
            rgb: luma.flatMap {
                let value = UInt8(max(0, min(255, $0)).rounded(.toNearestOrEven))
                return [value, value, value]
            },
            width: pixelWidth, height: pixelHeight,
            to: directory.appendingPathComponent("01-input-luma.png")
        )
        try Self.writePNG(
            rgb: Self.activationSheet(
                y, channels: yChannels, height: latentHeight, width: latentWidth,
                columns: 16, scale: 1
            ),
            width: latentWidth * 16, height: latentHeight * 10,
            to: directory.appendingPathComponent("02-y-latent-160-channels.png")
        )
        try Self.writePNG(
            rgb: Self.energyMap(
                y, channels: yChannels, height: latentHeight, width: latentWidth, scale: 10
            ),
            width: latentWidth * 10, height: latentHeight * 10,
            to: directory.appendingPathComponent("03-y-latent-energy.png")
        )
        try Self.writePNG(
            rgb: Self.activationSheet(
                z, channels: zChannels, height: hyperHeight, width: hyperWidth,
                columns: 16, scale: 4
            ),
            width: hyperWidth * 16 * 4, height: hyperHeight * 10 * 4,
            to: directory.appendingPathComponent("04-z-hyperlatent-160-channels.png")
        )
        try Self.writePNG(
            rgb: Self.energyMap(
                z, channels: zChannels, height: hyperHeight, width: hyperWidth, scale: 40
            ),
            width: hyperWidth * 40, height: hyperHeight * 40,
            to: directory.appendingPathComponent("05-z-hyperlatent-energy.png")
        )
        try Self.writePNG(
            rgb: Self.maskMap(
                mask, channels: yChannels, height: latentHeight, width: latentWidth, scale: 10
            ),
            width: latentWidth * 10, height: latentHeight * 10,
            to: directory.appendingPathComponent("06-entropy-mask-density.png")
        )
        let note = """
        # JPEG AI inference tensors

        These are tensors captured from one real native Core ML encode.

        1. `01-input-luma.png`: BT.709 luma entering the analysis transform.
        2. `02-y-latent-160-channels.png`: all 160 learned `y` feature channels. Orange is positive; blue is negative; each channel is normalized independently.
        3. `03-y-latent-energy.png`: mean absolute `y` activation at each spatial location.
        4. `04-z-hyperlatent-160-channels.png`: all 160 rounded hyper-latent `z` channels.
        5. `05-z-hyperlatent-energy.png`: mean absolute `z` activation.
        6. `06-entropy-mask-density.png`: fraction of `y` channels entropy-coded at each position.

        `y` shape: [1, \(yChannels), \(latentHeight), \(latentWidth)]
        `z` shape: [1, \(zChannels), \(hyperHeight), \(hyperWidth)]

        Learned channels are not RGB images. Their colors show signed activation, not semantic color.
        """
        try Data(note.utf8).write(
            to: directory.appendingPathComponent("README.md"), options: .atomic
        )
    }

    private static func activationSheet(
        _ values: [Float32], channels: Int, height: Int, width: Int,
        columns: Int, scale: Int
    ) -> [UInt8] {
        let rows = (channels + columns - 1) / columns
        let outputWidth = columns * width * scale
        let outputHeight = rows * height * scale
        let plane = height * width
        var output = [UInt8](repeating: 18, count: outputWidth * outputHeight * 3)
        for channel in 0 ..< channels {
            let start = channel * plane
            let maximum = max(1e-9, values[start ..< start + plane].map(abs).max() ?? 1)
            let tileX = channel % columns * width * scale
            let tileY = channel / columns * height * scale
            for row in 0 ..< height {
                for column in 0 ..< width {
                    let colour = signedColour(values[start + row * width + column] / maximum)
                    fill(
                        &output, colour: colour,
                        x: tileX + column * scale, y: tileY + row * scale,
                        scale: scale, width: outputWidth
                    )
                }
            }
        }
        return output
    }

    private static func energyMap(
        _ values: [Float32], channels: Int, height: Int, width: Int, scale: Int
    ) -> [UInt8] {
        let plane = height * width
        var energy = [Float32](repeating: 0, count: plane)
        for channel in 0 ..< channels {
            for index in 0 ..< plane { energy[index] += abs(values[channel * plane + index]) }
        }
        energy = energy.map { $0 / Float32(channels) }
        let sorted = energy.sorted()
        let ceiling = max(1e-9, sorted[Int(Double(sorted.count - 1) * 0.98)])
        return scaledMap(energy.map { heatColour(min(1, $0 / ceiling)) }, width: width, scale: scale)
    }

    private static func maskMap(
        _ values: [UInt8], channels: Int, height: Int, width: Int, scale: Int
    ) -> [UInt8] {
        let plane = height * width
        var density = [Float32](repeating: 0, count: plane)
        for channel in 0 ..< channels {
            for index in 0 ..< plane {
                density[index] += Float32(values[channel * plane + index]) / Float32(channels)
            }
        }
        return scaledMap(density.map(heatColour), width: width, scale: scale)
    }

    private static func scaledMap(
        _ colours: [(UInt8, UInt8, UInt8)], width: Int, scale: Int
    ) -> [UInt8] {
        let height = colours.count / width
        let outputWidth = width * scale
        var output = [UInt8](repeating: 0, count: outputWidth * height * scale * 3)
        for row in 0 ..< height {
            for column in 0 ..< width {
                fill(
                    &output, colour: colours[row * width + column],
                    x: column * scale, y: row * scale, scale: scale, width: outputWidth
                )
            }
        }
        return output
    }

    private static func fill(
        _ output: inout [UInt8], colour: (UInt8, UInt8, UInt8),
        x: Int, y: Int, scale: Int, width: Int
    ) {
        for row in y ..< y + scale {
            for column in x ..< x + scale {
                let index = (row * width + column) * 3
                (output[index], output[index + 1], output[index + 2]) = colour
            }
        }
    }

    private static func signedColour(_ value: Float32) -> (UInt8, UInt8, UInt8) {
        let amount = max(-1, min(1, value))
        if amount >= 0 {
            return (UInt8(128 + 127 * amount), UInt8(128 - 50 * amount), UInt8(128 - 110 * amount))
        }
        let magnitude = -amount
        return (UInt8(128 - 105 * magnitude), UInt8(128 + 20 * magnitude), UInt8(128 + 127 * magnitude))
    }

    private static func heatColour(_ value: Float32) -> (UInt8, UInt8, UInt8) {
        let value = max(0, min(1, value))
        if value < 0.5 {
            let t = value * 2
            return (UInt8(12 + 18 * t), UInt8(25 + 190 * t), UInt8(70 + 145 * t))
        }
        let t = (value - 0.5) * 2
        return (UInt8(30 + 225 * t), UInt8(215 + 30 * t), UInt8(215 - 185 * t))
    }

    private static func writePNG(rgb: [UInt8], width: Int, height: Int, to url: URL) throws {
        try JPEGAIDecodedImage(width: width, height: height, rgb: rgb).writePNG(to: url)
    }
}

extension JPEGAIBitstream {
    static func simpleProfileData(
        codedWidth: Int, codedHeight: Int,
        displayWidth: Int, displayHeight: Int,
        model: Int, beta: Int,
        primaryResidual: Data, secondaryResidual: Data, hyperLatent: Data
    ) throws -> Data {
        guard codedWidth >= 64, codedHeight >= 64,
              codedWidth - 64 <= 65535, codedHeight - 64 <= 65535,
              (0 ... 63).contains(codedWidth - displayWidth),
              (0 ... 63).contains(codedHeight - displayHeight) else {
            throw JPEGAIBitstreamError.invalidHeader
        }
        var header = MSBBitWriter()
        header.write(0, count: 4) // stream profile
        header.write(0, count: 4) // decoder profile
        header.write(0, count: 4) // one synthesis transform
        header.write(0, count: 4) // synthesis transform zero
        header.write(52, count: 8)
        header.write(codedWidth - 64, count: 16)
        header.write(codedHeight - 64, count: 16)
        header.write(codedWidth - displayWidth, count: 6)
        header.write(codedHeight - displayHeight, count: 6)
        header.write(0, count: 3) // 8-bit source
        header.write(0, count: 1) // source vertical 4:4:4
        header.write(0, count: 1) // source horizontal 4:4:4
        header.write(0, count: 1) // coded vertical 4:4:4
        header.write(0, count: 1) // coded horizontal 4:4:4
        header.write(1, count: 2) // BT.709 RGB-to-YUV
        header.write(model, count: 4)
        header.write(0, count: 1) // one z entropy thread
        header.write(beta + 2048, count: 12)
        header.write(0, count: 1) // no regions
        header.write(0, count: 1) // same beta for Y and UV
        for channels in [160, 96] {
            header.write(0, count: 1) // one residual entropy thread
            header.write(channels, count: 8)
            header.write(0, count: 4) // no optional component tools
        }
        header.write(0, count: 1) // no quality map

        var data = Data([0xFF, 0x80])
        data.appendSubstream(.pictureHeader, payload: header.data)
        data.appendSubstream(.toolHeader, payload: Data([0]))
        data.appendSubstream(.renderingInformation, payload: Data([0]))
        data.appendSubstream(.primaryResidual, payload: primaryResidual)
        data.appendSubstream(.secondaryResidual, payload: secondaryResidual)
        data.appendSubstream(.hyperLatent, payload: hyperLatent)
        data.append(contentsOf: [0xFF, 0x81])
        return data
    }
}

private struct MSBBitWriter {
    private var bytes = [UInt8]()
    private var bitCount = 0

    mutating func write(_ value: Int, count: Int) {
        precondition(count > 0 && value >= 0 && (count == Int.bitWidth || value < 1 << count))
        for shift in (0 ..< count).reversed() {
            if bitCount % 8 == 0 { bytes.append(0) }
            bytes[bytes.count - 1] |= UInt8(value >> shift & 1) << (7 - bitCount % 8)
            bitCount += 1
        }
    }

    var data: Data { Data(bytes) }
}

private extension Data {
    mutating func appendSubstream(_ marker: JPEGAIMarker, payload: Data) {
        append(UInt8(marker.rawValue >> 8))
        append(UInt8(marker.rawValue & 0xFF))
        var size = MSBBitWriter()
        let code = payload.count + 1
        let bits = Int.bitWidth - code.leadingZeroBitCount
        if bits > 1 { size.write(0, count: bits - 1) }
        size.write(code, count: bits)
        append(size.data)
        append(payload)
    }
}
#endif
