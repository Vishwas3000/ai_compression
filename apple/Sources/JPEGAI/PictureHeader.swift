import Foundation

public struct JPEGAIPictureHeader: Sendable {
    public let streamProfile: Int
    public let decoderProfile: Int
    public let synthesisTransforms: [Int]
    public let level: Int
    public let codedWidth: Int
    public let codedHeight: Int
    public let displayWidth: Int
    public let displayHeight: Int
    public let bitDepth: Int
    public let sourceChromaSubsampling: (vertical: Int, horizontal: Int)
    public let codedChromaSubsampling: (vertical: Int, horizontal: Int)
    public let colourTransform: Int
    public let model: Int
    public let betaDisplacementY: Int
    public let betaDisplacementUV: Int
    public let channelsY: Int
    public let channelsUV: Int

    public init(payload: Data) throws {
        var bits = PayloadBitReader(payload)
        streamProfile = try bits.read(4)
        decoderProfile = try bits.read(4)
        let transformCount = try bits.read(4) + 1
        synthesisTransforms = try (0 ..< transformCount).map { _ in try bits.read(4) }
        level = try bits.read(8)
        codedWidth = try bits.read(16) + 64
        codedHeight = try bits.read(16) + 64
        displayWidth = codedWidth - (try bits.read(6))
        displayHeight = codedHeight - (try bits.read(6))
        let depths = [8, 10, 12, 14, 16]
        let depthIndex = try bits.read(3)
        guard depths.indices.contains(depthIndex) else { throw JPEGAIBitstreamError.invalidHeader }
        bitDepth = depths[depthIndex]
        let sourceVertical = try bits.read(1) + 1
        let sourceHorizontal = try bits.read(1) + 1
        let codedVertical = sourceVertical == 1 ? try bits.read(1) + 1 : 2
        let codedHorizontal = sourceHorizontal == 1 ? try bits.read(1) + 1 : 2
        sourceChromaSubsampling = (sourceVertical, sourceHorizontal)
        codedChromaSubsampling = (codedVertical, codedHorizontal)

        colourTransform = try bits.read(2)
        guard colourTransform != 2 else {
            throw JPEGAIBitstreamError.unsupportedFeature("custom colour transform")
        }
        model = try bits.read(4)
        _ = try Self.threadCount(from: &bits, name: "z")
        betaDisplacementY = try bits.read(12) - 2048
        guard try bits.read(1) == 0 else {
            throw JPEGAIBitstreamError.unsupportedFeature("region partitioning")
        }
        if try bits.read(1) == 1 {
            betaDisplacementUV = try bits.read(12) - 2048
        } else {
            betaDisplacementUV = betaDisplacementY
        }
        channelsY = try Self.readComponentHeader(&bits, name: "Y")
        channelsUV = try Self.readComponentHeader(&bits, name: "UV")
        guard try bits.read(1) == 0 else {
            throw JPEGAIBitstreamError.unsupportedFeature("quality map")
        }
    }

    private static func threadCount(from bits: inout PayloadBitReader, name: String) throws -> Int {
        // ponytail: one entropy thread covers the simple profile; add framing when multi-thread streams matter.
        guard try bits.read(1) == 1 else { return 1 }
        let count = 1 << (try bits.read(2) + 1)
        guard count == 1 else { throw JPEGAIBitstreamError.unsupportedFeature("multi-threaded \(name)") }
        return count
    }

    private static func readComponentHeader(_ bits: inout PayloadBitReader, name: String) throws -> Int {
        _ = try threadCount(from: &bits, name: name)
        let channels = try bits.read(8)
        guard try bits.read(1) == 0 else { throw JPEGAIBitstreamError.unsupportedFeature("skip map") }
        guard try bits.read(1) == 0 else { throw JPEGAIBitstreamError.unsupportedFeature("residual scaling") }
        guard try bits.read(1) == 0 else { throw JPEGAIBitstreamError.unsupportedFeature("channel gain") }
        guard try bits.read(1) == 0 else { throw JPEGAIBitstreamError.unsupportedFeature("synthesis tiling") }
        return channels
    }
}

private struct PayloadBitReader {
    private let bytes: [UInt8]
    private var offset = 0

    init(_ data: Data) { bytes = [UInt8](data) }

    mutating func read(_ count: Int) throws -> Int {
        guard count > 0, offset <= bytes.count * 8 - count else {
            throw JPEGAIBitstreamError.truncatedHeader
        }
        var value = 0
        for _ in 0 ..< count {
            value = value << 1 | Int(bytes[offset / 8] >> (7 - offset % 8) & 1)
            offset += 1
        }
        return value
    }
}
