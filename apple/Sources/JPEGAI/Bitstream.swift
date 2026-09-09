import Foundation

public enum JPEGAIMarker: UInt16, Sendable {
    case start = 0xFF80
    case end = 0xFF81
    case pictureHeader = 0xFF82
    case toolHeader = 0xFF83
    case renderingInformation = 0xFF84
    case hyperLatent = 0xFF88
    case primaryResidual = 0xFF89
    case secondaryResidual = 0xFF8A
    case qualityMap = 0xFF8B
    case userData = 0xFF8C
}

public struct JPEGAISubstream: Sendable {
    public let marker: JPEGAIMarker
    public let payload: Data
}

public struct JPEGAIBitstream: Sendable {
    public let substreams: [JPEGAISubstream]

    public var pictureHeader: JPEGAIPictureHeader {
        get throws {
            guard let payload = substreams.first(where: { $0.marker == .pictureHeader })?.payload else {
                throw JPEGAIBitstreamError.missingPictureHeader
            }
            return try JPEGAIPictureHeader(payload: payload)
        }
    }

    public init(data: Data) throws {
        let bytes = [UInt8](data)
        var offset = 0

        guard try Self.readMarker(bytes, at: &offset) == .start else {
            throw JPEGAIBitstreamError.missingStartMarker
        }

        var parsed: [JPEGAISubstream] = []
        while offset < bytes.count {
            let marker = try Self.readMarker(bytes, at: &offset)
            if marker == .end {
                guard offset == bytes.count else { throw JPEGAIBitstreamError.trailingData }
                guard parsed.first?.marker == .pictureHeader else {
                    throw JPEGAIBitstreamError.missingPictureHeader
                }
                substreams = parsed
                return
            }
            guard marker != .start else { throw JPEGAIBitstreamError.unexpectedStartMarker }

            let size = try Self.readUnsignedExpGolomb(bytes, at: &offset)
            guard size <= bytes.count - offset else { throw JPEGAIBitstreamError.truncatedPayload }
            parsed.append(JPEGAISubstream(marker: marker, payload: Data(bytes[offset ..< offset + size])))
            offset += size
        }
        throw JPEGAIBitstreamError.missingEndMarker
    }

    public init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    private static func readMarker(_ bytes: [UInt8], at offset: inout Int) throws -> JPEGAIMarker {
        guard offset <= bytes.count - 2 else { throw JPEGAIBitstreamError.truncatedMarker }
        let raw = UInt16(bytes[offset]) << 8 | UInt16(bytes[offset + 1])
        offset += 2
        guard let marker = JPEGAIMarker(rawValue: raw) else {
            throw JPEGAIBitstreamError.unknownMarker(raw)
        }
        return marker
    }

    private static func readUnsignedExpGolomb(_ bytes: [UInt8], at offset: inout Int) throws -> Int {
        var bit = offset * 8
        var leadingZeros = 0
        while try readBit(bytes, at: &bit) == 0 {
            leadingZeros += 1
            guard leadingZeros < Int.bitWidth - 1 else { throw JPEGAIBitstreamError.sizeOverflow }
        }

        var suffix = 0
        for _ in 0 ..< leadingZeros {
            suffix = suffix << 1 | Int(try readBit(bytes, at: &bit))
        }
        offset = (bit + 7) / 8
        return (1 << leadingZeros) + suffix - 1
    }

    private static func readBit(_ bytes: [UInt8], at bit: inout Int) throws -> UInt8 {
        guard bit < bytes.count * 8 else { throw JPEGAIBitstreamError.truncatedSize }
        defer { bit += 1 }
        return bytes[bit / 8] >> (7 - bit % 8) & 1
    }
}

public enum JPEGAIBitstreamError: Error, Equatable {
    case missingStartMarker
    case missingPictureHeader
    case missingEndMarker
    case unexpectedStartMarker
    case unknownMarker(UInt16)
    case truncatedMarker
    case truncatedSize
    case truncatedPayload
    case trailingData
    case sizeOverflow
    case truncatedHeader
    case invalidHeader
    case unsupportedFeature(String)
}
