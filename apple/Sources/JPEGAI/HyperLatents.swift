import Foundation

public struct JPEGAIZTables: Sendable {
    let y: [UInt8]
    let uv: [UInt8]

    public init(directory: URL, model: Int) throws {
        let names = ["0.002", "0.012", "0.075", "0.5"]
        guard names.indices.contains(model) else { throw JPEGAIEntropyError.invalidTables }
        let distributions = try Self.csv(directory.appendingPathComponent("unique_z_distributions.csv"))
        guard distributions.count == 128, distributions.allSatisfy({ $0.count == 63 }) else {
            throw JPEGAIEntropyError.invalidTables
        }
        let normalized = try distributions.map { row -> [UInt8] in
            let total = row.reduce(0, +)
            guard total > 0 else { throw JPEGAIEntropyError.invalidTables }
            var cumulative: Int64 = 0
            return row.map {
                cumulative += $0
                return UInt8((cumulative * 255 + total / 2) / total)
            }
        }
        y = try Self.mapped("Y", names[model], directory, normalized)
        uv = try Self.mapped("UV", names[model], directory, normalized)
    }

    private static func mapped(
        _ component: String, _ model: String, _ directory: URL, _ distributions: [[UInt8]]
    ) throws -> [UInt8] {
        let rows = try csv(directory.appendingPathComponent("\(component)_\(model).csv"))
        return try rows.flatMap { row in
            guard row.count == 1, distributions.indices.contains(Int(row[0])) else {
                throw JPEGAIEntropyError.invalidTables
            }
            return distributions[Int(row[0])]
        }
    }

    private static func csv(_ url: URL) throws -> [[Int64]] {
        let text = try String(contentsOf: url, encoding: .utf8)
        return try text.split(whereSeparator: \.isNewline).map { line in
            try line.split(separator: ",").map {
                guard let value = Int64($0.trimmingCharacters(in: .whitespaces)) else {
                    throw JPEGAIEntropyError.invalidTables
                }
                return value
            }
        }
    }
}

public struct JPEGAIHyperLatents: Sendable {
    public let ySymbols: [UInt8]
    public let uvSymbols: [UInt8]
    public let height: Int
    public let width: Int

    public var y: [Int8] { ySymbols.map { Int8(Int($0) - 31) } }
    public var uv: [Int8] { uvSymbols.map { Int8(Int($0) - 31) } }
}

public extension JPEGAIBitstream {
    func decodeHyperLatents(using tables: JPEGAIZTables) throws -> JPEGAIHyperLatents {
        let header = try pictureHeader
        guard header.decoderProfile == 0, header.synthesisTransforms.first == 0 else {
            throw JPEGAIBitstreamError.unsupportedFeature("non-simple decoder profile")
        }
        guard tables.y.count == header.channelsY * 63,
              tables.uv.count == header.channelsUV * 63,
              let payload = substreams.first(where: { $0.marker == .hyperLatent })?.payload else {
            throw JPEGAIEntropyError.invalidTables
        }
        let height = (header.codedHeight + 63) / 64
        let width = (header.codedWidth + 63) / 64
        let decoder = try JPEGAIEntropyDecoder(stream: payload)
        let y = try decoder.decodeFactorized(
            cdfs: tables.y, channels: header.channelsY, valuesPerChannel: height * width
        )
        let uv = try decoder.decodeFactorized(
            cdfs: tables.uv, channels: header.channelsUV, valuesPerChannel: height * width
        )
        return JPEGAIHyperLatents(ySymbols: y, uvSymbols: uv, height: height, width: width)
    }
}
