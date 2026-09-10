import Foundation
import Testing
@testable import JPEGAI

@Test func parsesMarkerDelimitedBitstream() throws {
    let data = Data([
        0xFF, 0x80,
        0xFF, 0x82, 0x20, 0x01, 0x02, 0x03,
        0xFF, 0x88, 0x80,
        0xFF, 0x81,
    ])

    let stream = try JPEGAIBitstream(data: data)
    #expect(stream.substreams.map(\.marker) == [.pictureHeader, .hyperLatent])
    #expect(stream.substreams.map(\.payload.count) == [3, 0])
}

@Test func rejectsTruncatedPayload() {
    let data = Data([0xFF, 0x80, 0xFF, 0x82, 0x20, 0x01])
    #expect(throws: JPEGAIBitstreamError.truncatedPayload) {
        try JPEGAIBitstream(data: data)
    }
}

@Test func parsesSimpleProfilePictureHeader() throws {
    let payload = Data([
        0x00, 0x00, 0x34, 0x01, 0xF0, 0x03, 0x38, 0x00,
        0x00, 0x09, 0x20, 0xEC, 0x50, 0x01, 0x80, 0x00,
    ])
    let header = try JPEGAIPictureHeader(payload: payload)

    #expect(header.decoderProfile == 0)
    #expect(header.synthesisTransforms == [0])
    #expect(header.codedWidth == 560)
    #expect(header.codedHeight == 888)
    #expect(header.model == 2)
    #expect(header.betaDisplacementY == 59)
    #expect(header.channelsY == 160)
    #expect(header.channelsUV == 96)
}

@Test func decodesOfficialFactorizedEntropyFixture() throws {
    let cdf: [UInt8] = [
        4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 49, 53, 57, 61, 65,
        69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
        130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178,
        182, 186, 190, 194, 198, 202, 206, 210, 215, 219, 223, 227, 231,
        235, 239, 243, 247, 251, 255,
    ]
    let stream = Data([0, 48, 11, 36, 93, 119, 0, 240, 26, 132, 0, 32])
    let decoder = try JPEGAIEntropyDecoder(stream: stream)

    #expect(try decoder.decodeFactorized(cdfs: cdf, channels: 1, valuesPerChannel: 13) ==
        [0, 1, 2, 3, 7, 15, 31, 62, 5, 12, 24, 48, 60])
}

@Test func encodesOfficialFactorizedEntropyFixture() throws {
    let cdf: [UInt8] = [
        4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 49, 53, 57, 61, 65,
        69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
        130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178,
        182, 186, 190, 194, 198, 202, 206, 210, 215, 219, 223, 227, 231,
        235, 239, 243, 247, 251, 255,
    ]
    let values: [UInt8] = [0, 1, 2, 3, 7, 15, 31, 62, 5, 12, 24, 48, 60]
    let expected = Data([0, 48, 11, 36, 93, 119, 0, 240, 26, 132, 0, 32])
    let encoder = try JPEGAIEntropyEncoder(capacity: 1024)

    try encoder.encodeFactorized(cdfs: cdf, values: values, channels: 1, valuesPerChannel: 13)
    #expect(try encoder.finish() == expected)
}

@Test func writesSimpleProfileBitstream() throws {
    let data = try JPEGAIBitstream.simpleProfileData(
        codedWidth: 560, codedHeight: 888,
        displayWidth: 560, displayHeight: 888,
        model: 2, beta: 59,
        primaryResidual: Data(), secondaryResidual: Data(), hyperLatent: Data()
    )
    let stream = try JPEGAIBitstream(data: data)
    let expectedHeader = Data([
        0x00, 0x00, 0x34, 0x01, 0xF0, 0x03, 0x38, 0x00,
        0x00, 0x09, 0x20, 0xEC, 0x50, 0x01, 0x80, 0x00,
    ])

    #expect(stream.substreams.map(\.marker) == [
        .pictureHeader, .toolHeader, .renderingInformation,
        .primaryResidual, .secondaryResidual, .hyperLatent,
    ])
    #expect(stream.substreams[0].payload == expectedHeader)
    #expect(try stream.pictureHeader.betaDisplacementY == 59)
}

@Test func sgmEncoderRoundTripsWithBundledTables() throws {
    let apple = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let models = apple.appendingPathComponent("Models")
    let required = [
        "residual_encode_transitions.csv", "residual_state_maps.csv",
        "residual_transitions.csv", "residual_bounds.csv",
    ]
    guard required.allSatisfy({
        FileManager.default.fileExists(atPath: models.appendingPathComponent($0).path)
    }) else { return }

    func values<T: FixedWidthInteger>(_ name: String, as type: T.Type) throws -> [T] {
        try readJPEGAICSV(models.appendingPathComponent(name)).flatMap { row in
            try row.map {
                guard let value = T(exactly: $0) else { throw JPEGAIEntropyError.invalidTables }
                return value
            }
        }
    }
    let indexes = (0 ..< 65).map { UInt8($0 % 32) }
    let input: [Int16] = (0 ..< 65).map {
        [0, 1, -1, 2, -2, 8, -8, 100, -100, 32767, -32768][$0 % 11]
    }
    let masks = (0 ..< 65).map { UInt8($0 % 7 == 0 ? 0 : 1) }
    let expected = zip(input, masks).map { $0.1 == 0 ? Int16(0) : $0.0 }
    let encoder = try JPEGAIEntropyEncoder(capacity: 4096)
    try encoder.setSGMTables(
        transitions: values("residual_encode_transitions.csv", as: UInt32.self),
        bounds: values("residual_bounds.csv", as: UInt8.self),
        stateMaps: values("residual_state_maps.csv", as: UInt8.self)
    )
    try encoder.encodeSGM(indexes: indexes, values: input, masks: masks)
    let decoder = try JPEGAIEntropyDecoder(stream: encoder.finish())
    try decoder.setSGMTables(
        transitions: values("residual_transitions.csv", as: UInt32.self),
        bounds: values("residual_bounds.csv", as: UInt8.self)
    )

    #expect(try decoder.decodeSGM(indexes: indexes, masks: masks) == expected)
}

@Test func decodesOfficialSGMEntropyFixture() throws {
    var transitions = [UInt32](repeating: 0, count: 256)
    transitions[0] = 510 << 16
    for state in 1 ..< 255 { transitions[state] = UInt32(state - 1) << 16 }
    transitions[255] = 8 << 24 | 0xFFFF
    let stream = Data([
        12, 208, 255, 79, 255, 223, 49, 128, 255, 128, 127, 99, 0, 15, 0,
        28, 0, 236, 255, 255, 255, 191, 38, 254, 255, 255, 255, 7, 8,
    ])
    let decoder = try JPEGAIEntropyDecoder(stream: stream)
    try decoder.setSGMTables(transitions: transitions, bounds: [1])

    let values = try decoder.decodeSGM(indexes: [UInt8](repeating: 0, count: 13),
                                       masks: [UInt8](repeating: 1, count: 13))
    #expect(values == [0, 1, -1, 2, -2, 8, -8, 100, -100, 32766, -32767, 0, 3])
}

@Test func spatialShuffleRoundTripsOddDimensions() {
    let values = (0 ..< 30).map(Float32.init)
    let parts = JPEGAIBitstream.downShuffle(values, channels: 2, height: 3, width: 5)

    #expect(JPEGAIBitstream.downShuffle([0, 1, 2, 3], channels: 1, height: 2, width: 2) ==
        [[0], [3], [1], [2]])
    #expect(parts.map(\.count) == [12, 12, 12, 12])
    #expect(JPEGAIBitstream.upShuffle(parts, channels: 2, height: 3, width: 5) == values)
}
