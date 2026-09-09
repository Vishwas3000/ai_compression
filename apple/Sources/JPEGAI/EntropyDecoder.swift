import CJPEGAIEntropy
import Foundation

public final class JPEGAIEntropyDecoder {
    private let handle: OpaquePointer

    public init(stream: Data) throws {
        let decoder = stream.withUnsafeBytes {
            jpegai_ans_decoder_create($0.bindMemory(to: UInt8.self).baseAddress, stream.count)
        }
        guard let decoder else { throw JPEGAIEntropyError.invalidStream }
        handle = decoder
    }

    deinit { jpegai_ans_decoder_destroy(handle) }

    public func decodeFactorized(
        cdfs: [UInt8], channels: Int, valuesPerChannel: Int
    ) throws -> [UInt8] {
        guard channels > 0, valuesPerChannel > 0, cdfs.count == channels * 63 else {
            throw JPEGAIEntropyError.invalidInput
        }
        var values = [UInt8](repeating: 0, count: channels * valuesPerChannel)
        let status = cdfs.withUnsafeBufferPointer { cdfs in
            values.withUnsafeMutableBufferPointer { values in
                jpegai_ans_decoder_decode_factorized(
                    handle, cdfs.baseAddress, values.baseAddress, channels, valuesPerChannel
                )
            }
        }
        guard status == 0 else { throw JPEGAIEntropyError.decodeFailed }
        return values
    }

    public func setSGMTables(transitions: [UInt32], bounds: [UInt8]) throws {
        guard !bounds.isEmpty, transitions.count == bounds.count * 256 else {
            throw JPEGAIEntropyError.invalidInput
        }
        let status = transitions.withUnsafeBufferPointer { transitions in
            bounds.withUnsafeBufferPointer { bounds in
                jpegai_ans_decoder_set_sgm_tables(
                    handle, transitions.baseAddress, bounds.baseAddress, bounds.count
                )
            }
        }
        guard status == 0 else { throw JPEGAIEntropyError.invalidInput }
    }

    public func decodeSGM(indexes: [UInt8], masks: [UInt8]) throws -> [Int16] {
        guard indexes.count == masks.count else { throw JPEGAIEntropyError.invalidInput }
        var values = [Int16](repeating: 0, count: indexes.count)
        let status = indexes.withUnsafeBufferPointer { indexes in
            masks.withUnsafeBufferPointer { masks in
                values.withUnsafeMutableBufferPointer { values in
                    jpegai_ans_decoder_decode_sgm(
                        handle, indexes.baseAddress, values.baseAddress, masks.baseAddress, indexes.count
                    )
                }
            }
        }
        guard status == 0 else { throw JPEGAIEntropyError.decodeFailed }
        return values
    }
}

public enum JPEGAIEntropyError: Error {
    case invalidStream
    case invalidInput
    case invalidTables
    case decodeFailed
}
