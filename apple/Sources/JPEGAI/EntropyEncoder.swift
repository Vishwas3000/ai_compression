import CJPEGAIEntropy
import Foundation

public final class JPEGAIEntropyEncoder {
    private let handle: OpaquePointer
    private let capacity: Int

    public init(capacity: Int) throws {
        guard capacity > 0, let encoder = jpegai_ans_encoder_create(capacity) else {
            throw JPEGAIEntropyError.invalidInput
        }
        self.capacity = capacity
        handle = encoder
    }

    deinit { jpegai_ans_encoder_destroy(handle) }

    public func setSGMTables(
        transitions: [UInt32], bounds: [UInt8], stateMaps: [UInt8]
    ) throws {
        guard !bounds.isEmpty,
              transitions.count == bounds.count * 256,
              stateMaps.count == transitions.count else {
            throw JPEGAIEntropyError.invalidInput
        }
        let status = transitions.withUnsafeBufferPointer { transitions in
            bounds.withUnsafeBufferPointer { bounds in
                stateMaps.withUnsafeBufferPointer { stateMaps in
                    jpegai_ans_encoder_set_sgm_tables(
                        handle, transitions.baseAddress, bounds.baseAddress,
                        stateMaps.baseAddress, bounds.count
                    )
                }
            }
        }
        guard status == 0 else { throw JPEGAIEntropyError.invalidTables }
    }

    public func encodeSGM(
        indexes: [UInt8], values: [Int16], masks: [UInt8]
    ) throws {
        guard indexes.count == values.count, values.count == masks.count else {
            throw JPEGAIEntropyError.invalidInput
        }
        var mutableValues = values
        let status = indexes.withUnsafeBufferPointer { indexes in
            mutableValues.withUnsafeMutableBufferPointer { values in
                masks.withUnsafeBufferPointer { masks in
                    jpegai_ans_encoder_encode_sgm(
                        handle, indexes.baseAddress, values.baseAddress,
                        masks.baseAddress, indexes.count
                    )
                }
            }
        }
        guard status == 0 else { throw JPEGAIEntropyError.encodeFailed }
    }

    public func encodeFactorized(
        cdfs: [UInt8], values: [UInt8], channels: Int, valuesPerChannel: Int
    ) throws {
        guard channels > 0, valuesPerChannel > 0,
              cdfs.count == channels * 63,
              values.count == channels * valuesPerChannel else {
            throw JPEGAIEntropyError.invalidInput
        }
        var mutableValues = values
        let status = cdfs.withUnsafeBufferPointer { cdfs in
            mutableValues.withUnsafeMutableBufferPointer { values in
                jpegai_ans_encoder_encode_factorized(
                    handle, cdfs.baseAddress, values.baseAddress,
                    channels, valuesPerChannel
                )
            }
        }
        guard status == 0 else { throw JPEGAIEntropyError.encodeFailed }
    }

    public func finish() throws -> Data {
        var output = [UInt8](repeating: 0, count: capacity)
        let size = output.withUnsafeMutableBufferPointer {
            jpegai_ans_encoder_finish(handle, $0.baseAddress, capacity)
        }
        guard size >= 0 else { throw JPEGAIEntropyError.encodeFailed }
        return Data(output.prefix(Int(size)))
    }
}
