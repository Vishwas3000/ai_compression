#if canImport(CoreML)
import CoreML
import Foundation

@available(macOS 13, iOS 16, *)
public actor JPEGAICoreMLModelSet {
    private let directory: URL
    private let configuration: MLModelConfiguration
    private var loaded: [URL: MLModel] = [:]

    public init(directory: URL) {
        self.directory = directory
        configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
    }

    public func predict(
        tool: Int, component: String, path: String, inputs: [MLMultiArray]
    ) async throws -> MLMultiArray {
        let url = directory
            .appendingPathComponent("tools_\(tool)")
            .appendingPathComponent(component)
            .appendingPathComponent(path)
            .appendingPathExtension("mlpackage")
        let model: MLModel
        if let existing = loaded[url] {
            model = existing
        } else {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw JPEGAICoreMLError.missingModel(url)
            }
            let compiled = try await MLModel.compileModel(at: url)
            model = try MLModel(contentsOf: compiled, configuration: configuration)
            loaded[url] = model
        }
        let values = Dictionary(uniqueKeysWithValues: inputs.enumerated().map {
            ("input_\($0.offset)", MLFeatureValue(multiArray: $0.element))
        })
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: values))
        guard let name = model.modelDescription.outputDescriptionsByName.keys.first,
              let output = result.featureValue(for: name)?.multiArrayValue else {
            throw JPEGAICoreMLError.missingOutput
        }
        return output
    }

    public static func floatArray(
        _ values: [Int8], channels: Int, height: Int, width: Int
    ) throws -> MLMultiArray {
        guard values.count == channels * height * width else {
            throw JPEGAICoreMLError.invalidShape
        }
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: values.count)
        for (index, value) in values.enumerated() { pointer[index] = Float32(value) }
        return array
    }

    public static func floatArray(
        _ values: [Float32], channels: Int, height: Int, width: Int
    ) throws -> MLMultiArray {
        guard values.count == channels * height * width else {
            throw JPEGAICoreMLError.invalidShape
        }
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        values.withUnsafeBufferPointer {
            array.dataPointer.copyMemory(
                from: $0.baseAddress!, byteCount: values.count * MemoryLayout<Float32>.stride
            )
        }
        return array
    }

    public static func int32Array(
        _ values: [Int8], channels: Int, height: Int, width: Int
    ) throws -> MLMultiArray {
        guard values.count == channels * height * width else {
            throw JPEGAICoreMLError.invalidShape
        }
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)],
            dataType: .int32
        )
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for (index, value) in values.enumerated() { pointer[index] = Int32(value) }
        return array
    }

    public static func int32Values(
        _ array: MLMultiArray, channels: Int, height: Int, width: Int
    ) throws -> [Int32] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard array.dataType == .int32, shape.count == 4, shape[0] == 1,
              shape[1] == channels, shape[2] >= height, shape[3] >= width else {
            throw JPEGAICoreMLError.invalidShape
        }
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        var values = [Int32]()
        values.reserveCapacity(channels * height * width)
        for channel in 0 ..< channels {
            for row in 0 ..< height {
                for column in 0 ..< width {
                    values.append(pointer[channel * strides[1] + row * strides[2] + column * strides[3]])
                }
            }
        }
        return values
    }

    public static func float32Values(
        _ array: MLMultiArray, channels: Int, height: Int, width: Int
    ) throws -> [Float32] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard array.dataType == .float32, shape.count == 4, shape[0] == 1,
              shape[1] == channels, shape[2] >= height, shape[3] >= width else {
            throw JPEGAICoreMLError.invalidShape
        }
        let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        var values = [Float32]()
        values.reserveCapacity(channels * height * width)
        for channel in 0 ..< channels {
            for row in 0 ..< height {
                for column in 0 ..< width {
                    values.append(pointer[channel * strides[1] + row * strides[2] + column * strides[3]])
                }
            }
        }
        return values
    }
}

public enum JPEGAICoreMLError: Error {
    case missingModel(URL)
    case missingOutput
    case invalidShape
}
#endif
