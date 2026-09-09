#if canImport(CoreML)
import CoreML
import Foundation

@available(macOS 13, iOS 16, *)
public final class JPEGAICoreMLModelSet {
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
}

public enum JPEGAICoreMLError: Error {
    case missingModel(URL)
    case missingOutput
    case invalidShape
}
#endif
