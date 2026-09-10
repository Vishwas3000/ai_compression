import AppKit
import JPEGAI
import UniformTypeIdentifiers

@MainActor
private final class AppDelegate: NSObject, NSApplicationDelegate {
    private let encodeButton = NSButton(title: "Encode PNG…", target: nil, action: nil)
    private let decodeButton = NSButton(title: "Decode .bits…", target: nil, action: nil)
    private let presetPicker = NSPopUpButton()
    private let diagnosticsButton = NSButton(
        checkboxWithTitle: "Export y/z inference visualizations", target: nil, action: nil
    )
    private let revealDiagnosticsButton = NSButton(
        title: "View Inference Steps", target: nil, action: nil
    )
    private let imageView = NSImageView()
    private let progress = NSProgressIndicator()
    private let status = NSTextField(
        wrappingLabelWithString: "Encode a PNG or decode an untiled 4:4:4 simple-profile JPEG AI file."
    )
    private var window: NSWindow!
    private var codec: BenchmarkCodec?
    private var diagnosticsDirectory: URL?
    private var inferenceWindow: InferenceWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let title = NSTextField(labelWithString: "JPEG AI Codec Lab")
        title.font = .systemFont(ofSize: 26, weight: .semibold)
        title.alignment = .center

        let subtitle = NSTextField(
            wrappingLabelWithString: "Native JPEG AI encoding, entropy decoding, and Core ML reconstruction on Apple silicon."
        )
        subtitle.alignment = .center
        subtitle.textColor = .secondaryLabelColor

        encodeButton.target = self
        encodeButton.action = #selector(chooseImage)
        encodeButton.bezelStyle = .rounded
        encodeButton.controlSize = .large
        encodeButton.setAccessibilityLabel("Encode a PNG as JPEG AI")

        decodeButton.target = self
        decodeButton.action = #selector(chooseBitstream)
        decodeButton.bezelStyle = .rounded
        decodeButton.controlSize = .large
        decodeButton.setAccessibilityLabel("Decode a JPEG AI bitstream")

        presetPicker.addItems(withTitles: RatePreset.all.map(\.title))
        presetPicker.selectItem(at: 3)
        presetPicker.setAccessibilityLabel("JPEG AI nominal rate preset")
        diagnosticsButton.state = .on
        revealDiagnosticsButton.target = self
        revealDiagnosticsButton.action = #selector(revealDiagnostics)
        revealDiagnosticsButton.isEnabled = false

        imageView.imageScaling = .scaleProportionallyUpOrDown
        imageView.imageFrameStyle = .photo
        imageView.setAccessibilityLabel("JPEG AI decoded image preview")

        progress.style = .spinning
        progress.isDisplayedWhenStopped = false
        status.alignment = .center
        status.textColor = .secondaryLabelColor
        status.maximumNumberOfLines = 5

        let encodeRow = NSStackView(views: [encodeButton, presetPicker])
        encodeRow.orientation = .horizontal
        encodeRow.alignment = .centerY
        encodeRow.spacing = 10

        let stack = NSStackView(views: [
            title, subtitle, imageView, encodeRow, diagnosticsButton,
            revealDiagnosticsButton, decodeButton, progress, status,
        ])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView()
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: content.centerYAnchor),
            stack.leadingAnchor.constraint(greaterThanOrEqualTo: content.leadingAnchor, constant: 32),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: content.trailingAnchor, constant: -32),
            subtitle.widthAnchor.constraint(lessThanOrEqualToConstant: 520),
            imageView.widthAnchor.constraint(equalToConstant: 540),
            imageView.heightAnchor.constraint(equalToConstant: 380),
            status.widthAnchor.constraint(lessThanOrEqualToConstant: 560),
        ])

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 640, height: 700),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "JPEG AI Codec Lab"
        window.contentView = content
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    @objc private func chooseImage() {
        let panel = NSOpenPanel()
        panel.title = "Choose a PNG to encode"
        panel.allowedContentTypes = [.png]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let input = panel.url else { return }

        let save = NSSavePanel()
        save.title = "Save JPEG AI bitstream"
        save.allowedContentTypes = [UTType(filenameExtension: "bits") ?? .data]
        save.directoryURL = input.deletingLastPathComponent()
        save.nameFieldStringValue = input.deletingPathExtension().lastPathComponent + ".bits"
        guard save.runModal() == .OK, let output = save.url else { return }

        let preset = RatePreset.all[presetPicker.indexOfSelectedItem]
        let preview = Self.availableSibling(
            of: output, named: output.deletingPathExtension().lastPathComponent + ".decoded.png"
        )
        let diagnostics = diagnosticsButton.state == .on ? Self.availableSibling(
            of: output,
            named: output.deletingPathExtension().lastPathComponent + ".visualizations"
        ) : nil
        encode(input, to: output, preview: preview, diagnostics: diagnostics, preset: preset)
    }

    @objc private func chooseBitstream() {
        let panel = NSOpenPanel()
        panel.title = "Choose a JPEG AI bitstream"
        panel.allowedContentTypes = [UTType(filenameExtension: "bits") ?? .data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let input = panel.url else { return }

        let save = NSSavePanel()
        save.title = "Save decoded image"
        save.allowedContentTypes = [.png]
        save.directoryURL = input.deletingLastPathComponent()
        save.nameFieldStringValue = input.deletingPathExtension().lastPathComponent + ".decoded.png"
        guard save.runModal() == .OK, let output = save.url else { return }
        decode(input, to: output)
    }

    @objc private func revealDiagnostics() {
        guard let diagnosticsDirectory else { return }
        let controller = InferenceWindowController(directory: diagnosticsDirectory)
        inferenceWindow = controller
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }

    private func encode(
        _ input: URL, to output: URL, preview: URL,
        diagnostics: URL?, preset: RatePreset
    ) {
        guard let codec = sharedCodec() else { return }
        setBusy(true, message: "Encoding \(input.lastPathComponent)…")
        Task {
            do {
                let measurement = try await codec.encode(
                    input, to: output, preview: preview,
                    diagnostics: diagnostics, preset: preset,
                    progress: { stage in
                        await MainActor.run {
                            self.status.stringValue = stage.rawValue
                        }
                    }
                )
                imageView.image = NSImage(contentsOf: preview)
                let kind = measurement.firstRun ? "First" : "Repeat"
                var message = String(
                    format: "%@ encode: %.3f s • %.3f bpp • %d bytes\nVerification decode: %.3f s\nSaved %@",
                    kind, measurement.encodeSeconds, measurement.bitsPerPixel,
                    measurement.bytes, measurement.decodeSeconds, output.path
                )
                if let diagnostics {
                    diagnosticsDirectory = diagnostics
                    revealDiagnosticsButton.isEnabled = true
                    message += "\nVisualizations: \(diagnostics.path)"
                }
                status.stringValue = message
            } catch {
                report(error, operation: "encode")
            }
            setBusy(false)
        }
    }

    private func decode(_ input: URL, to output: URL) {
        guard let codec = sharedCodec() else { return }
        setBusy(true, message: "Decoding \(input.lastPathComponent)…")
        Task {
            do {
                let measurement = try await codec.decode(input, to: output)
                imageView.image = NSImage(contentsOf: output)
                let kind = measurement.firstRun ? "First" : "Repeat"
                status.stringValue = String(
                    format: "%@ decode: %.3f s (%d×%d)\nSaved %@",
                    kind, measurement.seconds, measurement.width, measurement.height, output.path
                )
            } catch {
                report(error, operation: "decode")
            }
            setBusy(false)
        }
    }

    private func sharedCodec() -> BenchmarkCodec? {
        if let codec { return codec }
        guard let resources = Bundle.main.resourceURL else {
            reportMessage("The application resources are missing.", operation: "start")
            return nil
        }
        let bundledTables = resources.appendingPathComponent("Tables", isDirectory: true)
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let localModels = packageRoot.appendingPathComponent("Models", isDirectory: true)
        let useBundle = FileManager.default.fileExists(
            atPath: bundledTables.appendingPathComponent("unique_z_distributions.csv").path
        )
        let codec = BenchmarkCodec(
            tables: useBundle ? bundledTables : localModels,
            modelDirectory: useBundle
                ? resources.appendingPathComponent("apple-coreml-simple", isDirectory: true)
                : localModels.appendingPathComponent("apple-coreml-simple", isDirectory: true)
        )
        self.codec = codec
        return codec
    }

    private func setBusy(_ busy: Bool, message: String? = nil) {
        encodeButton.isEnabled = !busy
        decodeButton.isEnabled = !busy
        presetPicker.isEnabled = !busy
        diagnosticsButton.isEnabled = !busy
        revealDiagnosticsButton.isEnabled = !busy && diagnosticsDirectory != nil
        if busy { progress.startAnimation(nil) } else { progress.stopAnimation(nil) }
        if let message {
            status.textColor = .secondaryLabelColor
            status.stringValue = message
        }
    }

    private func report(_ error: Error, operation: String) {
        let cocoa = error as NSError
        let message: String
        if cocoa.domain == NSCocoaErrorDomain,
           cocoa.code == CocoaError.Code.fileReadNoSuchFile.rawValue {
            let path = cocoa.userInfo[NSFilePathErrorKey] as? String
            let name = path.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "a codec resource"
            message = "The bundled codec resource \(name) is missing. Rebuild the app with all model tables."
        } else {
            message = error.localizedDescription
        }
        reportMessage(message, operation: operation)
    }

    private func reportMessage(_ message: String, operation: String) {
        status.textColor = .systemRed
        status.stringValue = message
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "JPEG AI \(operation) failed"
        alert.informativeText = message
        alert.runModal()
    }

    private static func availableSibling(of url: URL, named name: String) -> URL {
        let directory = url.deletingLastPathComponent()
        let proposed = directory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: proposed.path) else { return proposed }
        let stem = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        var suffix = 2
        while true {
            let candidateName = ext.isEmpty ? "\(name)-\(suffix)" : "\(stem)-\(suffix).\(ext)"
            let candidate = directory.appendingPathComponent(candidateName)
            if !FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            suffix += 1
        }
    }
}

@MainActor
private final class InferenceWindowController: NSWindowController {
    private struct Step {
        let title: String
        let filename: String
        let explanation: String
    }

    private static let definitions = [
        Step(
            title: "1. Input luma", filename: "01-input-luma.png",
            explanation: "The PNG has been converted to BT.709 YUV. This luminance plane enters the learned analysis transform."
        ),
        Step(
            title: "2. Learned y channels", filename: "02-y-latent-160-channels.png",
            explanation: "The analysis network compressed the luma plane into 160 spatial feature channels. Orange is positive and blue is negative; each tile is normalized independently."
        ),
        Step(
            title: "3. y activation energy", filename: "03-y-latent-energy.png",
            explanation: "Mean absolute activation across the 160 y channels. It shows where the primary latent spends representation capacity."
        ),
        Step(
            title: "4. Quantized z channels", filename: "04-z-hyperlatent-160-channels.png",
            explanation: "The hyper-encoder reduced y to a coarser z space and rounded it to symbols. z describes how y should be predicted and entropy-coded."
        ),
        Step(
            title: "5. z activation energy", filename: "05-z-hyperlatent-energy.png",
            explanation: "Mean absolute activation across z. Its coarse grid carries coding context, not a miniature reconstruction."
        ),
        Step(
            title: "6. Entropy-mask density", filename: "06-entropy-mask-density.png",
            explanation: "The integer hyper-scale decoder used z to choose probability distributions and decide which y values are coded. Brighter areas code a larger fraction of channels."
        ),
        Step(
            title: "7. Quantized y residual", filename: "07-y-quantized-residual-energy.png",
            explanation: "After four-stage context prediction, only the quantized prediction residual is sent to me-tANS. Bright regions needed larger corrections."
        ),
    ]

    private let directory: URL
    private let steps: [Step]
    private let picker = NSPopUpButton()
    private let imageView = NSImageView()
    private let explanation = NSTextField(wrappingLabelWithString: "")
    private let previousButton = NSButton(title: "Previous", target: nil, action: nil)
    private let nextButton = NSButton(title: "Next", target: nil, action: nil)

    init(directory: URL) {
        self.directory = directory
        steps = Self.definitions.filter {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0.filename).path)
        }
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 920, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false
        )
        super.init(window: window)
        window.title = "JPEG AI Inference Steps"

        let title = NSTextField(labelWithString: "Runtime inference explorer")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        let flow = NSTextField(
            labelWithString: "Pixels → y → z → probability model → y residual → me-tANS → bitstream"
        )
        flow.textColor = .secondaryLabelColor

        picker.addItems(withTitles: steps.map(\.title))
        picker.target = self
        picker.action = #selector(selectStep)
        picker.setAccessibilityLabel("Inference visualization step")

        previousButton.target = self
        previousButton.action = #selector(previousStep)
        nextButton.target = self
        nextButton.action = #selector(nextStep)
        let reveal = NSButton(title: "Reveal Files in Finder", target: self, action: #selector(revealFiles))
        let navigation = NSStackView(views: [previousButton, picker, nextButton, reveal])
        navigation.orientation = .horizontal
        navigation.alignment = .centerY
        navigation.spacing = 10

        explanation.alignment = .center
        explanation.textColor = .secondaryLabelColor
        explanation.maximumNumberOfLines = 4
        imageView.imageScaling = .scaleProportionallyUpOrDown
        imageView.imageFrameStyle = .photo

        let stack = NSStackView(views: [title, flow, navigation, explanation, imageView])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        let content = NSView()
        content.addSubview(stack)
        window.contentView = content
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 24),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -24),
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 30),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -30),
            explanation.widthAnchor.constraint(equalTo: stack.widthAnchor),
            imageView.widthAnchor.constraint(equalTo: stack.widthAnchor),
            imageView.heightAnchor.constraint(greaterThanOrEqualToConstant: 500),
        ])
        renderStep()
        window.center()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is unavailable") }

    @objc private func selectStep() { renderStep() }

    @objc private func previousStep() {
        picker.selectItem(at: max(0, picker.indexOfSelectedItem - 1))
        renderStep()
    }

    @objc private func nextStep() {
        picker.selectItem(at: min(steps.count - 1, picker.indexOfSelectedItem + 1))
        renderStep()
    }

    @objc private func revealFiles() { NSWorkspace.shared.open(directory) }

    private func renderStep() {
        guard !steps.isEmpty else {
            explanation.stringValue = "No inference images were found. Encode with visualization export enabled."
            previousButton.isEnabled = false
            nextButton.isEnabled = false
            return
        }
        let index = max(0, picker.indexOfSelectedItem)
        let step = steps[index]
        imageView.image = NSImage(contentsOf: directory.appendingPathComponent(step.filename))
        imageView.setAccessibilityLabel(step.title)
        explanation.stringValue = step.explanation
        previousButton.isEnabled = index > 0
        nextButton.isEnabled = index + 1 < steps.count
    }
}

let application = NSApplication.shared
private let applicationDelegate = AppDelegate()
application.setActivationPolicy(.regular)
application.delegate = applicationDelegate
application.run()

private struct RatePreset: Sendable {
    let title: String
    let model: Int
    let beta: Int

    static let all = [
        Self(title: "Point 12", model: 0, beta: 0),
        Self(title: "Point 25", model: 1, beta: 0),
        Self(title: "Point 50", model: 2, beta: -184),
        Self(title: "Point 75", model: 2, beta: 0),
        Self(title: "Point 100", model: 3, beta: 0),
    ]
}

private struct DecodeMeasurement: Sendable {
    let seconds: Double
    let width: Int
    let height: Int
    let firstRun: Bool
}

private struct EncodeMeasurement: Sendable {
    let encodeSeconds: Double
    let decodeSeconds: Double
    let bytes: Int
    let bitsPerPixel: Double
    let firstRun: Bool
}

private actor BenchmarkCodec {
    private let tables: URL
    private let models: JPEGAICoreMLModelSet
    private var encodedModels = Set<Int>()
    private var decodedModels = Set<Int>()

    init(tables: URL, modelDirectory: URL) {
        self.tables = tables
        models = JPEGAICoreMLModelSet(directory: modelDirectory)
    }

    func decode(_ input: URL, to output: URL) async throws -> DecodeMeasurement {
        let stream = try JPEGAIBitstream(contentsOf: input)
        let model = try stream.pictureHeader.model
        let started = ContinuousClock.now
        let image = try await stream.decodeImage(tablesDirectory: tables, models: models)
        let seconds = Self.seconds(started.duration(to: .now))
        try image.writePNG(to: output)
        return DecodeMeasurement(
            seconds: seconds, width: image.width, height: image.height,
            firstRun: decodedModels.insert(model).inserted
        )
    }

    func encode(
        _ input: URL, to output: URL, preview: URL,
        diagnostics: URL?, preset: RatePreset,
        progress: @escaping @Sendable (JPEGAIEncodingStage) async -> Void
    ) async throws -> EncodeMeasurement {
        let image = try JPEGAIDecodedImage.readPNG(from: input)
        let started = ContinuousClock.now
        let encoded = try await image.encodeJPEGAI(
            tablesDirectory: tables, models: models,
            model: preset.model, beta: preset.beta,
            includeDiagnostics: diagnostics != nil,
            progress: progress
        )
        let encodeSeconds = Self.seconds(started.duration(to: .now))
        try encoded.write(to: output)
        if let diagnostics {
            await progress(.diagnostics)
            try encoded.diagnostics?.write(to: diagnostics)
        }

        await progress(.verificationDecode)
        let stream = try JPEGAIBitstream(data: encoded.data)
        let decodeStarted = ContinuousClock.now
        let reconstructed = try await stream.decodeImage(tablesDirectory: tables, models: models)
        let decodeSeconds = Self.seconds(decodeStarted.duration(to: .now))
        try reconstructed.writePNG(to: preview)
        decodedModels.insert(preset.model)
        return EncodeMeasurement(
            encodeSeconds: encodeSeconds,
            decodeSeconds: decodeSeconds,
            bytes: encoded.data.count,
            bitsPerPixel: Double(encoded.data.count * 8) / Double(image.width * image.height),
            firstRun: encodedModels.insert(preset.model).inserted
        )
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
