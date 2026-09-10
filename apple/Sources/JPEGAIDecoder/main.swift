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
        title: "Reveal Visualizations", target: nil, action: nil
    )
    private let imageView = NSImageView()
    private let progress = NSProgressIndicator()
    private let status = NSTextField(
        wrappingLabelWithString: "Encode a PNG or decode an untiled 4:4:4 simple-profile JPEG AI file."
    )
    private var window: NSWindow!
    private var codec: BenchmarkCodec?
    private var diagnosticsDirectory: URL?

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
        NSWorkspace.shared.open(diagnosticsDirectory)
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
                    diagnostics: diagnostics, preset: preset
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
        diagnostics: URL?, preset: RatePreset
    ) async throws -> EncodeMeasurement {
        let image = try JPEGAIDecodedImage.readPNG(from: input)
        let started = ContinuousClock.now
        let encoded = try await image.encodeJPEGAI(
            tablesDirectory: tables, models: models,
            model: preset.model, beta: preset.beta,
            includeDiagnostics: diagnostics != nil
        )
        let encodeSeconds = Self.seconds(started.duration(to: .now))
        try encoded.write(to: output)
        if let diagnostics { try encoded.diagnostics?.write(to: diagnostics) }

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
