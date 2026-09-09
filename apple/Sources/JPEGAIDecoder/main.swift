import AppKit
import JPEGAI
import UniformTypeIdentifiers

@main
@MainActor
struct JPEGAIDecoderApp {
    private static let delegate = AppDelegate()

    static func main() {
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        application.delegate = delegate
        application.run()
    }
}

@MainActor
private final class AppDelegate: NSObject, NSApplicationDelegate {
    private let button = NSButton(title: "Choose JPEG AI File…", target: nil, action: nil)
    private let imageView = NSImageView()
    private let progress = NSProgressIndicator()
    private let status = NSTextField(
        wrappingLabelWithString: "Choose an untiled 4:4:4 simple-profile .bits file."
    )
    private var window: NSWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let title = NSTextField(labelWithString: "JPEG AI Decoder")
        title.font = .systemFont(ofSize: 26, weight: .semibold)
        title.alignment = .center

        let subtitle = NSTextField(
            wrappingLabelWithString: "Native entropy decoding and Core ML reconstruction on Apple silicon."
        )
        subtitle.alignment = .center
        subtitle.textColor = .secondaryLabelColor

        button.target = self
        button.action = #selector(chooseFile)
        button.bezelStyle = .rounded
        button.controlSize = .large
        button.setAccessibilityLabel("Choose a JPEG AI bitstream")

        imageView.imageScaling = .scaleProportionallyUpOrDown
        imageView.imageFrameStyle = .photo
        imageView.setAccessibilityLabel("Decoded image preview")

        progress.style = .spinning
        progress.isDisplayedWhenStopped = false
        status.alignment = .center
        status.textColor = .secondaryLabelColor

        let stack = NSStackView(views: [title, subtitle, imageView, button, progress, status])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false

        let content = NSView()
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: content.centerYAnchor),
            stack.leadingAnchor.constraint(greaterThanOrEqualTo: content.leadingAnchor, constant: 32),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: content.trailingAnchor, constant: -32),
            subtitle.widthAnchor.constraint(lessThanOrEqualToConstant: 440),
            imageView.widthAnchor.constraint(equalToConstant: 540),
            imageView.heightAnchor.constraint(equalToConstant: 400),
            status.widthAnchor.constraint(lessThanOrEqualToConstant: 540),
        ])

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 620, height: 650),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "JPEG AI Decoder"
        window.contentView = content
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.async { self.chooseFile() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    @objc private func chooseFile() {
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

    private func decode(_ input: URL, to output: URL) {
        guard let resources = Bundle.main.resourceURL else {
            show(error: "The application resources are missing.")
            return
        }
        let tables = resources.appendingPathComponent("Tables", isDirectory: true)
        let modelDirectory = resources.appendingPathComponent("apple-coreml-simple", isDirectory: true)
        button.isEnabled = false
        progress.startAnimation(nil)
        status.textColor = .secondaryLabelColor
        status.stringValue = "Decoding \(input.lastPathComponent)…"

        Task {
            do {
                try await Task.detached(priority: .userInitiated) {
                    let stream = try JPEGAIBitstream(contentsOf: input)
                    let image = try await stream.decodeImage(
                        tablesDirectory: tables,
                        models: JPEGAICoreMLModelSet(directory: modelDirectory)
                    )
                    try image.writePNG(to: output)
                }.value
                imageView.image = NSImage(contentsOf: output)
                status.stringValue = "Saved \(output.path)"
            } catch {
                let message = displayedMessage(for: error)
                status.textColor = .systemRed
                status.stringValue = message
                show(error: message)
            }
            progress.stopAnimation(nil)
            button.isEnabled = true
        }
    }

    private func show(error: String) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "JPEG AI decode failed"
        alert.informativeText = error
        alert.runModal()
    }

    private func displayedMessage(for error: Error) -> String {
        let cocoa = error as NSError
        guard cocoa.domain == NSCocoaErrorDomain,
              cocoa.code == CocoaError.Code.fileReadNoSuchFile.rawValue else {
            return error.localizedDescription
        }
        let path = cocoa.userInfo[NSFilePathErrorKey] as? String
        let name = path.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "a decoder resource"
        return "The bundled decoder resource \(name) is missing. Rebuild the app with all model tables."
    }
}
