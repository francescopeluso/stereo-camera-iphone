import AVFoundation
import CoreImage
import ImageIO
import Observation
import Photos
import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class Capture {
    private(set) var isRecording = false
    private(set) var recordingStart: Date?
    private(set) var message: String?

    @ObservationIgnored weak var model: AppModel?
    @ObservationIgnored private var photoPending = false
    @ObservationIgnored private var writer: AVAssetWriter?
    @ObservationIgnored private var input: AVAssetWriterInput?
    @ObservationIgnored private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    @ObservationIgnored private var t0: CFTimeInterval?
    @ObservationIgnored private var messageTask: Task<Void, Never>?

    func takePhoto() { photoPending = true }

    func toggleRecording() { isRecording ? stopRecording() : startRecording() }

    func target() -> (CVPixelBuffer, @MainActor (CVPixelBuffer) -> Void)? {
        guard let model else { return nil }
        if photoPending {
            photoPending = false
            guard let frame = model.pipeline.store.latest,
                  let pb = Self.buffer(1920, Self.even(1920 / model.aspect)) else { return nil }
            let snapshot = Snapshot(frame: frame, depth: frame.depthCopy(), analysis: model.analysis, meta: model.metadata())
            return (pb, { [weak self] in self?.finishPhoto($0, snapshot) })
        }
        guard isRecording, let adaptor, let input, input.isReadyForMoreMediaData, let pool = adaptor.pixelBufferPool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
        guard let pb else { return nil }
        let t = CACurrentMediaTime()
        return (pb, { [weak self] in self?.append($0, t) })
    }

    private func startRecording() {
        guard let model else { return }
        let w = 1280, h = Self.even(1280 / model.aspect)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("StereoVision-\(Self.stamp()).mp4")
        do {
            let wr = try AVAssetWriter(outputURL: url, fileType: .mp4)
            let inp = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: w, AVVideoHeightKey: h,
            ])
            inp.expectsMediaDataInRealTime = true
            let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp, sourcePixelBufferAttributes: Self.attributes(w, h))
            wr.add(inp)
            guard wr.startWriting() else { throw wr.error ?? CocoaError(.fileWriteUnknown) }
            wr.startSession(atSourceTime: .zero)
            writer = wr
            input = inp
            adaptor = ad
            t0 = nil
            isRecording = true
            recordingStart = .now
        } catch {
            flash("Recording failed: \(error.localizedDescription)")
        }
    }

    private func stopRecording() {
        guard let writer, let input else { return }
        isRecording = false
        recordingStart = nil
        self.writer = nil
        self.input = nil
        adaptor = nil
        input.markAsFinished()
        let held = Unchecked(writer)
        writer.finishWriting { [weak self] in
            let ok = held.value.status == .completed
            let url = held.value.outputURL
            Task { @MainActor in
                guard let self else { return }
                guard ok else { return self.flash("Recording failed") }
                await self.save { PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url) }
                try? FileManager.default.removeItem(at: url)
                self.flash("Video saved to Photos")
            }
        }
    }

    private func append(_ pb: CVPixelBuffer, _ t: CFTimeInterval) {
        guard isRecording, let adaptor, let input, input.isReadyForMoreMediaData else { return }
        drawOverlay(pb)
        let start = t0 ?? t
        t0 = start
        adaptor.append(pb, withPresentationTime: CMTime(seconds: t - start, preferredTimescale: 600))
    }

    private func finishPhoto(_ pb: CVPixelBuffer, _ snap: Snapshot) {
        drawOverlay(pb)
        guard let view = CIContext().createCGImage(CIImage(cvPixelBuffer: pb), from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))) else { return }
        let image = UIImage(cgImage: view)
        Task {
            await save { PHAssetChangeRequest.creationRequestForAsset(from: image) }
            let dir = await Task.detached { try? Self.export(snap, view) }.value
            flash(dir.map { "Saved to Photos + Files › StereoVision › \($0.lastPathComponent)" } ?? "Saved to Photos (raw export failed)")
        }
    }

    private func save(_ change: @escaping () -> Void) async {
        guard await PHPhotoLibrary.requestAuthorization(for: .addOnly) == .authorized else {
            return flash("Photo library access denied")
        }
        do {
            try await PHPhotoLibrary.shared().performChanges(change)
        } catch {
            flash("Save failed: \(error.localizedDescription)")
        }
    }

    private func drawOverlay(_ pb: CVPixelBuffer) {
        guard let model else { return }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else { return }
        let lw: CGFloat = 720, lh = lw * CGFloat(h) / CGFloat(w)
        ctx.scaleBy(x: CGFloat(w) / lw, y: CGFloat(h) / lh)
        ImageRenderer(content: OverlayView(model: model, hints: false).frame(width: lw, height: lh)).render { _, draw in draw(ctx) }
    }

    private func flash(_ text: String) {
        message = text
        messageTask?.cancel()
        messageTask = Task {
            try? await Task.sleep(for: .seconds(3))
            if !Task.isCancelled { message = nil }
        }
    }

    nonisolated private static func even(_ v: CGFloat) -> Int { Int((v / 2).rounded()) * 2 }

    nonisolated static func stamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f.string(from: .now)
    }

    nonisolated private static func attributes(_ w: Int, _ h: Int) -> [String: Any] {
        [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
    }

    nonisolated private static func buffer(_ w: Int, _ h: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attributes(w, h) as CFDictionary, &pb)
        return pb
    }
}

struct Snapshot: @unchecked Sendable {
    let frame: StereoFrame
    let depth: [Float]
    let analysis: Analysis
    let meta: [String: Any]
}

extension Capture {
    nonisolated static func export(_ s: Snapshot, _ view: CGImage) throws -> URL {
        let f = s.frame
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent(stamp(), isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        try png(view, dir.appendingPathComponent("view.png"))
        if let rgb = CIContext().createCGImage(CIImage(cvPixelBuffer: f.image), from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(f.image), height: CVPixelBufferGetHeight(f.image))) {
            try png(rgb, dir.appendingPathComponent("rgb.png"))
        }

        let mm = s.depth.map { UInt16(min(65535, max(0, $0 * 1000)).rounded()) }
        let depthData = mm.withUnsafeBytes { Data($0) }
        if let provider = CGDataProvider(data: depthData as CFData),
           let img = CGImage(width: f.width, height: f.height, bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: f.width * 2,
                             space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue),
                             provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) {
            try png(img, dir.appendingPathComponent("depth_mm.png"))
        }

        try ply(s, dir.appendingPathComponent("cloud.ply"))

        var meta = s.meta
        meta["depth"] = ["width": f.width, "height": f.height, "unit": "mm", "file": "depth_mm.png", "invalid": 0]
        meta["intrinsics"] = ["fx": f.K.fx, "fy": f.K.fy, "cx": f.K.cx, "cy": f.K.cy, "note": "pixel units of the depth map"]
        meta["baseline_mm"] = f.baselineMM
        meta["objects"] = s.analysis.objects.map { o -> [String: Any] in
            var d: [String: Any] = ["id": o.id, "label": o.label, "confidence": o.confidence,
                                    "bbox_norm": [o.rect.minX, o.rect.minY, o.rect.width, o.rect.height]]
            if let z = o.distance { d["distance_m"] = z; d["x_m"] = o.x; d["width_m"] = o.width; d["height_m"] = o.height }
            return d
        }
        meta["point_cloud"] = "cloud.ply, camera frame: x right, y down, z forward, meters"
        try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: dir.appendingPathComponent("meta.json"))
        return dir
    }

    private nonisolated static func png(_ img: CGImage, _ url: URL) throws {
        guard let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(dst, img, nil)
        guard CGImageDestinationFinalize(dst) else { throw CocoaError(.fileWriteUnknown) }
    }

    private nonisolated static func ply(_ s: Snapshot, _ url: URL) throws {
        let f = s.frame, pb = f.image
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pb), cw = CVPixelBufferGetWidth(pb), ch = CVPixelBufferGetHeight(pb)
        var body = ""
        body.reserveCapacity(s.depth.count * 40)
        var n = 0
        for y in 0..<f.height {
            for x in 0..<f.width {
                let z = s.depth[y * f.width + x]
                guard z > 0 else { continue }
                let X = (Float(x) + 0.5 - f.K.cx) * z / f.K.fx
                let Y = (Float(y) + 0.5 - f.K.cy) * z / f.K.fy
                let p = base + min(ch - 1, y * ch / f.height) * bpr + min(cw - 1, x * cw / f.width) * 4
                body += String(format: "%.4f %.4f %.4f %d %d %d\n", X, Y, z, p[2], p[1], p[0])
                n += 1
            }
        }
        let header = "ply\nformat ascii 1.0\nelement vertex \(n)\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        try (header + body).write(to: url, atomically: true, encoding: .ascii)
    }
}
