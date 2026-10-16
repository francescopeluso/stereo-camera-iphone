import AVFoundation
import Metal
import Observation
import QuartzCore
import UIKit
import os

enum ViewMode: Int, CaseIterable, Identifiable {
    case standard, depth, overlay, disparity, stereoPair, anaglyph, segmentation, radar, pointCloud
    var id: Self { self }

    var title: String {
        switch self {
        case .standard: "Standard"
        case .depth: "Depth map"
        case .overlay: "Overlay"
        case .disparity: "Disparity"
        case .stereoPair: "Rectified pair"
        case .anaglyph: "Anaglyph"
        case .segmentation: "3D segmentation"
        case .radar: "Bird's-eye"
        case .pointCloud: "Point cloud"
        }
    }

    var icon: String {
        switch self {
        case .standard: "camera"
        case .depth: "flame"
        case .overlay: "square.2.layers.3d"
        case .disparity: "circle.lefthalf.filled"
        case .stereoPair: "rectangle.split.2x1"
        case .anaglyph: "eyeglasses"
        case .segmentation: "square.grid.3x3.fill"
        case .radar: "scope"
        case .pointCloud: "cube.transparent"
        }
    }

    var needsStereoPair: Bool { self == .stereoPair || self == .anaglyph }
    var shaderIndex: Int32 { Int32(min(rawValue, 7)) }
}

enum Matcher: String, CaseIterable, Identifiable, Sendable {
    case sgm, block
    var id: Self { self }
    var title: String { self == .sgm ? "SGM 4 paths" : "Block matching 5×5" }
}

struct PipelineSettings: Sendable {
    var matcher = Matcher.sgm
    var lrCheck = true
    var p1 = 8
    var p2 = 96
    var minZ: Float = 0.12
    var maxZ: Float = 3
    var uniqueness: Float = 0.1
    var maxDisparity = 64
    var yOffset: Float = 0
    var flip = false
    var detection = true
    var confidence: Float = 0.4
    var calibrate = true
}

struct FrameReport: Sendable {
    var analysis: Analysis
    var fps: Double
    var gpuMs: Double
    var baselineMM: Float
    var K: Intrinsics
    var depthSize: CGSize
    var aspect: CGFloat
    var calibration: (flip: Bool, dy: Float)?
}

final class FrameStore: @unchecked Sendable {
    private let lock = NSLock()
    private var frame: StereoFrame?
    var latest: StereoFrame? {
        get { lock.withLock { frame } }
        set { lock.withLock { frame = newValue } }
    }
}

final class Pipeline: @unchecked Sendable {
    let engine: StereoEngine
    let detector = Detector()
    let store = FrameStore()
    let settings = OSAllocatedUnfairLock(initialState: PipelineSettings())
    var publish: ((FrameReport) -> Void)?
    private let analyzer = Analyzer()
    private var stamps: [Double] = []

    init(engine: StereoEngine) { self.engine = engine }

    func stereo(_ wide: CMSampleBuffer, _ ultra: CMSampleBuffer) {
        var s = settings.withLock { st in
            defer { st.calibrate = false }
            return st
        }
        var cal: (flip: Bool, dy: Float)?
        if s.calibrate {
            let c = engine.calibrate(wide, ultra, s)
            s.flip = c.flip
            s.yOffset = c.dy
            settings.withLock { $0.flip = c.flip; $0.yOffset = c.dy }
            cal = c
        }
        guard let f = engine.stereo(wide, ultra, s) else { return }
        finish(f, s, cal)
    }

    func depthData(_ video: CMSampleBuffer, _ depth: AVDepthData) {
        let s = settings.withLock { $0 }
        guard let f = engine.depthData(video, depth, s) else { return }
        finish(f, s, nil)
    }

    private func finish(_ f: StereoFrame, _ s: PipelineSettings, _ cal: (flip: Bool, dy: Float)?) {
        store.latest = f
        if s.detection { detector.submit(f.image, minConfidence: s.confidence) }
        let det = s.detection ? detector.latest : (seq: -1, items: [])
        let now = CACurrentMediaTime()
        stamps.append(now)
        stamps.removeAll { now - $0 > 1 }
        let a = analyzer.analyze(f, detections: det.items, seq: det.seq, now: now)
        publish?(FrameReport(analysis: a, fps: Double(stamps.count), gpuMs: f.gpuMs, baselineMM: f.baselineMM, K: f.K,
                             depthSize: CGSize(width: f.width, height: f.height),
                             aspect: CGFloat(f.color.width) / CGFloat(f.color.height), calibration: cal))
    }
}

@MainActor
@Observable
final class AppModel {
    private(set) var source: DepthSource = .stereo
    var mode: ViewMode = .standard
    var minZ: Float = 0.12 { didSet { push { $0.minZ = self.minZ } } }
    var maxZ: Float = 3 { didSet { push { $0.maxZ = self.maxZ } } }
    var uniqueness: Float = 0.1 { didSet { push { $0.uniqueness = self.uniqueness } } }
    var maxDisparity = 64 { didSet { push { $0.maxDisparity = self.maxDisparity } } }
    var matcher = Matcher.sgm { didSet { push { $0.matcher = self.matcher } } }
    var lrCheck = true { didSet { push { $0.lrCheck = self.lrCheck } } }
    var p1: Float = 8 { didSet { push { $0.p1 = Int(self.p1) } } }
    var p2: Float = 96 { didSet { push { $0.p2 = Int(self.p2) } } }
    var showMasks = true
    var yOffset: Float = 0 { didSet { push { $0.yOffset = self.yOffset } } }
    var flip = false { didSet { push { $0.flip = self.flip } } }
    var detection = true { didSet { push { $0.detection = self.detection } } }
    var confidence: Float = 0.4 { didSet { push { $0.confidence = self.confidence } } }
    var overlayAlpha: Float = 0.55
    var yaw: Float = 0
    var pitch: Float = 0
    var orbit: Float = 1.2
    var calibrating = false

    private(set) var analysis = Analysis()
    private(set) var fps = 0.0
    private(set) var gpuMs = 0.0
    private(set) var baselineMM: Float = 0
    private(set) var K = Intrinsics(fx: 1, fy: 1, cx: 0, cy: 0)
    private(set) var depthSize = CGSize(width: 1, height: 1)
    private(set) var aspect: CGFloat = 16 / 9
    private(set) var status: String?
    private(set) var tap: CGPoint?
    private(set) var tapDistance: Float?

    @ObservationIgnored let pipeline: Pipeline
    @ObservationIgnored let renderer: Renderer
    let capture = Capture()
    @ObservationIgnored private let camera = CameraController()

    var detectorName: String { pipeline.detector.name }

    init() {
        let device = MTLCreateSystemDefaultDevice()!
        let engine = StereoEngine(device: device)
        pipeline = Pipeline(engine: engine)
        renderer = Renderer(device: device, store: pipeline.store)
        renderer.model = self
        capture.model = self

        let p = pipeline
        camera.onStereo = { p.stereo($0, $1) }
        camera.onDepth = { p.depthData($0, $1) }
        camera.onDevices = { engine.configure(wide: $0, ultra: $1) }
        camera.onError = { [weak self] msg in DispatchQueue.main.async { self?.status = msg } }
        pipeline.publish = { [weak self] r in DispatchQueue.main.async { self?.apply(r) } }
    }

    func start() async {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            status = "Camera access denied: enable it in Settings"
            return
        }
        UIApplication.shared.isIdleTimerDisabled = true
        setSource(source)
    }

    func setSource(_ s: DepthSource) {
        source = s
        status = nil
        if s != .stereo && mode.needsStereoPair { mode = .standard }
        pipeline.store.latest = nil
        analysis = Analysis()
        if s == .stereo { calibrate() }
        camera.start(s)
    }

    func calibrate() {
        calibrating = true
        push { $0.calibrate = true }
    }

    func measure(at p: CGPoint) {
        if tap.map({ hypot($0.x - p.x, $0.y - p.y) < 0.03 }) == true {
            tap = nil
            tapDistance = nil
            return
        }
        tap = p
        tapDistance = sampleTap()
    }

    func metadata() -> [String: Any] {
        [
            "app": "StereoVision",
            "date": ISO8601DateFormatter().string(from: .now),
            "source": source.rawValue,
            "mode": mode.title,
            "matcher": source == .stereo ? matcher.rawValue : "n/a",
            "max_disparity": maxDisparity,
            "lr_check": lrCheck,
            "p1": Int(p1), "p2": Int(p2),
            "uniqueness": uniqueness,
            "range_m": [minZ, maxZ],
            "rectification": ["flip": flip, "dy_px": yOffset],
            "detector": detectorName,
        ]
    }

    func resetOrbit() {
        yaw = 0
        pitch = 0
        orbit = 1.2
    }

    private func sampleTap() -> Float? {
        guard let tap, let f = pipeline.store.latest else { return nil }
        return f.sample(CGRect(x: tap.x - 0.012, y: tap.y - 0.02, width: 0.024, height: 0.04), q: 0.5)
    }

    private func push(_ change: (inout PipelineSettings) -> Void) {
        pipeline.settings.withLockUnchecked { change(&$0) }
    }

    private func apply(_ r: FrameReport) {
        analysis = r.analysis
        fps = r.fps
        gpuMs = r.gpuMs
        baselineMM = r.baselineMM
        K = r.K
        depthSize = r.depthSize
        if abs(aspect - r.aspect) > 0.01 { aspect = r.aspect }
        if let c = r.calibration {
            flip = c.flip
            yOffset = c.dy
            calibrating = false
        }
        if tap != nil { tapDistance = sampleTap() }
    }
}
