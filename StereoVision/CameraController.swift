import AVFoundation

enum DepthSource: String, CaseIterable, Identifiable {
    case stereo, apple, lidar
    var id: Self { self }

    var title: String {
        switch self {
        case .stereo: "Custom stereo (Census + SGM)"
        case .apple: "Apple Dual-Wide depth"
        case .lidar: "LiDAR depth"
        }
    }

    var short: String { rawValue.uppercased() }

    var deviceType: AVCaptureDevice.DeviceType {
        self == .lidar ? .builtInLiDARDepthCamera : .builtInDualWideCamera
    }

    static let available: [DepthSource] = allCases.filter {
        AVCaptureDevice.default($0.deviceType, for: .video, position: .back) != nil
    }
}

final class CameraController: NSObject, AVCaptureDataOutputSynchronizerDelegate, @unchecked Sendable {
    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ s: String) { errorDescription = s }
    }

    let dataQueue = DispatchQueue(label: "camera.data", qos: .userInteractive)
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private var session: AVCaptureSession?
    private var sync: AVCaptureDataOutputSynchronizer?
    private var outA: AVCaptureOutput?
    private var outB: AVCaptureOutput?

    var onStereo: ((CMSampleBuffer, CMSampleBuffer) -> Void)?
    var onDepth: ((CMSampleBuffer, AVDepthData) -> Void)?
    var onDevices: ((AVCaptureDevice, AVCaptureDevice) -> Void)?
    var onError: ((String) -> Void)?

    func start(_ source: DepthSource) {
        sessionQueue.async { [self] in
            session?.stopRunning()
            session = nil
            sync = nil
            do {
                let s = try source == .stereo ? makeStereo() : makeDepth(source.deviceType)
                session = s
                s.startRunning()
            } catch {
                onError?(error.localizedDescription)
            }
        }
    }

    func stop() {
        sessionQueue.async { [self] in session?.stopRunning() }
    }

    private func device(_ type: AVCaptureDevice.DeviceType) throws -> AVCaptureDevice {
        guard let d = AVCaptureDevice.default(type, for: .video, position: .back) else {
            throw Failure("\(type.rawValue) not available")
        }
        return d
    }

    private func dims(_ f: AVCaptureDevice.Format) -> CMVideoDimensions {
        CMVideoFormatDescriptionGetDimensions(f.formatDescription)
    }

    private func is169(_ f: AVCaptureDevice.Format) -> Bool {
        let d = dims(f)
        return d.width * 9 == d.height * 16
    }

    private func makeStereo() throws -> AVCaptureSession {
        guard AVCaptureMultiCamSession.isMultiCamSupported else { throw Failure("Multi-cam not supported") }
        let dev = try device(.builtInDualWideCamera)
        guard let wide = dev.constituentDevices.first(where: { $0.deviceType == .builtInWideAngleCamera }),
              let ultra = dev.constituentDevices.first(where: { $0.deviceType == .builtInUltraWideCamera }) else {
            throw Failure("Constituent cameras not found")
        }
        let formats = dev.formats.filter { f in
            f.isMultiCamSupported && is169(f) && f.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 30 }
        }
        guard let format = formats.min(by: { abs(dims($0).width - 1280) < abs(dims($1).width - 1280) }) else {
            throw Failure("No 16:9 multi-cam format")
        }

        let session = AVCaptureMultiCamSession()
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        let input = try AVCaptureDeviceInput(device: dev)
        guard session.canAddInput(input) else { throw Failure("Cannot add input") }
        session.addInputWithNoConnections(input)

        func output(_ type: AVCaptureDevice.DeviceType) throws -> AVCaptureVideoDataOutput {
            guard let port = input.ports(for: .video, sourceDeviceType: type, sourceDevicePosition: .back).first else {
                throw Failure("Missing port \(type.rawValue)")
            }
            let out = AVCaptureVideoDataOutput()
            out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            out.alwaysDiscardsLateVideoFrames = true
            guard session.canAddOutput(out) else { throw Failure("Cannot add output") }
            session.addOutputWithNoConnections(out)
            let conn = AVCaptureConnection(inputPorts: [port], output: out)
            guard session.canAddConnection(conn) else { throw Failure("Cannot add connection") }
            session.addConnection(conn)
            if conn.isCameraIntrinsicMatrixDeliverySupported { conn.isCameraIntrinsicMatrixDeliveryEnabled = true }
            return out
        }

        let wOut = try output(.builtInWideAngleCamera)
        let uOut = try output(.builtInUltraWideCamera)

        try dev.lockForConfiguration()
        dev.activeFormat = format
        dev.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        dev.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        dev.unlockForConfiguration()

        outA = wOut
        outB = uOut
        let s = AVCaptureDataOutputSynchronizer(dataOutputs: [wOut, uOut])
        s.setDelegate(self, queue: dataQueue)
        sync = s
        onDevices?(wide, ultra)
        return session
    }

    private func makeDepth(_ type: AVCaptureDevice.DeviceType) throws -> AVCaptureSession {
        let dev = try device(type)
        let session = AVCaptureSession()
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .inputPriority

        let input = try AVCaptureDeviceInput(device: dev)
        guard session.canAddInput(input) else { throw Failure("Cannot add input") }
        session.addInput(input)

        let video = AVCaptureVideoDataOutput()
        video.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        video.alwaysDiscardsLateVideoFrames = true
        let depth = AVCaptureDepthDataOutput()
        depth.isFilteringEnabled = true
        depth.alwaysDiscardsLateDepthData = true
        guard session.canAddOutput(video), session.canAddOutput(depth) else { throw Failure("Cannot add depth output") }
        session.addOutput(video)
        session.addOutput(depth)

        let candidates = dev.formats.filter { !$0.supportedDepthDataFormats.isEmpty }
        let preferred = candidates.filter(is169).isEmpty ? candidates : candidates.filter(is169)
        guard let format = preferred.min(by: { abs(dims($0).width - 1920) < abs(dims($1).width - 1920) }) else {
            throw Failure("No depth-capable format on \(type.rawValue)")
        }
        let depthFormat = format.supportedDepthDataFormats
            .filter {
                let t = CMFormatDescriptionGetMediaSubType($0.formatDescription)
                return t == kCVPixelFormatType_DepthFloat32 || t == kCVPixelFormatType_DisparityFloat16 || t == kCVPixelFormatType_DepthFloat16 || t == kCVPixelFormatType_DisparityFloat32
            }
            .max { dims($0).width < dims($1).width }

        try dev.lockForConfiguration()
        dev.activeFormat = format
        if let depthFormat { dev.activeDepthDataFormat = depthFormat }
        dev.unlockForConfiguration()

        depth.connection(with: .depthData)?.isEnabled = true
        outA = video
        outB = depth
        let s = AVCaptureDataOutputSynchronizer(dataOutputs: [video, depth])
        s.setDelegate(self, queue: dataQueue)
        sync = s
        return session
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let outA, let outB else { return }
        if let depthOut = outB as? AVCaptureDepthDataOutput {
            guard let v = collection.synchronizedData(for: outA) as? AVCaptureSynchronizedSampleBufferData, !v.sampleBufferWasDropped,
                  let d = collection.synchronizedData(for: depthOut) as? AVCaptureSynchronizedDepthData, !d.depthDataWasDropped else { return }
            onDepth?(v.sampleBuffer, d.depthData)
        } else {
            guard let a = collection.synchronizedData(for: outA) as? AVCaptureSynchronizedSampleBufferData, !a.sampleBufferWasDropped,
                  let b = collection.synchronizedData(for: outB) as? AVCaptureSynchronizedSampleBufferData, !b.sampleBufferWasDropped else { return }
            onStereo?(a.sampleBuffer, b.sampleBuffer)
        }
    }
}
