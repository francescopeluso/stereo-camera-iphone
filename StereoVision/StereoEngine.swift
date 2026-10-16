import AVFoundation
import CoreVideo
import Metal
import simd

struct Intrinsics: Sendable {
    var fx: Float, fy: Float, cx: Float, cy: Float
}

final class StereoFrame: @unchecked Sendable {
    let color: MTLTexture
    let right: MTLTexture?
    let depth: MTLTexture
    let width: Int, height: Int
    let K: Intrinsics
    let image: CVPixelBuffer
    let gpuMs: Double
    let baselineMM: Float
    private let buffer: MTLBuffer
    private let rowStride: Int
    private let retained: [AnyObject]

    init(color: MTLTexture, right: MTLTexture?, target: DepthTarget, K: Intrinsics, image: CVPixelBuffer, gpuMs: Double, baselineMM: Float, retained: [AnyObject]) {
        self.color = color
        self.right = right
        self.depth = target.texture
        self.buffer = target.buffer
        self.width = target.width
        self.height = target.height
        self.rowStride = target.bytesPerRow / 4
        self.K = K
        self.image = image
        self.gpuMs = gpuMs
        self.baselineMM = baselineMM
        self.retained = retained
    }

    func z(_ x: Int, _ y: Int) -> Float {
        buffer.contents().assumingMemoryBound(to: Float.self)[y * rowStride + x]
    }

    func row(_ y: Int) -> UnsafePointer<Float> {
        UnsafePointer(buffer.contents().assumingMemoryBound(to: Float.self) + y * rowStride)
    }

    func sample(_ m: Mask) -> Float? {
        func set(_ x: Int, _ y: Int) -> Bool { x >= 0 && y >= 0 && x < m.w && y < m.h && m.bits[y * m.w + x] != 0 }
        var inner: [Float] = [], all: [Float] = []
        for y in 0..<m.h {
            for x in 0..<m.w where set(x, y) {
                let u = m.rect.minX + (CGFloat(x) + 0.5) / CGFloat(m.w) * m.rect.width
                let v = m.rect.minY + (CGFloat(y) + 0.5) / CGFloat(m.h) * m.rect.height
                let z = self.z(min(width - 1, Int(u * CGFloat(width))), min(height - 1, Int(v * CGFloat(height))))
                guard z > 0 else { continue }
                all.append(z)
                if set(x - 1, y) && set(x + 1, y) && set(x, y - 1) && set(x, y + 1) { inner.append(z) }
            }
        }
        var v = inner.count >= 8 ? inner : all
        guard v.count >= 6 else { return nil }
        v.sort()
        return v[v.count / 2]
    }

    func sample(_ r: CGRect, q: Float) -> Float? {
        let x0 = max(0, Int(r.minX * CGFloat(width))), x1 = min(width, Int(r.maxX * CGFloat(width)))
        let y0 = max(0, Int(r.minY * CGFloat(height))), y1 = min(height, Int(r.maxY * CGFloat(height)))
        guard x1 > x0, y1 > y0 else { return nil }
        let step = max(1, Int(sqrt(Double((x1 - x0) * (y1 - y0)) / 400)))
        var v: [Float] = []
        v.reserveCapacity(500)
        for y in stride(from: y0, to: y1, by: step) {
            let p = row(y)
            for x in stride(from: x0, to: x1, by: step) where p[x] > 0 { v.append(p[x]) }
        }
        guard v.count >= 6 else { return nil }
        v.sort()
        return v[Int(Float(v.count - 1) * q)]
    }
}

final class DepthTarget {
    let buffer: MTLBuffer
    let texture: MTLTexture
    let width: Int, height: Int, bytesPerRow: Int

    init(device: MTLDevice, width: Int, height: Int) {
        let align = device.minimumLinearTextureAlignment(for: .r32Float)
        bytesPerRow = (width * 4 + align - 1) / align * align
        buffer = device.makeBuffer(length: bytesPerRow * height, options: .storageModeShared)!
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .shared
        texture = buffer.makeTexture(descriptor: d, offset: 0, bytesPerRow: bytesPerRow)!
        self.width = width
        self.height = height
    }
}

final class StereoEngine: @unchecked Sendable {
    static let workW = 640, workH = 360, maxD = 96

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let rectifyPSO, censusPSO, matchPSO, depthPSO: MTLComputePipelineState
    private let costPSO, pathPSO, wtaRightPSO, wtaLeftPSO: MTLComputePipelineState
    private let costBuffer, sumBuffer: MTLBuffer
    private let dispRight: MTLTexture
    private var cache: CVMetalTextureCache!
    private var leftPool: CVPixelBufferPool!
    private let slots: [Slot]
    private var appleTargets: [DepthTarget] = []
    private var slotIndex = 0

    private var R = matrix_identity_float3x3
    private var t = SIMD3<Float>(12, 0, 0)
    private var wideFOV: Float = 70, ultraFOV: Float = 108
    private(set) var hasExtrinsics = false

    private final class Slot {
        let right, censusL, censusR, disp, cost: MTLTexture
        let target: DepthTarget

        init(device: MTLDevice, w: Int, h: Int) {
            func tex(_ f: MTLPixelFormat, shared: Bool = false) -> MTLTexture {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: w, height: h, mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = shared ? .shared : .private
                return device.makeTexture(descriptor: d)!
            }
            right = tex(.rgba8Unorm, shared: true)
            censusL = tex(.rg32Uint)
            censusR = tex(.rg32Uint)
            disp = tex(.r32Float)
            cost = tex(.r32Float, shared: true)
            target = DepthTarget(device: device, width: w, height: h)
        }
    }

    struct Geometry {
        var MW: simd_float3x3, MU: simd_float3x3
        var K: Intrinsics
        var dir: Int32
        var fB: Float
        var baselineMM: Float
    }

    init(device: MTLDevice) {
        self.device = device
        queue = device.makeCommandQueue()!
        let lib = device.makeDefaultLibrary()!
        func pso(_ n: String) -> MTLComputePipelineState { try! device.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
        rectifyPSO = pso("rectify")
        censusPSO = pso("census")
        matchPSO = pso("match")
        depthPSO = pso("toDepth")
        costPSO = pso("costVolume")
        pathPSO = pso("sgmPath")
        wtaRightPSO = pso("wtaRight")
        wtaLeftPSO = pso("wtaLeft")
        let cells = Self.workW * Self.workH * Self.maxD
        costBuffer = device.makeBuffer(length: cells, options: .storageModePrivate)!
        sumBuffer = device.makeBuffer(length: cells * 2, options: .storageModePrivate)!
        let rd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: Self.workW, height: Self.workH, mipmapped: false)
        rd.usage = [.shaderRead, .shaderWrite]
        rd.storageMode = .private
        dispRight = device.makeTexture(descriptor: rd)!
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: Self.workW,
            kCVPixelBufferHeightKey as String: Self.workH,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &leftPool)
        slots = (0..<3).map { _ in Slot(device: device, w: Self.workW, h: Self.workH) }
    }

    func configure(wide: AVCaptureDevice, ultra: AVCaptureDevice) {
        wideFOV = wide.activeFormat.videoFieldOfView
        ultraFOV = ultra.activeFormat.videoFieldOfView
        guard let data = AVCaptureDevice.extrinsicMatrix(from: ultra, to: wide) else { hasExtrinsics = false; return }
        let m = data.withUnsafeBytes { $0.loadUnaligned(as: simd_float4x3.self) }
        R = simd_float3x3(columns: (m.columns.0, m.columns.1, m.columns.2))
        t = m.columns.3
        hasExtrinsics = simd_length(t) > 1
        if !hasExtrinsics { R = matrix_identity_float3x3; t = SIMD3(12, 0, 0) }
    }

    private func texture(_ pb: CVPixelBuffer, writable: Bool = false) -> (MTLTexture, CVMetalTexture)? {
        let attrs = writable ? [kCVMetalTextureUsage: MTLTextureUsage([.shaderRead, .shaderWrite]).rawValue] as CFDictionary : nil
        var cv: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, attrs, .bgra8Unorm, CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), 0, &cv)
        guard let cv, let tex = CVMetalTextureGetTexture(cv) else { return nil }
        return (tex, cv)
    }

    private func intrinsics(_ sb: CMSampleBuffer, fov: Float, size: SIMD2<Float>) -> simd_float3x3 {
        if let data = CMGetAttachment(sb, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data {
            return data.withUnsafeBytes { $0.loadUnaligned(as: simd_float3x3.self) }
        }
        let f = size.x / (2 * tan(fov * .pi / 360))
        return simd_float3x3(columns: (SIMD3(f, 0, 0), SIMD3(0, f, 0), SIMD3(size.x / 2, size.y / 2, 1)))
    }

    func geometry(KW: simd_float3x3, KU: simd_float3x3, srcW: Float, flip: Bool, dy: Float) -> Geometry {
        var R = self.R, t = self.t
        if flip { R = R.transpose; t = -(R * t) }
        let dir: Float = t.x >= 0 ? 1 : -1
        let e1 = simd_normalize(t) * dir
        let e2 = simd_normalize(SIMD3(-e1.y, e1.x, 0))
        let e3 = simd_cross(e1, e2)
        let Rr = simd_float3x3(rows: [e1, e2, e3])
        let s = Float(Self.workW) / srcW
        let K = Intrinsics(fx: KW[0][0] * s, fy: KW[1][1] * s, cx: KW[2][0] * s, cy: KW[2][1] * s)
        let Kr = simd_float3x3(columns: (SIMD3(K.fx, 0, 0), SIMD3(0, K.fy, 0), SIMD3(K.cx, K.cy, 1)))
        let T = simd_float3x3(columns: (SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, dy, 1)))
        let back = Rr.transpose * Kr.inverse
        let baseline = simd_length(t)
        return Geometry(MW: KW * back, MU: KU * R.transpose * back * T, K: K, dir: Int32(dir), fB: K.fx * baseline / 1000, baselineMM: baseline)
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ textures: [MTLTexture], buffers: [MTLBuffer] = [], bytes: UnsafeRawPointer? = nil, length: Int = 0) {
        enc.setComputePipelineState(pso)
        for (i, t) in textures.enumerated() { enc.setTexture(t, index: i) }
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        if let bytes { enc.setBytes(bytes, length: length, index: buffers.count) }
        let tw = pso.threadExecutionWidth
        let th = max(1, pso.maxTotalThreadsPerThreadgroup / tw)
        enc.dispatchThreads(MTLSize(width: Self.workW, height: Self.workH, depth: 1), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }

    private func encodeRectify(_ enc: MTLComputeCommandEncoder, _ src: MTLTexture, _ dst: MTLTexture, _ M: simd_float3x3) {
        var p = RectParams(M: M, srcSize: SIMD2(Float(src.width), Float(src.height)))
        dispatch(enc, rectifyPSO, [src, dst], bytes: &p, length: MemoryLayout<RectParams>.stride)
    }

    private func encodeMatch(_ enc: MTLComputeCommandEncoder, _ slot: Slot, _ g: Geometry, _ s: PipelineSettings) {
        var m = MatchParams(maxDisp: Int32(s.maxDisparity), dir: g.dir, uniqueness: s.uniqueness, maxCost: 1200)
        dispatch(enc, matchPSO, [slot.censusL, slot.censusR, slot.disp, slot.cost], bytes: &m, length: MemoryLayout<MatchParams>.stride)
    }

    private func encodeSGM(_ enc: MTLComputeCommandEncoder, _ slot: Slot, _ left: MTLTexture, _ g: Geometry, _ s: PipelineSettings) {
        let D = min(Self.maxD, max(32, s.maxDisparity / 32 * 32))
        var p = SGMParams(W: Int32(Self.workW), H: Int32(Self.workH), D: Int32(D), dir: g.dir, dx: 0, dy: 0,
                          P1: UInt32(s.p1), P2: UInt32(s.p2), first: 1, uniqueness: s.uniqueness, lrCheck: s.lrCheck ? 1 : 0, pad: 0)
        let len = MemoryLayout<SGMParams>.stride
        dispatch(enc, costPSO, [slot.censusL, slot.censusR], buffers: [costBuffer], bytes: &p, length: len)
        enc.setComputePipelineState(pathPSO)
        enc.setBuffer(costBuffer, offset: 0, index: 0)
        enc.setBuffer(sumBuffer, offset: 0, index: 1)
        enc.setTexture(left, index: 0)
        for (i, (dx, dy)) in [(1, 0), (-1, 0), (0, 1), (0, -1)].enumerated() {
            p.dx = Int32(dx)
            p.dy = Int32(dy)
            p.first = i == 0 ? 1 : 0
            enc.setBytes(&p, length: len, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: dx != 0 ? Self.workH : Self.workW, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
        }
        dispatch(enc, wtaRightPSO, [dispRight], buffers: [sumBuffer], bytes: &p, length: len)
        dispatch(enc, wtaLeftPSO, [dispRight, slot.disp], buffers: [sumBuffer], bytes: &p, length: len)
    }

    private struct Inputs {
        let wTex: MTLTexture, uTex: MTLTexture
        let KW: simd_float3x3, KU: simd_float3x3
        let keep: [AnyObject]
    }

    private func inputs(_ wide: CMSampleBuffer, _ ultra: CMSampleBuffer) -> Inputs? {
        guard let wpb = CMSampleBufferGetImageBuffer(wide), let upb = CMSampleBufferGetImageBuffer(ultra),
              let (wTex, wcv) = texture(wpb), let (uTex, ucv) = texture(upb) else { return nil }
        let ws = SIMD2(Float(wTex.width), Float(wTex.height)), us = SIMD2(Float(uTex.width), Float(uTex.height))
        return Inputs(wTex: wTex, uTex: uTex, KW: intrinsics(wide, fov: wideFOV, size: ws), KU: intrinsics(ultra, fov: ultraFOV, size: us), keep: [wcv, ucv])
    }

    func stereo(_ wide: CMSampleBuffer, _ ultra: CMSampleBuffer, _ s: PipelineSettings) -> StereoFrame? {
        guard let inp = inputs(wide, ultra) else { return nil }
        var lpb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, leftPool, &lpb)
        guard let lpb, let (lTex, lcv) = texture(lpb, writable: true), let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return nil }
        slotIndex = (slotIndex + 1) % slots.count
        let slot = slots[slotIndex]
        let g = geometry(KW: inp.KW, KU: inp.KU, srcW: Float(inp.wTex.width), flip: s.flip, dy: s.yOffset)
        encodeRectify(enc, inp.wTex, lTex, g.MW)
        encodeRectify(enc, inp.uTex, slot.right, g.MU)
        dispatch(enc, censusPSO, [lTex, slot.censusL])
        dispatch(enc, censusPSO, [slot.right, slot.censusR])
        if s.matcher == .sgm { encodeSGM(enc, slot, lTex, g, s) } else { encodeMatch(enc, slot, g, s) }
        var d = DepthParams(fB: g.fB, minZ: s.minZ, maxZ: s.maxZ, pad: 0)
        dispatch(enc, depthPSO, [slot.disp, slot.target.texture], bytes: &d, length: MemoryLayout<DepthParams>.stride)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        return StereoFrame(color: lTex, right: slot.right, target: slot.target, K: g.K, image: lpb, gpuMs: ms, baselineMM: g.baselineMM, retained: [lcv])
    }

    func calibrate(_ wide: CMSampleBuffer, _ ultra: CMSampleBuffer, _ s: PipelineSettings) -> (flip: Bool, dy: Float) {
        guard let inp = inputs(wide, ultra) else { return (s.flip, s.yOffset) }
        let slot = slots[0]
        let n = Self.workW * Self.workH
        var costs = [Float](repeating: 0, count: n)

        func score(flip: Bool, dy: Float) -> Float {
            let g = geometry(KW: inp.KW, KU: inp.KU, srcW: Float(inp.wTex.width), flip: flip, dy: dy)
            guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return .infinity }
            encodeRectify(enc, inp.wTex, slot.right, g.MW)
            dispatch(enc, censusPSO, [slot.right, slot.censusL])
            encodeRectify(enc, inp.uTex, slot.right, g.MU)
            dispatch(enc, censusPSO, [slot.right, slot.censusR])
            encodeMatch(enc, slot, g, s)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            costs.withUnsafeMutableBytes {
                slot.cost.getBytes($0.baseAddress!, bytesPerRow: Self.workW * 4, from: MTLRegionMake2D(0, 0, Self.workW, Self.workH), mipmapLevel: 0)
            }
            var sum: Float = 0, cnt: Float = 0
            for c in costs where c >= 0 { sum += c; cnt += 1 }
            return cnt > 0 ? sum / cnt : .infinity
        }

        var best = (flip: s.flip, dy: s.yOffset, score: Float.infinity)
        for flip in [false, true] {
            for i in -12...12 {
                let v = score(flip: flip, dy: Float(i))
                if v < best.score { best = (flip, Float(i), v) }
            }
        }
        for d in [-0.5, -0.25, 0.25, 0.5] as [Float] {
            let v = score(flip: best.flip, dy: best.dy + d)
            if v < best.score { best = (best.flip, best.dy + d, v) }
        }
        return (best.flip, best.dy)
    }

    func depthData(_ video: CMSampleBuffer, _ depthData: AVDepthData, _ s: PipelineSettings) -> StereoFrame? {
        guard let pb = CMSampleBufferGetImageBuffer(video), let (tex, cv) = texture(pb) else { return nil }
        let data = depthData.depthDataType == kCVPixelFormatType_DepthFloat32 ? depthData : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = data.depthDataMap
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        if appleTargets.first?.width != w || appleTargets.first?.height != h {
            appleTargets = (0..<3).map { _ in DepthTarget(device: device, width: w, height: h) }
        }
        slotIndex = (slotIndex + 1) % appleTargets.count
        let target = appleTargets[slotIndex]
        CVPixelBufferLockBaseAddress(map, .readOnly)
        let src = CVPixelBufferGetBaseAddress(map)!
        let srcStride = CVPixelBufferGetBytesPerRow(map)
        let dst = target.buffer.contents()
        for y in 0..<h {
            let a = (src + y * srcStride).assumingMemoryBound(to: Float.self)
            let b = (dst + y * target.bytesPerRow).assumingMemoryBound(to: Float.self)
            for x in 0..<w {
                let z = a[x]
                b[x] = z.isFinite && z >= s.minZ && z <= s.maxZ ? z : 0
            }
        }
        CVPixelBufferUnlockBaseAddress(map, .readOnly)

        var K: Intrinsics
        if let cal = data.cameraCalibrationData {
            let m = cal.intrinsicMatrix
            let ref = cal.intrinsicMatrixReferenceDimensions
            let sc = Float(w) / Float(ref.width)
            let off = (Float(ref.height) * sc - Float(h)) / 2
            K = Intrinsics(fx: m[0][0] * sc, fy: m[1][1] * sc, cx: m[2][0] * sc, cy: m[2][1] * sc - off)
        } else {
            let f = Float(w) / (2 * tan(wideFOV * .pi / 360))
            K = Intrinsics(fx: f, fy: f, cx: Float(w) / 2, cy: Float(h) / 2)
        }
        return StereoFrame(color: tex, right: nil, target: target, K: K, image: pb, gpuMs: 0, baselineMM: 0, retained: [cv])
    }
}

struct RectParams { var M: simd_float3x3; var srcSize: SIMD2<Float> }
struct MatchParams { var maxDisp: Int32; var dir: Int32; var uniqueness: Float; var maxCost: Float }
struct DepthParams { var fB: Float; var minZ: Float; var maxZ: Float; var pad: Int32 }
struct SGMParams {
    var W: Int32, H: Int32, D: Int32, dir: Int32, dx: Int32, dy: Int32
    var P1: UInt32, P2: UInt32
    var first: Int32
    var uniqueness: Float
    var lrCheck: Int32, pad: Int32
}
