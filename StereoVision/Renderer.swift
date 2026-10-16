import MetalKit
import simd

struct DisplayParams {
    var mode: Int32
    var minZ: Float
    var maxZ: Float
    var alpha: Float
    var hasRight: Int32
    var hasLabels: Int32
    var pad0: Int32 = 0
    var pad1: Int32 = 0
}

struct PCParams {
    var mvp: simd_float4x4
    var fx: Float, fy: Float, cx: Float, cy: Float
    var minZ: Float, maxZ: Float, pointSize: Float
    var w: UInt32, h: UInt32, step: UInt32
}

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    weak var model: AppModel?
    private let queue: MTLCommandQueue
    private let display: MTLRenderPipelineState
    private let points: MTLRenderPipelineState
    private let depthOff: MTLDepthStencilState
    private let depthOn: MTLDepthStencilState
    private let store: FrameStore
    private var labelTex: MTLTexture

    init(device: MTLDevice, store: FrameStore) {
        self.device = device
        self.store = store
        queue = device.makeCommandQueue()!
        let lib = device.makeDefaultLibrary()!
        func pipeline(_ v: String, _ f: String) -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: v)
            d.fragmentFunction = lib.makeFunction(name: f)
            d.colorAttachments[0].pixelFormat = .bgra8Unorm
            d.depthAttachmentPixelFormat = .depth32Float
            return try! device.makeRenderPipelineState(descriptor: d)
        }
        display = pipeline("fullscreenVertex", "displayFragment")
        points = pipeline("pointVertex", "pointFragment")
        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .always
        ds.isDepthWriteEnabled = false
        depthOff = device.makeDepthStencilState(descriptor: ds)!
        ds.depthCompareFunction = .less
        ds.isDepthWriteEnabled = true
        depthOn = device.makeDepthStencilState(descriptor: ds)!
        labelTex = Self.makeLabels(device, 1, 1)
    }

    private static func makeLabels(_ device: MTLDevice, _ w: Int, _ h: Int) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: w, height: h, mipmapped: false)
        d.usage = .shaderRead
        return device.makeTexture(descriptor: d)!
    }

    func configure(_ v: MTKView) {
        v.device = device
        v.colorPixelFormat = .bgra8Unorm
        v.depthStencilPixelFormat = .depth32Float
        v.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        v.preferredFramesPerSecond = 30
        v.isUserInteractionEnabled = false
        v.delegate = self
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let model, let frame = store.latest, let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }

        if model.mode == .pointCloud {
            drawPoints(enc, frame, model, Float(view.drawableSize.width / max(1, view.drawableSize.height)))
        } else {
            let a = model.analysis
            let hasLabels = a.gridW > 0 && a.labels.count == a.gridW * a.gridH
            if hasLabels {
                if labelTex.width != a.gridW || labelTex.height != a.gridH { labelTex = Self.makeLabels(device, a.gridW, a.gridH) }
                a.labels.withUnsafeBytes {
                    labelTex.replace(region: MTLRegionMake2D(0, 0, a.gridW, a.gridH), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: a.gridW)
                }
            }
            var p = DisplayParams(mode: model.mode.shaderIndex, minZ: model.minZ, maxZ: model.maxZ, alpha: model.overlayAlpha,
                                  hasRight: frame.right == nil ? 0 : 1, hasLabels: hasLabels ? 1 : 0)
            enc.setRenderPipelineState(display)
            enc.setDepthStencilState(depthOff)
            enc.setFragmentTexture(frame.color, index: 0)
            enc.setFragmentTexture(frame.right ?? frame.color, index: 1)
            enc.setFragmentTexture(frame.depth, index: 2)
            enc.setFragmentTexture(labelTex, index: 3)
            enc.setFragmentBytes(&p, length: MemoryLayout<DisplayParams>.stride, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }

    private func drawPoints(_ enc: MTLRenderCommandEncoder, _ f: StereoFrame, _ m: AppModel, _ aspect: Float) {
        let step = max(1, f.width / 320)
        let fovy = 2 * atan(Float(f.height) / 2 / f.K.fy)
        let pivot: Float = 1.2
        let view = translate(0, 0, -m.orbit) * rotateX(m.pitch) * rotateY(m.yaw) * translate(0, 0, pivot)
        var p = PCParams(mvp: perspective(fovy: fovy, aspect: aspect, near: 0.02, far: 50) * view,
                         fx: f.K.fx, fy: f.K.fy, cx: f.K.cx, cy: f.K.cy, minZ: m.minZ, maxZ: m.maxZ, pointSize: 4,
                         w: UInt32(f.width), h: UInt32(f.height), step: UInt32(step))
        enc.setRenderPipelineState(points)
        enc.setDepthStencilState(depthOn)
        enc.setVertexTexture(f.depth, index: 0)
        enc.setVertexTexture(f.color, index: 1)
        enc.setVertexBytes(&p, length: MemoryLayout<PCParams>.stride, index: 0)
        enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: (f.width / step) * (f.height / step))
    }
}

func perspective(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let ys = 1 / tan(fovy / 2), xs = ys / aspect, zs = far / (near - far)
    return simd_float4x4(columns: (SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0), SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * near, 0)))
}

func translate(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = SIMD4(x, y, z, 1)
    return m
}

func rotateX(_ a: Float) -> simd_float4x4 {
    simd_float4x4(simd_quatf(angle: a, axis: SIMD3(1, 0, 0)))
}

func rotateY(_ a: Float) -> simd_float4x4 {
    simd_float4x4(simd_quatf(angle: a, axis: SIMD3(0, 1, 0)))
}
