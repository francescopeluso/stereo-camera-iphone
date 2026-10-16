import Foundation
import Metal

struct MatchParams { var maxDisp: Int32; var dir: Int32; var uniqueness: Float; var maxCost: Float }
struct SGMParams {
    var W: Int32, H: Int32, D: Int32, dir: Int32, dx: Int32, dy: Int32
    var P1: UInt32, P2: UInt32
    var first: Int32
    var uniqueness: Float
    var lrCheck: Int32, pad: Int32
}

let W = 640, H = 360, D = 64
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let src = try String(contentsOf: root.appendingPathComponent("StereoVision/Shaders.metal"), encoding: .utf8)
let dev = MTLCreateSystemDefaultDevice()!
let lib = try dev.makeLibrary(source: src, options: nil)
let queue = dev.makeCommandQueue()!
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

func truth(_ x: Int, _ y: Int) -> Float {
    if x >= 260 && x < 420 && y >= 110 && y < 250 { return 30 }
    return 8 + Float(y) / Float(H) * 6
}

var rng = SystemRandomNumberGenerator()
let texW = W + 64
var base = [Float](repeating: 0, count: texW * H)
for i in base.indices { base[i] = Float.random(in: 0...1, using: &rng) }
var blurred = base
for y in 1..<(H - 1) {
    for x in 1..<(texW - 1) {
        var s: Float = 0
        for dy in -1...1 { for dx in -1...1 { s += base[(y + dy) * texW + x + dx] } }
        blurred[y * texW + x] = s / 9
    }
}
func sampleTex(_ x: Float, _ y: Int) -> Float {
    let x0 = max(0, min(texW - 2, Int(x)))
    let f = x - Float(x0)
    let v = blurred[y * texW + x0] * (1 - f) + blurred[y * texW + x0 + 1] * f
    let contrast: Float = y > 200 && y < 330 ? 0.04 : 1
    return 0.5 + (v - 0.5) * contrast
}
func image(_ f: (Int, Int) -> Float) -> MTLTexture {
    var px = [UInt8](repeating: 255, count: W * H * 4)
    for y in 0..<H {
        for x in 0..<W {
            let v = UInt8(max(0, min(255, f(x, y) * 255 + Float.random(in: -3...3, using: &rng))))
            px[(y * W + x) * 4] = v; px[(y * W + x) * 4 + 1] = v; px[(y * W + x) * 4 + 2] = v
        }
    }
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: W, height: H, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]
    let t = dev.makeTexture(descriptor: d)!
    t.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0, withBytes: px, bytesPerRow: W * 4)
    return t
}
let left = image { x, y in sampleTex(Float(x), y) }
let right = image { x, y in sampleTex(Float(x) + truth(x, y), y) }

func tex(_ f: MTLPixelFormat) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: W, height: H, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]
    d.storageMode = .shared
    return dev.makeTexture(descriptor: d)!
}
let cl = tex(.rg32Uint), cr = tex(.rg32Uint), disp = tex(.r32Float), dispR = tex(.r32Float), cost = tex(.r32Float)
let C = dev.makeBuffer(length: W * H * D)!, S = dev.makeBuffer(length: W * H * D * 2)!

func grid(_ enc: MTLComputeCommandEncoder, _ p: MTLComputePipelineState) {
    enc.setComputePipelineState(p)
    enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: p.threadExecutionWidth, height: max(1, p.maxTotalThreadsPerThreadgroup / p.threadExecutionWidth), depth: 1))
}

func run(sgm: Bool, lr: Bool) -> (good: Double, bad: Double, invalid: Double, ms: Double) {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    for (s, d) in [(left, cl), (right, cr)] { enc.setTexture(s, index: 0); enc.setTexture(d, index: 1); grid(enc, pso("census")) }
    if sgm {
        var p = SGMParams(W: Int32(W), H: Int32(H), D: Int32(D), dir: 1, dx: 0, dy: 0, P1: 8, P2: 96, first: 1, uniqueness: 0.1, lrCheck: lr ? 1 : 0, pad: 0)
        let len = MemoryLayout<SGMParams>.stride
        enc.setTexture(cl, index: 0); enc.setTexture(cr, index: 1); enc.setBuffer(C, offset: 0, index: 0); enc.setBytes(&p, length: len, index: 1)
        grid(enc, pso("costVolume"))
        enc.setComputePipelineState(pso("sgmPath"))
        enc.setBuffer(C, offset: 0, index: 0); enc.setBuffer(S, offset: 0, index: 1); enc.setTexture(left, index: 0)
        for (i, (dx, dy)) in [(1, 0), (-1, 0), (0, 1), (0, -1)].enumerated() {
            p.dx = Int32(dx); p.dy = Int32(dy); p.first = i == 0 ? 1 : 0
            enc.setBytes(&p, length: len, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: dx != 0 ? H : W, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
        }
        enc.setBuffer(S, offset: 0, index: 0); enc.setBytes(&p, length: len, index: 1)
        enc.setTexture(dispR, index: 0); grid(enc, pso("wtaRight"))
        enc.setTexture(dispR, index: 0); enc.setTexture(disp, index: 1); grid(enc, pso("wtaLeft"))
    } else {
        var m = MatchParams(maxDisp: Int32(D), dir: 1, uniqueness: 0.1, maxCost: 1200)
        enc.setTexture(cl, index: 0); enc.setTexture(cr, index: 1); enc.setTexture(disp, index: 2); enc.setTexture(cost, index: 3)
        enc.setBytes(&m, length: MemoryLayout<MatchParams>.stride, index: 0)
        grid(enc, pso("match"))
    }
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    var out = [Float](repeating: 0, count: W * H)
    disp.getBytes(&out, bytesPerRow: W * 4, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
    var good = 0, bad = 0, invalid = 0, n = 0
    for y in 8..<(H - 8) {
        for x in (D + 8)..<(W - 8) {
            n += 1
            let d = out[y * W + x]
            if d <= 0 { invalid += 1 } else if abs(d - truth(x, y)) <= 1 { good += 1 } else { bad += 1 }
        }
    }
    let t = Double(n)
    return (Double(good) / t * 100, Double(bad) / t * 100, Double(invalid) / t * 100, (cb.gpuEndTime - cb.gpuStartTime) * 1000)
}

_ = run(sgm: true, lr: true)
print("Scene: slanted plane d=8..14 px, rectangle d=30 px, band y 200-330 with near-zero texture (4% contrast)")
for (name, sgm, lr) in [("Block 5x5   ", false, false), ("SGM-4       ", true, false), ("SGM-4 + LR  ", true, true)] {
    let r = run(sgm: sgm, lr: lr)
    print(String(format: "%@ correct(±1px) %5.1f%%  wrong %5.1f%%  invalid %5.1f%%  GPU %5.1f ms", name, r.good, r.bad, r.invalid, r.ms))
}
