import CoreGraphics
import Foundation
import ImageIO
import Metal

struct MatchParams { var maxDisp: Int32; var dir: Int32; var uniqueness: Float; var maxCost: Float }
struct SGMParams {
    var W: Int32, H: Int32, D: Int32, dir: Int32, dx: Int32, dy: Int32
    var P1: UInt32, P2: UInt32
    var first: Int32
    var uniqueness: Float
    var lrCheck: Int32, pad: Int32
}

let args = CommandLine.arguments
guard args.count == 5 else {
    print("usage: swift run_sgm.swift <Shaders.metal> <left.png> <right.png> <outdir>")
    exit(1)
}
let W = 640, H = 360, D = 64
let dev = MTLCreateSystemDefaultDevice()!
let lib = try dev.makeLibrary(source: String(contentsOfFile: args[1], encoding: .utf8), options: nil)
let queue = dev.makeCommandQueue()!
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

func load(_ path: String) -> MTLTexture {
    let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)!
    let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
    var px = [UInt8](repeating: 0, count: W * H * 4)
    let ctx = CGContext(data: &px, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: W, height: H, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]
    let t = dev.makeTexture(descriptor: d)!
    t.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0, withBytes: px, bytesPerRow: W * 4)
    return t
}

func tex(_ f: MTLPixelFormat) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: W, height: H, mipmapped: false)
    d.usage = [.shaderRead, .shaderWrite]
    d.storageMode = .shared
    return dev.makeTexture(descriptor: d)!
}

let left = load(args[2]), right = load(args[3])
let cl = tex(.rg32Uint), cr = tex(.rg32Uint), disp = tex(.r32Float), dispR = tex(.r32Float), cost = tex(.r32Float)
let C = dev.makeBuffer(length: W * H * D)!, S = dev.makeBuffer(length: W * H * D * 2)!

func grid(_ enc: MTLComputeCommandEncoder, _ p: MTLComputePipelineState) {
    enc.setComputePipelineState(p)
    enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: p.threadExecutionWidth, height: max(1, p.maxTotalThreadsPerThreadgroup / p.threadExecutionWidth), depth: 1))
}

func run(sgm: Bool, lr: Bool, name: String) throws {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    for (s, d) in [(left, cl), (right, cr)] {
        enc.setTexture(s, index: 0)
        enc.setTexture(d, index: 1)
        grid(enc, pso("census"))
    }
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
    try out.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: args[4]).appendingPathComponent("\(name).f32")) }
    print(name, String(format: "%.1f%% valid, GPU %.1f ms", Double(out.filter { $0 > 0 }.count) / Double(W * H) * 100, (cb.gpuEndTime - cb.gpuStartTime) * 1000))
}

try run(sgm: true, lr: true, name: "disp_sgm_lr")
try run(sgm: true, lr: false, name: "disp_sgm")
try run(sgm: false, lr: false, name: "disp_block")
