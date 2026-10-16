import AppKit
import CoreGraphics

let size = 1024
let transparent = CommandLine.arguments.contains("--transparent")
let out = CommandLine.arguments.dropFirst().first { !$0.hasPrefix("--") } ?? (transparent ? "docs/logo.png" : "StereoVision/Assets.xcassets/AppIcon.appiconset/AppIcon.png")
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                    bitmapInfo: (transparent ? CGImageAlphaInfo.premultipliedLast : CGImageAlphaInfo.noneSkipLast).rawValue)!
let S = CGFloat(size)

func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor { CGColor(srgbRed: r, green: g, blue: b, alpha: a) }

func turbo(_ x: CGFloat) -> CGColor {
    let x = min(1, max(0, x))
    let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return color(r, g, b)
}

if !transparent {
    let bg = CGGradient(colorsSpace: cs, colors: [color(0.07, 0.09, 0.19), color(0.02, 0.025, 0.06)] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(bg, start: CGPoint(x: 0, y: S), end: CGPoint(x: 0, y: 0), options: [])

    let horizon = S * 0.30
    for i in 0..<9 {
        let t = CGFloat(i) / 8
        let y = horizon - pow(t, 1.8) * horizon
        ctx.setStrokeColor(turbo(0.15 + 0.8 * t).copy(alpha: 0.55)!)
        ctx.setLineWidth(3 + 5 * t)
        ctx.move(to: CGPoint(x: 0, y: y))
        ctx.addLine(to: CGPoint(x: S, y: y))
        ctx.strokePath()
    }
    ctx.setStrokeColor(color(0.55, 0.65, 1, 0.16))
    ctx.setLineWidth(3)
    for i in -8...8 {
        ctx.move(to: CGPoint(x: S / 2 + CGFloat(i) * 22, y: horizon))
        ctx.addLine(to: CGPoint(x: S / 2 + CGFloat(i) * 210, y: 0))
    }
    ctx.strokePath()
    ctx.setStrokeColor(color(0.35, 0.95, 0.55, 0.10))
    ctx.setLineWidth(3)
    for y in stride(from: horizon + 48, to: S, by: 64) {
        ctx.move(to: CGPoint(x: 0, y: y))
        ctx.addLine(to: CGPoint(x: S, y: y))
    }
    ctx.strokePath()
}

let cy = S * 0.585, r: CGFloat = 245, dx: CGFloat = 140
ctx.setBlendMode(.screen)
for (cx, c) in [(S / 2 - dx, color(1, 0.16, 0.28)), (S / 2 + dx, color(0, 0.82, 1))] {
    ctx.setFillColor(c)
    ctx.fillEllipse(in: CGRect(x: cx - r, y: cy - r, width: 2 * r, height: 2 * r))
}
ctx.setBlendMode(.normal)

for cx in [S / 2 - dx, S / 2 + dx] {
    let ri: CGFloat = 104
    let rect = CGRect(x: cx - ri, y: cy - ri, width: 2 * ri, height: 2 * ri)
    ctx.setFillColor(color(0.03, 0.035, 0.07))
    ctx.fillEllipse(in: rect.insetBy(dx: -22, dy: -22))
    ctx.saveGState()
    ctx.addEllipse(in: rect)
    ctx.clip()
    let glass = CGGradient(colorsSpace: cs, colors: [color(0.22, 0.27, 0.45), color(0.02, 0.02, 0.05)] as CFArray, locations: [0, 1])!
    ctx.drawRadialGradient(glass, startCenter: CGPoint(x: cx, y: cy), startRadius: 0, endCenter: CGPoint(x: cx, y: cy), endRadius: ri, options: [])
    ctx.restoreGState()
    ctx.setStrokeColor(color(0.5, 0.55, 0.7, 0.55))
    ctx.setLineWidth(6)
    ctx.strokeEllipse(in: rect)
    ctx.setStrokeColor(color(0.6, 0.7, 1, 0.25))
    ctx.setLineWidth(4)
    ctx.strokeEllipse(in: rect.insetBy(dx: 38, dy: 38))
    ctx.setLineCap(.round)
    ctx.setStrokeColor(color(1, 1, 1, 0.75))
    ctx.setLineWidth(12)
    ctx.addArc(center: CGPoint(x: cx, y: cy), radius: ri - 24, startAngle: .pi * 0.58, endAngle: .pi * 0.86, clockwise: false)
    ctx.strokePath()
}

var img = ctx.makeImage()!
if transparent {
    img = img.cropping(to: CGRect(x: S / 2 - dx - r - 12, y: S - cy - r - 12, width: 2 * (dx + r + 12), height: 2 * (r + 12)))!
}
let rep = NSBitmapImageRep(cgImage: img)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
