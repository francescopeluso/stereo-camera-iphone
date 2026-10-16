import SwiftUI

struct OverlayView: View {
    let model: AppModel
    var hints = true

    var body: some View {
        Canvas { ctx, size in
            let a = model.analysis
            switch model.mode {
            case .pointCloud:
                guard hints else { break }
                pill(&ctx, "drag: rotate · pinch: zoom · tap: reset", at: CGPoint(x: 10, y: size.height - 8), color: .black)
            case .stereoPair, .anaglyph:
                break
            case .radar:
                radar(&ctx, size, a)
            case .segmentation:
                for b in a.blobs { blob(&ctx, size, b) }
                crosshair(&ctx, size, a.center)
            default:
                for o in a.objects { object(&ctx, size, o) }
                crosshair(&ctx, size, a.center)
            }
            if let t = model.tap, model.mode != .pointCloud, model.mode != .radar {
                let p = CGPoint(x: t.x * size.width, y: t.y * size.height)
                ctx.stroke(Path(ellipseIn: CGRect(x: p.x - 9, y: p.y - 9, width: 18, height: 18)), with: .color(.yellow), lineWidth: 2)
                pill(&ctx, model.tapDistance.map { String(format: "%.2f m", $0) } ?? "n/a", at: CGPoint(x: p.x + 12, y: p.y), color: .yellow.opacity(0.9), dark: true)
            }
        }
        .allowsHitTesting(false)
    }

    private func object(_ ctx: inout GraphicsContext, _ size: CGSize, _ o: TrackedObject) {
        let r = CGRect(x: o.rect.minX * size.width, y: o.rect.minY * size.height, width: o.rect.width * size.width, height: o.rect.height * size.height)
        let c = color(o.label)
        if model.showMasks, let m = o.mask {
            let mr = CGRect(x: m.rect.minX * size.width, y: m.rect.minY * size.height, width: m.rect.width * size.width, height: m.rect.height * size.height)
            ctx.draw(Image(decorative: m.image, scale: 1), in: mr)
        }
        ctx.stroke(Path(roundedRect: r, cornerRadius: 4), with: .color(c), lineWidth: 2.5)
        let dist = o.distance.map { String(format: " · %.2f m", $0) } ?? ""
        pill(&ctx, "#\(o.id) \(o.label) \(Int(o.confidence * 100))%\(dist)", at: CGPoint(x: r.minX, y: max(r.minY, 18)), color: c)
        if o.distance != nil {
            let dims = String(format: "%.0f×%.0f cm · X %+.2f m", o.width * 100, o.height * 100, o.x)
            pill(&ctx, dims, at: CGPoint(x: r.minX + 2, y: r.maxY - 2), color: .black.opacity(0.6), small: true)
        }
    }

    private func blob(_ ctx: inout GraphicsContext, _ size: CGSize, _ b: Blob) {
        guard b.cells >= 80 else { return }
        let r = CGRect(x: b.rect.minX * size.width, y: b.rect.minY * size.height, width: b.rect.width * size.width, height: b.rect.height * size.height)
        ctx.stroke(Path(r), with: .color(.white.opacity(0.7)), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        pill(&ctx, String(format: "S%d · %.2f m · ⌀%.0f cm", b.id, b.distance, b.width * 100), at: CGPoint(x: r.minX, y: max(r.minY, 18)), color: .black.opacity(0.65), small: true)
    }

    private func crosshair(_ ctx: inout GraphicsContext, _ size: CGSize, _ z: Float?) {
        let c = CGPoint(x: size.width / 2, y: size.height / 2)
        var p = Path()
        p.move(to: CGPoint(x: c.x - 12, y: c.y)); p.addLine(to: CGPoint(x: c.x + 12, y: c.y))
        p.move(to: CGPoint(x: c.x, y: c.y - 12)); p.addLine(to: CGPoint(x: c.x, y: c.y + 12))
        ctx.stroke(p, with: .color(.white.opacity(0.85)), lineWidth: 1.5)
        pill(&ctx, z.map { String(format: "%.2f m", $0) } ?? "— m", at: CGPoint(x: c.x + 14, y: c.y - 4), color: .black.opacity(0.6), small: true)
    }

    private func radar(_ ctx: inout GraphicsContext, _ size: CGSize, _ a: Analysis) {
        let maxZ = CGFloat(model.maxZ)
        let o = CGPoint(x: size.width / 2, y: size.height - 14)
        let s = (size.height - 34) / maxZ
        let K = model.K
        let w = Float(model.depthSize.width)
        let half = atan(Double(w / 2 / K.fx))

        var fov = Path()
        fov.move(to: CGPoint(x: o.x - sin(half) * maxZ * s / cos(half), y: o.y - maxZ * s))
        fov.addLine(to: o)
        fov.addLine(to: CGPoint(x: o.x + sin(half) * maxZ * s / cos(half), y: o.y - maxZ * s))
        ctx.fill(fov, with: .color(.white.opacity(0.05)))
        ctx.stroke(fov, with: .color(.white.opacity(0.35)), lineWidth: 1)

        let ringStep: CGFloat = maxZ > 4 ? 1 : 0.5
        for r in stride(from: ringStep, through: maxZ, by: ringStep) {
            let rr = r * s
            ctx.stroke(Path(ellipseIn: CGRect(x: o.x - rr, y: o.y - rr, width: rr * 2, height: rr * 2)), with: .color(.white.opacity(0.18)), lineWidth: 1)
            ctx.draw(Text(String(format: "%.1f m", r)).font(.system(size: 9).monospacedDigit()).foregroundStyle(.white.opacity(0.6)), at: CGPoint(x: o.x + 4, y: o.y - rr - 6), anchor: .leading)
        }

        let n = a.scan.count
        for (i, z) in a.scan.enumerated() where z > 0 {
            let u = (Float(i) + 0.5) / Float(n) * w
            let x = (u - K.cx) * z / K.fx
            let p = CGPoint(x: o.x + CGFloat(x) * s, y: o.y - CGFloat(z) * s)
            let t = Double((z - model.minZ) / (model.maxZ - model.minZ))
            ctx.fill(Path(ellipseIn: CGRect(x: p.x - 2, y: p.y - 2, width: 4, height: 4)), with: .color(Color(hue: 0.66 * min(1, max(0, t)), saturation: 0.9, brightness: 1)))
        }

        for b in a.blobs where b.cells >= 80 {
            let p = CGPoint(x: o.x + CGFloat(b.x) * s, y: o.y - CGFloat(b.distance) * s)
            let hw = max(3, CGFloat(b.width) * s / 2)
            ctx.stroke(Path(CGRect(x: p.x - hw, y: p.y - 2, width: hw * 2, height: 4)), with: .color(.white.opacity(0.5)), lineWidth: 1)
        }

        for obj in a.objects {
            guard let z = obj.distance else { continue }
            let p = CGPoint(x: o.x + CGFloat(obj.x) * s, y: o.y - CGFloat(z) * s)
            let r = max(6, CGFloat(obj.width) * s / 2)
            let c = color(obj.label)
            ctx.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)), with: .color(c.opacity(0.35)))
            ctx.stroke(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)), with: .color(c), lineWidth: 2)
            pill(&ctx, String(format: "%@ %.2f m", obj.label, z), at: CGPoint(x: p.x + r + 2, y: p.y + 6), color: c, small: true)
        }

        var cam = Path()
        cam.move(to: CGPoint(x: o.x - 8, y: o.y + 6)); cam.addLine(to: CGPoint(x: o.x, y: o.y - 6)); cam.addLine(to: CGPoint(x: o.x + 8, y: o.y + 6)); cam.closeSubpath()
        ctx.fill(cam, with: .color(.yellow))
    }

    private func pill(_ ctx: inout GraphicsContext, _ s: String, at p: CGPoint, color: Color, dark: Bool = false, small: Bool = false) {
        let t = ctx.resolve(Text(s).font(.system(size: small ? 10 : 12, weight: .semibold, design: .rounded)).foregroundStyle(dark ? .black : .white))
        let m = t.measure(in: CGSize(width: 600, height: 40))
        let r = CGRect(x: p.x, y: p.y - m.height - 4, width: m.width + 10, height: m.height + 4)
        ctx.fill(Path(roundedRect: r, cornerRadius: 4), with: .color(color))
        ctx.draw(t, at: CGPoint(x: r.midX, y: r.midY))
    }

    private func color(_ label: String) -> Color {
        Color(hue: labelHue(label), saturation: 0.8, brightness: 0.95)
    }
}
