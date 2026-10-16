import CoreGraphics

struct TrackedObject: Identifiable, Sendable {
    let id: Int
    var rect: CGRect
    var label: String
    var confidence: Float
    var lastSeen: Double
    var mask: Mask?
    var distance: Float?
    var x: Float = 0
    var width: Float = 0
    var height: Float = 0
}

struct Blob: Identifiable, Sendable {
    let id: Int
    var rect: CGRect
    var distance: Float
    var x: Float
    var width: Float
    var cells: Int
}

struct Analysis: Sendable {
    var objects: [TrackedObject] = []
    var blobs: [Blob] = []
    var scan: [Float] = []
    var labels: [UInt8] = []
    var gridW = 0
    var gridH = 0
    var center: Float?
}

final class Analyzer {
    private var tracks: [TrackedObject] = []
    private var nextID = 1
    private var lastSeq = -1

    func analyze(_ f: StereoFrame, detections: [Detection], seq: Int, now: Double) -> Analysis {
        if seq != lastSeq {
            lastSeq = seq
            associate(detections, now)
        }
        tracks.removeAll { now - $0.lastSeen > 0.7 }
        for i in tracks.indices { measure(&tracks[i], f) }

        var a = Analysis()
        a.objects = tracks
        a.center = f.sample(CGRect(x: 0.475, y: 0.455, width: 0.05, height: 0.09), q: 0.5)
        grid(f, &a)
        return a
    }

    private func associate(_ dets: [Detection], _ now: Double) {
        var used = Set<Int>()
        for d in dets {
            var bi = -1
            var bv: CGFloat = 0.25
            for (i, t) in tracks.enumerated() where !used.contains(i) && t.label == d.label {
                let v = iou(t.rect, d.rect)
                if v > bv { bv = v; bi = i }
            }
            if bi >= 0 {
                let o = tracks[bi].rect
                tracks[bi].rect = CGRect(x: o.minX + (d.rect.minX - o.minX) * 0.6, y: o.minY + (d.rect.minY - o.minY) * 0.6,
                                         width: o.width + (d.rect.width - o.width) * 0.6, height: o.height + (d.rect.height - o.height) * 0.6)
                tracks[bi].confidence = d.confidence
                tracks[bi].mask = d.mask
                tracks[bi].lastSeen = now
                used.insert(bi)
            } else {
                tracks.append(TrackedObject(id: nextID, rect: d.rect, label: d.label, confidence: d.confidence, lastSeen: now, mask: d.mask))
                used.insert(tracks.count - 1)
                nextID += 1
            }
        }
    }

    private func measure(_ t: inout TrackedObject, _ f: StereoFrame) {
        let inner = t.rect.insetBy(dx: t.rect.width * 0.2, dy: t.rect.height * 0.2)
        guard let z = t.mask.flatMap({ f.sample($0) }) ?? f.sample(inner, q: 0.3) else { return }
        let d: Float = if let p = t.distance, abs(p - z) < 0.4 * p { p + (z - p) * 0.35 } else { z }
        t.distance = d
        t.x = (Float(t.rect.midX) * Float(f.width) - f.K.cx) * d / f.K.fx
        t.width = Float(t.rect.width) * Float(f.width) * d / f.K.fx
        t.height = Float(t.rect.height) * Float(f.height) * d / f.K.fy
    }

    private func grid(_ f: StereoFrame, _ a: inout Analysis) {
        let step = max(1, f.width / 160)
        let gw = f.width / step, gh = f.height / step
        let n = gw * gh
        var g = [Float](repeating: 0, count: n)
        for gy in 0..<gh {
            for gx in 0..<gw {
                var s: Float = 0, c = 0
                for yy in 0..<step {
                    let r = f.row(gy * step + yy)
                    for xx in 0..<step {
                        let v = r[gx * step + xx]
                        if v > 0 { s += v; c += 1 }
                    }
                }
                if c * 2 >= step * step { g[gy * gw + gx] = s / Float(c) }
            }
        }

        var scan = [Float](repeating: 0, count: gw)
        for gx in 0..<gw {
            var m = Float.infinity
            for gy in (gh * 3 / 10)..<(gh * 7 / 10) {
                let v = g[gy * gw + gx]
                if v > 0 && v < m { m = v }
            }
            scan[gx] = m.isFinite ? m : 0
        }

        struct Comp { var id: Int32; var count = 0; var sumZ: Float = 0; var sumX = 0; var x0 = Int.max, y0 = Int.max, x1 = 0, y1 = 0 }
        var lab = [Int32](repeating: 0, count: n)
        var comps: [Comp] = []
        var stack: [Int] = []
        var cid: Int32 = 0
        for i in 0..<n where g[i] > 0 && lab[i] == 0 {
            cid += 1
            lab[i] = cid
            stack.append(i)
            var c = Comp(id: cid)
            while let p = stack.popLast() {
                let z = g[p], px = p % gw, py = p / gw
                c.count += 1
                c.sumZ += z
                c.sumX += px
                c.x0 = min(c.x0, px); c.x1 = max(c.x1, px)
                c.y0 = min(c.y0, py); c.y1 = max(c.y1, py)
                let tol = 0.02 + 0.04 * z
                for (q, ok) in [(p - 1, px > 0), (p + 1, px < gw - 1), (p - gw, py > 0), (p + gw, py < gh - 1)] where ok {
                    if lab[q] == 0, g[q] > 0, abs(g[q] - z) < tol {
                        lab[q] = cid
                        stack.append(q)
                    }
                }
            }
            comps.append(c)
        }

        let kept = comps.filter { $0.count >= 40 }.sorted { $0.count > $1.count }.prefix(24)
        var remap = [Int32: UInt8]()
        for (k, c) in kept.enumerated() {
            let id = UInt8(k + 1)
            remap[c.id] = id
            let z = c.sumZ / Float(c.count)
            let u = (Float(c.sumX) / Float(c.count) + 0.5) * Float(step)
            let wpx = Float(c.x1 - c.x0 + 1) * Float(step)
            a.blobs.append(Blob(id: Int(id),
                                rect: CGRect(x: CGFloat(c.x0) / CGFloat(gw), y: CGFloat(c.y0) / CGFloat(gh),
                                             width: CGFloat(c.x1 - c.x0 + 1) / CGFloat(gw), height: CGFloat(c.y1 - c.y0 + 1) / CGFloat(gh)),
                                distance: z, x: (u - f.K.cx) * z / f.K.fx, width: wpx * z / f.K.fx, cells: c.count))
        }
        a.labels = lab.map { remap[$0] ?? 0 }
        a.gridW = gw
        a.gridH = gh
        a.scan = scan
    }
}
