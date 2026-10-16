import Accelerate
import CoreML
import UIKit
import Vision

struct Mask: @unchecked Sendable {
    let rect: CGRect
    let w: Int, h: Int
    let bits: [UInt8]
    let image: CGImage
}

struct Detection: Sendable {
    var rect: CGRect
    var label: String
    var confidence: Float
    var mask: Mask?
}

func labelHue(_ label: String) -> Double {
    let h = label.unicodeScalars.reduce(UInt32(7)) { ($0 &* 31) &+ $1.value }
    return Double(h % 360) / 360
}

final class Detector: @unchecked Sendable {
    let name: String
    private let queue = DispatchQueue(label: "detector", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var results: [Detection] = []
    private var seq = 0
    private let request: VNCoreMLRequest?
    private let segment: Bool
    private let names: [String]

    init() {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        func load(_ n: String) -> MLModel? {
            Bundle.main.url(forResource: n, withExtension: "mlmodelc").flatMap { try? MLModel(contentsOf: $0, configuration: cfg) }
        }
        if let ml = load("yolo11n-seg"), let vn = try? VNCoreMLModel(for: ml) {
            request = VNCoreMLRequest(model: vn)
            segment = true
            names = Self.classNames(ml)
            name = "YOLO11n-seg · 80 COCO classes + masks"
        } else if let ml = load("yolo11n"), let vn = try? VNCoreMLModel(for: ml) {
            request = VNCoreMLRequest(model: vn)
            segment = false
            names = []
            name = "YOLO11n · 80 COCO classes"
        } else {
            request = nil
            segment = false
            names = []
            name = "Vision built-in (people, animals, saliency)"
        }
        request?.imageCropAndScaleOption = .scaleFill
    }

    private static func classNames(_ ml: MLModel) -> [String] {
        let meta = ml.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        guard let raw = meta?["names"] else { return [] }
        let pairs = raw.matches(of: #/(\d+): '([^']*)'/#).map { (Int($0.output.1) ?? 0, String($0.output.2)) }
        var out = [String](repeating: "object", count: (pairs.map(\.0).max() ?? -1) + 1)
        for (i, n) in pairs { out[i] = n }
        return out
    }

    var latest: (seq: Int, items: [Detection]) {
        lock.withLock { (seq, results) }
    }

    func submit(_ pb: CVPixelBuffer, minConfidence: Float) {
        let go: Bool = lock.withLock {
            if busy { return false }
            busy = true
            return true
        }
        guard go else { return }
        queue.async { [self] in
            let r: [Detection]
            if let request {
                try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up).perform([request])
                r = segment ? decodeSeg(request, minConfidence) : decodeBoxes(request, minConfidence)
            } else {
                r = runBuiltin(pb, minConfidence)
            }
            lock.withLock {
                results = r
                seq += 1
                busy = false
            }
        }
    }

    private static func topLeft(_ r: CGRect) -> CGRect {
        CGRect(x: r.minX, y: 1 - r.maxY, width: r.width, height: r.height)
    }

    private func decodeBoxes(_ req: VNCoreMLRequest, _ minConf: Float) -> [Detection] {
        (req.results as? [VNRecognizedObjectObservation] ?? []).compactMap { o in
            guard let l = o.labels.first, l.confidence >= minConf else { return nil }
            return Detection(rect: Self.topLeft(o.boundingBox), label: l.identifier, confidence: l.confidence)
        }
    }

    private func decodeSeg(_ req: VNCoreMLRequest, _ minConf: Float) -> [Detection] {
        let arrays = (req.results as? [VNCoreMLFeatureValueObservation] ?? []).compactMap { $0.featureValue.multiArrayValue }
        guard let pred = arrays.first(where: { $0.shape.count == 3 }), let proto = arrays.first(where: { $0.shape.count == 4 }) else { return [] }
        let ch = pred.shape[1].intValue, n = pred.shape[2].intValue
        let nm = proto.shape[1].intValue, ph = proto.shape[2].intValue, pw = proto.shape[3].intValue
        let nc = ch - 4 - nm
        let s1 = pred.strides[1].intValue, s2 = pred.strides[2].intValue
        let read = reader(pred)

        var cands: [(i: Int, c: Int, s: Float)] = []
        for i in 0..<n {
            var best: Float = 0, bc = 0
            for c in 0..<nc {
                let v = read((4 + c) * s1 + i * s2)
                if v > best { best = v; bc = c }
            }
            if best >= minConf { cands.append((i, bc, best)) }
        }
        cands.sort { $0.s > $1.s }

        var kept: [(i: Int, c: Int, s: Float, r: CGRect)] = []
        for k in cands {
            let cx = read(k.i * s2), cy = read(s1 + k.i * s2), w = read(2 * s1 + k.i * s2), h = read(3 * s1 + k.i * s2)
            let r = CGRect(x: CGFloat(cx - w / 2) / 640, y: CGFloat(cy - h / 2) / 640, width: CGFloat(w) / 640, height: CGFloat(h) / 640)
                .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard !r.isNull, r.width > 0, r.height > 0 else { continue }
            if kept.contains(where: { $0.c == k.c && iou($0.r, r) > 0.45 }) { continue }
            kept.append((k.i, k.c, k.s, r))
            if kept.count == 30 { break }
        }
        guard !kept.isEmpty else { return [] }

        var coeffs = [Float](repeating: 0, count: kept.count * nm)
        for (j, k) in kept.enumerated() {
            for m in 0..<nm { coeffs[j * nm + m] = read((4 + nc + m) * s1 + k.i * s2) }
        }
        let protoDense = dense(proto, nm, ph, pw)
        var logits = [Float](repeating: 0, count: kept.count * ph * pw)
        vDSP_mmul(coeffs, 1, protoDense, 1, &logits, 1, vDSP_Length(kept.count), vDSP_Length(ph * pw), vDSP_Length(nm))

        return kept.enumerated().map { j, k in
            let label = k.c < names.count ? names[k.c] : "class \(k.c)"
            let x0 = max(0, Int(k.r.minX * CGFloat(pw))), x1 = min(pw, Int(ceil(k.r.maxX * CGFloat(pw))))
            let y0 = max(0, Int(k.r.minY * CGFloat(ph))), y1 = min(ph, Int(ceil(k.r.maxY * CGFloat(ph))))
            var mask: Mask?
            if x1 > x0, y1 > y0 {
                let mw = x1 - x0, mh = y1 - y0
                var bits = [UInt8](repeating: 0, count: mw * mh)
                let base = j * ph * pw
                for y in 0..<mh {
                    for x in 0..<mw where logits[base + (y0 + y) * pw + x0 + x] > 0 { bits[y * mw + x] = 1 }
                }
                let rect = CGRect(x: CGFloat(x0) / CGFloat(pw), y: CGFloat(y0) / CGFloat(ph), width: CGFloat(mw) / CGFloat(pw), height: CGFloat(mh) / CGFloat(ph))
                mask = Self.makeMask(bits, mw, mh, rect, hue: labelHue(label))
            }
            return Detection(rect: k.r, label: label, confidence: k.s, mask: mask)
        }
    }

    private func reader(_ a: MLMultiArray) -> (Int) -> Float {
        let p = a.dataPointer
        if a.dataType == .float16 {
            return { Float(p.load(fromByteOffset: $0 * 2, as: Float16.self)) }
        }
        return { p.load(fromByteOffset: $0 * 4, as: Float.self) }
    }

    private func dense(_ a: MLMultiArray, _ c: Int, _ h: Int, _ w: Int) -> [Float] {
        let read = reader(a)
        let s = a.strides.map(\.intValue)
        var out = [Float](repeating: 0, count: c * h * w)
        for k in 0..<c {
            for y in 0..<h {
                let row = k * s[1] + y * s[2]
                let o = (k * h + y) * w
                for x in 0..<w { out[o + x] = read(row + x * s[3]) }
            }
        }
        return out
    }

    private static func makeMask(_ bits: [UInt8], _ w: Int, _ h: Int, _ rect: CGRect, hue: Double) -> Mask? {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0
        UIColor(hue: hue, saturation: 0.8, brightness: 0.95, alpha: 1).getRed(&r, green: &g, blue: &b, alpha: nil)
        let a: CGFloat = 0.45
        let px = [UInt8(r * a * 255), UInt8(g * a * 255), UInt8(b * a * 255), UInt8(a * 255)]
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) where bits[i] != 0 {
            rgba[i * 4] = px[0]; rgba[i * 4 + 1] = px[1]; rgba[i * 4 + 2] = px[2]; rgba[i * 4 + 3] = px[3]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        return Mask(rect: rect, w: w, h: h, bits: bits, image: img)
    }

    private func runBuiltin(_ pb: CVPixelBuffer, _ minConf: Float) -> [Detection] {
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
        let humans = VNDetectHumanRectanglesRequest()
        humans.upperBodyOnly = false
        let animals = VNRecognizeAnimalsRequest()
        let saliency = VNGenerateObjectnessBasedSaliencyImageRequest()
        try? handler.perform([humans, animals, saliency])

        var out: [Detection] = (humans.results ?? []).filter { $0.confidence >= minConf }.map {
            Detection(rect: Self.topLeft($0.boundingBox), label: "person", confidence: $0.confidence)
        }
        for a in animals.results ?? [] {
            guard let l = a.labels.first, l.confidence >= minConf else { continue }
            out.append(Detection(rect: Self.topLeft(a.boundingBox), label: l.identifier.lowercased(), confidence: l.confidence))
        }
        let salient = saliency.results?.first?.salientObjects ?? []
        for s in salient.prefix(4) {
            let r = Self.topLeft(s.boundingBox)
            guard !out.contains(where: { iou($0.rect, r) > 0.4 }) else { continue }
            let cls = VNClassifyImageRequest()
            cls.regionOfInterest = s.boundingBox
            try? handler.perform([cls])
            let top = (cls.results ?? []).first { $0.confidence > 0.12 }
            out.append(Detection(rect: r, label: top?.identifier.replacingOccurrences(of: "_", with: " ") ?? "object", confidence: top?.confidence ?? s.confidence))
        }
        return out
    }
}

func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let i = a.intersection(b)
    guard !i.isNull else { return 0 }
    let ia = i.width * i.height
    return ia / (a.width * a.height + b.width * b.height - ia)
}
