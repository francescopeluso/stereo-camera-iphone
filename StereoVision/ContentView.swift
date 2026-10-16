import MetalKit
import SwiftUI

@main
struct StereoVisionApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct MetalView: UIViewRepresentable {
    let renderer: Renderer
    func makeUIView(context: Context) -> MTKView {
        let v = MTKView()
        renderer.configure(v)
        return v
    }
    func updateUIView(_ v: MTKView, context: Context) {}
}

struct ContentView: View {
    @State private var model = AppModel()
    @State private var showSettings = false
    @State private var dragStart: (yaw: Float, pitch: Float)?
    @State private var zoomStart: Float?

    var body: some View {
        HStack(spacing: 0) {
            ZStack {
                Color.black
                GeometryReader { geo in
                    ZStack {
                        MetalView(renderer: model.renderer)
                        OverlayView(model: model)
                    }
                    .contentShape(Rectangle())
                    .gesture(SpatialTapGesture().onEnded { v in
                        if model.mode == .pointCloud { model.resetOrbit(); return }
                        model.measure(at: CGPoint(x: v.location.x / geo.size.width, y: v.location.y / geo.size.height))
                    })
                    .simultaneousGesture(orbitDrag, including: model.mode == .pointCloud ? .all : .none)
                    .simultaneousGesture(orbitZoom, including: model.mode == .pointCloud ? .all : .none)
                }
                .aspectRatio(model.aspect, contentMode: .fit)
                .overlay(alignment: .topLeading) { HUD(model: model).padding(8) }
            }
            SidePanel(model: model, showSettings: $showSettings)
        }
        .background(Color.black.ignoresSafeArea())
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
        .sheet(isPresented: $showSettings) { SettingsView(model: model) }
        .task { await model.start() }
    }

    private var orbitDrag: some Gesture {
        DragGesture(minimumDistance: 4)
            .onChanged { v in
                let s = dragStart ?? (model.yaw, model.pitch)
                dragStart = s
                model.yaw = s.yaw + Float(v.translation.width) * 0.008
                model.pitch = min(1.4, max(-1.4, s.pitch + Float(v.translation.height) * 0.008))
            }
            .onEnded { _ in dragStart = nil }
    }

    private var orbitZoom: some Gesture {
        MagnifyGesture()
            .onChanged { v in
                let s = zoomStart ?? model.orbit
                zoomStart = s
                model.orbit = min(8, max(0.2, s / Float(v.magnification)))
            }
            .onEnded { _ in zoomStart = nil }
    }
}

struct HUD: View {
    let model: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(model.mode.title).font(.headline)
            Text("\(Int(model.fps)) fps" + (model.source == .stereo ? String(format: " · GPU %.1f ms", model.gpuMs) : ""))
            if model.source == .stereo {
                Text(String(format: "B %.1f mm · f %.0f px · Δy %.2f px%@", model.baselineMM, model.K.fx, model.yOffset, model.flip ? " · ⇄" : ""))
                Text(model.matcher == .sgm ? "SGM-4 · D \(model.maxDisparity)" + (model.lrCheck ? " · LR" : "") : "Block 5×5 · D \(model.maxDisparity)")
            } else {
                Text("\(model.source.title) · \(Int(model.depthSize.width))×\(Int(model.depthSize.height)) · f \(Int(model.K.fx)) px")
            }
            if model.calibrating { Text("Epipolar auto-calibration…").foregroundStyle(.yellow) }
            if let s = model.status { Text(s).foregroundStyle(.red) }
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.white)
        .padding(6)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
        .allowsHitTesting(false)
    }
}

struct SidePanel: View {
    let model: AppModel
    @Binding var showSettings: Bool

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 6) {
                ForEach(ViewMode.allCases) { m in
                    let disabled = model.source != .stereo && m.needsStereoPair
                    button(m.icon, m.title, selected: model.mode == m) { model.mode = m }
                        .disabled(disabled)
                        .opacity(disabled ? 0.3 : 1)
                }
                Divider().background(.white.opacity(0.3)).padding(.vertical, 4)
                Button {
                    let all = DepthSource.available
                    model.setSource(all[((all.firstIndex(of: model.source) ?? 0) + 1) % all.count])
                } label: {
                    Text(model.source.short)
                        .font(.system(size: 10, weight: .heavy, design: .monospaced))
                        .frame(width: 50, height: 32)
                        .background(model.source == .stereo ? Color.orange : model.source == .lidar ? Color.green : Color.blue, in: RoundedRectangle(cornerRadius: 8))
                        .foregroundStyle(.white)
                }
                button(model.detection ? "viewfinder" : "viewfinder.trianglebadge.exclamationmark", "Detection", selected: model.detection) {
                    model.detection.toggle()
                }
                if model.source == .stereo {
                    button("wand.and.stars", "Calibrate", selected: model.calibrating) { model.calibrate() }
                }
                button("slider.horizontal.3", "Settings", selected: false) { showSettings = true }
            }
            .padding(.vertical, 10)
        }
        .frame(width: 62)
        .background(Color(white: 0.08))
    }

    private func button(_ icon: String, _ label: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .semibold))
                .frame(width: 50, height: 40)
                .background(selected ? Color.white.opacity(0.22) : .clear, in: RoundedRectangle(cornerRadius: 8))
                .foregroundStyle(selected ? .yellow : .white)
        }
        .accessibilityLabel(label)
    }
}

struct SettingsView: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Depth range") {
                    slider("Min", $model.minZ, 0.05...1, "%.2f m")
                    slider("Max", $model.maxZ, 0.5...8, "%.1f m")
                    slider("Overlay opacity", $model.overlayAlpha, 0...1, "%.2f")
                }
                Section("Stereo matching") {
                    Picker("Algorithm", selection: $model.matcher) {
                        ForEach(Matcher.allCases) { Text($0.title).tag($0) }
                    }
                    Picker("Max disparity", selection: $model.maxDisparity) {
                        ForEach([32, 64, 96], id: \.self) { Text("\($0) px").tag($0) }
                    }
                    .pickerStyle(.segmented)
                    if model.matcher == .sgm {
                        Toggle("Left-right consistency check", isOn: $model.lrCheck)
                        slider("P1 (1 px jumps)", $model.p1, 1...40, "%.0f")
                        slider("P2 (discontinuities)", $model.p2, 20...200, "%.0f")
                    }
                    slider("Uniqueness", $model.uniqueness, 0...0.5, "%.2f")
                    slider("Vertical offset Δy", $model.yOffset, -12...12, "%.2f px")
                    Toggle("Flip baseline (⇄)", isOn: $model.flip)
                    Button("Auto-calibrate rectification") { model.calibrate(); dismiss() }
                }
                .disabled(model.source != .stereo)
                Section("Object detection") {
                    Toggle("Enabled", isOn: $model.detection)
                    Toggle("Show masks", isOn: $model.showMasks)
                    slider("Min confidence", $model.confidence, 0.1...0.9, "%.2f")
                    LabeledContent("Model", value: model.detectorName)
                }
                Section("Source") {
                    Picker("Depth", selection: Binding(get: { model.source }, set: { model.setSource($0) })) {
                        ForEach(DepthSource.available) { Text($0.title).tag($0) }
                    }
                    LabeledContent("Baseline", value: String(format: "%.2f mm", model.baselineMM))
                    LabeledContent("Focal length", value: String(format: "%.1f px", model.K.fx))
                    LabeledContent("Principal point", value: String(format: "%.1f, %.1f", model.K.cx, model.K.cy))
                    LabeledContent("Depth map", value: "\(Int(model.depthSize.width))×\(Int(model.depthSize.height))")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Done") { dismiss() } }
        }
    }

    private func slider(_ title: String, _ v: Binding<Float>, _ r: ClosedRange<Float>, _ fmt: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: fmt, v.wrappedValue)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: v, in: r)
        }
    }
}
