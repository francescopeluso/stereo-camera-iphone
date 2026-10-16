#include <metal_stdlib>
using namespace metal;

struct RectParams { float3x3 M; float2 srcSize; };
struct MatchParams { int maxDisp; int dir; float uniqueness; float maxCost; };
struct DepthParams { float fB; float minZ; float maxZ; int pad; };
struct DisplayParams { int mode; float minZ; float maxZ; float alpha; int hasRight; int hasLabels; int pad0; int pad1; };
struct PCParams { float4x4 mvp; float fx; float fy; float cx; float cy; float minZ; float maxZ; float pointSize; uint w; uint h; uint step; };

constant float3 kLuma = float3(0.299, 0.587, 0.114);

kernel void rectify(texture2d<float, access::sample> src [[texture(0)]],
                    texture2d<float, access::write> dst [[texture(1)]],
                    constant RectParams& p [[buffer(0)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(address::clamp_to_zero, filter::linear, coord::normalized);
    float4 acc = 0;
    for (int i = 0; i < 4; i++) {
        float2 o = float2((i & 1) ? 0.75 : 0.25, (i >> 1) ? 0.75 : 0.25);
        float3 q = p.M * float3(float2(gid) + o, 1.0);
        acc += src.sample(s, (q.xy / q.z) / p.srcSize);
    }
    dst.write(acc * 0.25, gid);
}

kernel void census(texture2d<float, access::read> src [[texture(0)]],
                   texture2d<uint, access::write> dst [[texture(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
    int w = src.get_width(), h = src.get_height();
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    float c = dot(src.read(gid).rgb, kLuma) - 0.004;
    uint2 bits = 0;
    uint k = 0;
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            if (dx == 0 && dy == 0) continue;
            uint2 q = uint2(clamp(int(gid.x) + dx, 0, w - 1), clamp(int(gid.y) + dy, 0, h - 1));
            uint b = dot(src.read(q).rgb, kLuma) < c ? 1u : 0u;
            if (k < 32) bits.x |= b << k; else bits.y |= b << (k - 32);
            k++;
        }
    }
    dst.write(uint4(bits, 0, 0), gid);
}

kernel void match(texture2d<uint, access::read> cl [[texture(0)]],
                  texture2d<uint, access::read> cr [[texture(1)]],
                  texture2d<float, access::write> disp [[texture(2)]],
                  texture2d<float, access::write> cost [[texture(3)]],
                  constant MatchParams& p [[buffer(0)]],
                  uint2 gid [[thread_position_in_grid]]) {
    const int R = 2;
    const int N = 96;
    int w = cl.get_width(), h = cl.get_height();
    int x = gid.x, y = gid.y;
    if (x >= w || y >= h) return;
    if (y < R || y >= h - R || x < R || x >= w - R) {
        disp.write(0.0, gid);
        cost.write(-1.0, gid);
        return;
    }
    uint2 L[25];
    for (int j = 0; j < 5; j++)
        for (int i = 0; i < 5; i++)
            L[j * 5 + i] = cl.read(uint2(x + i - R, y + j - R)).xy;

    ushort C[N];
    int nd = min(p.maxDisp, N);
    int count = 0;
    for (int d = 0; d < nd; d++) {
        int xr = x - p.dir * d;
        if (xr < R || xr >= w - R) break;
        uint s = 0;
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 5; i++) {
                uint2 r = cr.read(uint2(xr + i - R, y + j - R)).xy;
                uint2 l = L[j * 5 + i];
                s += popcount(l.x ^ r.x) + popcount(l.y ^ r.y);
            }
        }
        C[d] = ushort(s);
        count = d + 1;
    }
    if (count < 3) {
        disp.write(0.0, gid);
        cost.write(-1.0, gid);
        return;
    }
    int bd = 0;
    for (int d = 1; d < count; d++) if (C[d] < C[bd]) bd = d;
    uint second = 65535;
    for (int d = 0; d < count; d++) if (abs(d - bd) > 1) second = min(second, uint(C[d]));
    float best = float(C[bd]);
    cost.write(count == nd ? best / p.maxCost : -1.0, gid);
    bool ok = bd > 0 && bd < count - 1 && best * (1.0 + p.uniqueness) < float(second);
    float sub = 0;
    if (ok) {
        float c0 = float(C[bd - 1]), c2 = float(C[bd + 1]);
        float den = c0 - 2.0 * best + c2;
        if (den > 0) sub = clamp(0.5 * (c0 - c2) / den, -0.5, 0.5);
    }
    disp.write(ok ? float(bd) + sub : 0.0, gid);
}

struct SGMParams { int W; int H; int D; int dir; int dx; int dy; uint P1; uint P2; int first; float uniqueness; int lrCheck; int pad; };

kernel void costVolume(texture2d<uint, access::read> cl [[texture(0)]],
                       texture2d<uint, access::read> cr [[texture(1)]],
                       device uchar* C [[buffer(0)]],
                       constant SGMParams& p [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;
    uint2 l = cl.read(gid).xy;
    device uchar* out = C + (uint(y) * p.W + x) * p.D;
    for (int d = 0; d < p.D; d++) {
        int xr = x - p.dir * d;
        if (xr < 0 || xr >= p.W) { out[d] = 48; continue; }
        uint2 r = cr.read(uint2(xr, y)).xy;
        out[d] = uchar(popcount(l.x ^ r.x) + popcount(l.y ^ r.y));
    }
}

kernel void sgmPath(device const uchar* C [[buffer(0)]],
                    device ushort* S [[buffer(1)]],
                    constant SGMParams& p [[buffer(2)]],
                    texture2d<float, access::read> img [[texture(0)]],
                    uint d [[thread_position_in_threadgroup]],
                    uint line [[threadgroup_position_in_grid]],
                    uint sid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint nsimd [[simdgroups_per_threadgroup]]) {
    threadgroup uint prev[98];
    threadgroup uint mins[4];
    threadgroup uint prevMin;
    int x, y, len;
    if (p.dx != 0) { y = line; x = p.dx > 0 ? 0 : p.W - 1; len = p.W; }
    else { x = line; y = p.dy > 0 ? 0 : p.H - 1; len = p.H; }
    if (d == 0) { prev[0] = 0x3FFFFFFF; prev[p.D + 1] = 0x3FFFFFFF; }
    float Ip = 0;
    for (int s = 0; s < len; s++) {
        uint idx = (uint(y) * p.W + x) * p.D + d;
        uint c = C[idx];
        float I = dot(img.read(uint2(x, y)).rgb, kLuma);
        uint L = c;
        if (s > 0) {
            uint pm = prevMin;
            uint P2 = max(p.P1 + 1, uint(float(p.P2) / (1.0 + abs(I - Ip) * 16.0)));
            uint m = min(min(prev[d + 1], min(prev[d], prev[d + 2]) + p.P1), pm + P2);
            L = c + m - pm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prev[d + 1] = L;
        uint sm = simd_min(L);
        if (lane == 0) mins[sid] = sm;
        S[idx] = p.first ? ushort(L) : ushort(min(uint(S[idx]) + L, 65535u));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d == 0) {
            uint mm = mins[0];
            for (uint i = 1; i < nsimd; i++) mm = min(mm, mins[i]);
            prevMin = mm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Ip = I;
        x += p.dx;
        y += p.dy;
    }
}

kernel void wtaRight(device const ushort* S [[buffer(0)]],
                     texture2d<float, access::write> dispR [[texture(0)]],
                     constant SGMParams& p [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;
    uint best = 0xFFFFFFFF;
    int bd = 0;
    for (int d = 0; d < p.D; d++) {
        int xl = x + p.dir * d;
        if (xl < 0 || xl >= p.W) break;
        uint c = S[(uint(y) * p.W + xl) * p.D + d];
        if (c < best) { best = c; bd = d; }
    }
    dispR.write(float(bd), gid);
}

kernel void wtaLeft(device const ushort* S [[buffer(0)]],
                    texture2d<float, access::read> dispR [[texture(0)]],
                    texture2d<float, access::write> disp [[texture(1)]],
                    constant SGMParams& p [[buffer(1)]],
                    uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;
    device const ushort* s = S + (uint(y) * p.W + x) * p.D;
    int count = 0;
    for (int d = 0; d < p.D; d++) {
        int xr = x - p.dir * d;
        if (xr < 0 || xr >= p.W) break;
        count = d + 1;
    }
    if (count < 3) { disp.write(0.0, gid); return; }
    int bd = 0;
    for (int d = 1; d < count; d++) if (s[d] < s[bd]) bd = d;
    uint second = 0xFFFFFFFF;
    for (int d = 0; d < count; d++) if (abs(d - bd) > 1) second = min(second, uint(s[d]));
    float best = float(s[bd]);
    bool ok = bd > 0 && bd < count - 1 && best * (1.0 + p.uniqueness) < float(second);
    if (ok && p.lrCheck) {
        float dr = dispR.read(uint2(x - p.dir * bd, y)).r;
        ok = abs(dr - float(bd)) <= 1.0;
    }
    float sub = 0;
    if (ok) {
        float c0 = float(s[bd - 1]), c2 = float(s[bd + 1]);
        float den = c0 - 2.0 * best + c2;
        if (den > 0) sub = clamp(0.5 * (c0 - c2) / den, -0.5, 0.5);
    }
    disp.write(ok ? float(bd) + sub : 0.0, gid);
}

kernel void toDepth(texture2d<float, access::read> disp [[texture(0)]],
                    texture2d<float, access::write> depth [[texture(1)]],
                    constant DepthParams& p [[buffer(0)]],
                    uint2 gid [[thread_position_in_grid]]) {
    int w = disp.get_width(), h = disp.get_height();
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    float v[9];
    int n = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uint2 q = uint2(clamp(int(gid.x) + dx, 0, w - 1), clamp(int(gid.y) + dy, 0, h - 1));
            float d = disp.read(q).r;
            if (d > 0) v[n++] = d;
        }
    }
    float z = 0;
    if (n >= 5) {
        for (int i = 1; i < n; i++) {
            float k = v[i];
            int j = i - 1;
            while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; }
            v[j + 1] = k;
        }
        z = p.fB / v[n / 2];
        if (z < p.minZ || z > p.maxZ) z = 0;
    }
    depth.write(float4(z), gid);
}

float3 turbo(float x) {
    const float4 kr4 = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const float4 kg4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const float4 kb4 = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const float2 kr2 = float2(-152.94239396, 59.28637943);
    const float2 kg2 = float2(4.27729857, 2.82956604);
    const float2 kb2 = float2(-89.90310912, 27.34824973);
    x = saturate(x);
    float4 v4 = float4(1.0, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;
    return float3(dot(v4, kr4) + dot(v2, kr2), dot(v4, kg4) + dot(v2, kg2), dot(v4, kb4) + dot(v2, kb2));
}

float3 palette(uint id) {
    float h = fract(float(id) * 0.61803398875);
    float3 k = fract(float3(h) + float3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0;
    return mix(float3(1.0), saturate(abs(k) - 1.0), 0.8);
}

struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut fullscreenVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

fragment float4 displayFragment(VOut in [[stage_in]],
                                texture2d<float> color [[texture(0)]],
                                texture2d<float> right [[texture(1)]],
                                texture2d<float> depth [[texture(2)]],
                                texture2d<float> labels [[texture(3)]],
                                constant DisplayParams& p [[buffer(0)]]) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    constexpr sampler near(filter::nearest, address::clamp_to_edge);
    float2 uv = in.uv;
    float3 c = color.sample(lin, uv).rgb;
    float z = depth.sample(near, uv).r;
    bool valid = z > 0;
    float3 heat = turbo(1.0 - saturate((z - p.minZ) / (p.maxZ - p.minZ)));
    switch (p.mode) {
        case 0:
            return float4(c, 1);
        case 1:
            return float4(valid ? heat : float3(0), 1);
        case 2:
            return float4(valid ? mix(c, heat, p.alpha) : c * 0.7, 1);
        case 3: {
            float g = valid ? saturate((1.0 / z - 1.0 / p.maxZ) / (1.0 / p.minZ - 1.0 / p.maxZ)) : 0.0;
            return float4(g, g, g, 1);
        }
        case 4: {
            if (uv.y < 0.25 || uv.y > 0.75) return float4(0.04, 0.04, 0.05, 1);
            float2 q = float2(fract(uv.x * 2.0), (uv.y - 0.25) * 2.0);
            float3 img = uv.x < 0.5 ? color.sample(lin, q).rgb : (p.hasRight ? right.sample(lin, q).rgb : float3(0));
            float line = fract(q.y * 12.0) < 0.012 ? 1.0 : 0.0;
            img = mix(img, float3(0.1, 1.0, 0.35), line * 0.85);
            if (abs(uv.x - 0.5) < 0.0015) img = float3(1);
            return float4(img, 1);
        }
        case 5: {
            float l = dot(c, kLuma);
            float r = p.hasRight ? dot(right.sample(lin, uv).rgb, kLuma) : l;
            return float4(l, r, r, 1);
        }
        case 6: {
            float g = dot(c, kLuma);
            uint id = p.hasLabels ? uint(labels.sample(near, uv).r * 255.0 + 0.5) : 0;
            if (id == 0) return float4(float3(g * 0.4), 1);
            return float4(mix(float3(g), palette(id), 0.6), 1);
        }
        default:
            return float4(c * 0.25, 1);
    }
}

struct PCOut { float4 pos [[position]]; float3 color; float size [[point_size]]; };

vertex PCOut pointVertex(uint vid [[vertex_id]],
                         texture2d<float> depth [[texture(0)]],
                         texture2d<float> color [[texture(1)]],
                         constant PCParams& p [[buffer(0)]]) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    uint gw = p.w / p.step;
    uint2 px = uint2(vid % gw, vid / gw) * p.step;
    float z = depth.read(px).r;
    PCOut o;
    o.size = p.pointSize;
    if (z <= 0) {
        o.pos = float4(0, 0, 2, 1);
        o.color = 0;
        return o;
    }
    float2 f = float2(px) + 0.5;
    float3 P = float3((f.x - p.cx) * z / p.fx, -(f.y - p.cy) * z / p.fy, -z);
    o.pos = p.mvp * float4(P, 1);
    o.color = color.sample(lin, f / float2(p.w, p.h)).rgb;
    return o;
}

fragment float4 pointFragment(PCOut in [[stage_in]]) {
    return float4(in.color, 1);
}
