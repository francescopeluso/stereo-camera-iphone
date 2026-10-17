import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/out")
OUT.mkdir(parents=True, exist_ok=True)

W, H, SS = 1280, 720, 2
F = 940.0
B = 0.010
CAM_Y = 0.45
PITCH = np.radians(18)
FWD = np.array([0, -np.sin(PITCH), np.cos(PITCH)])
UP = np.array([0, np.cos(PITCH), np.sin(PITCH)])
LIGHT = np.array([-0.45, 1.0, -0.35]) / np.linalg.norm([-0.45, 1.0, -0.35])
rng = np.random.default_rng(7)
GRID = rng.random((512, 512)).astype(np.float32)

LABELS = {2: "laptop", 3: "cup", 4: "bottle", 5: "orange", 6: "book"}


def noise(x, y, s):
    gx, gy = x * s, y * s
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - x0, gy - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = GRID[y0 % 512, x0 % 512], GRID[y0 % 512, (x0 + 1) % 512]
    c, d = GRID[(y0 + 1) % 512, x0 % 512], GRID[(y0 + 1) % 512, (x0 + 1) % 512]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, s, o=4):
    return sum(noise(x + 13.1 * i, y - 7.7 * i, s * 2**i) / 2**i for i in range(o)) / (2 - 2 ** (1 - o))


def plane_y(o, d, y0):
    t = (y0 - o[..., 1]) / np.where(np.abs(d[..., 1]) < 1e-9, 1e-9, d[..., 1])
    return np.where(t > 1e-4, t, np.inf), np.broadcast_to(np.array([0, 1.0, 0]), d.shape)


def plane_z(o, d, z0):
    t = (z0 - o[..., 2]) / np.where(np.abs(d[..., 2]) < 1e-9, 1e-9, d[..., 2])
    return np.where(t > 1e-4, t, np.inf), np.broadcast_to(np.array([0, 0, -1.0]), d.shape)


def box(o, d, lo, hi):
    inv = 1 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t0, t1 = (np.array(lo) - o) * inv, (np.array(hi) - o) * inv
    tmin, tmax = np.minimum(t0, t1), np.maximum(t0, t1)
    tn, tf = tmin.max(-1), tmax.min(-1)
    hit = (tn <= tf) & (tf > 1e-4)
    t = np.where(hit, np.where(tn > 1e-4, tn, tf), np.inf)
    axis = tmin.argmax(-1)
    n = np.zeros_like(d)
    np.put_along_axis(n, axis[..., None], -np.sign(np.take_along_axis(d, axis[..., None], -1)), -1)
    return t, n


def cylinder(o, d, cx, cz, r, y0, y1):
    ox, oz = o[..., 0] - cx, o[..., 2] - cz
    a = d[..., 0] ** 2 + d[..., 2] ** 2
    b = 2 * (ox * d[..., 0] + oz * d[..., 2])
    c = ox * ox + oz * oz - r * r
    disc = b * b - 4 * a * c
    sq = np.sqrt(np.maximum(disc, 0))
    t = (-b - sq) / (2 * np.maximum(a, 1e-12))
    y = o[..., 1] + t * d[..., 1]
    side = (disc > 0) & (t > 1e-4) & (y >= y0) & (y <= y1)
    ts = np.where(side, t, np.inf)
    tc = (y1 - o[..., 1]) / np.where(np.abs(d[..., 1]) < 1e-9, 1e-9, d[..., 1])
    px, pz = o[..., 0] + tc * d[..., 0] - cx, o[..., 2] + tc * d[..., 2] - cz
    cap = (tc > 1e-4) & (px * px + pz * pz <= r * r)
    tc = np.where(cap, tc, np.inf)
    t = np.minimum(ts, tc)
    p = o + t[..., None] * d
    n = np.stack([p[..., 0] - cx, np.zeros_like(t), p[..., 2] - cz], -1) / r
    n = np.where((tc < ts)[..., None], np.array([0, 1.0, 0]), n)
    return t, n


def sphere(o, d, c, r):
    oc = o - np.array(c)
    a = (d * d).sum(-1)
    b = 2 * (d * oc).sum(-1)
    cc = (oc * oc).sum(-1) - r * r
    disc = b * b - 4 * a * cc
    t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    t = np.where((disc > 0) & (t > 1e-4), t, np.inf)
    n = (o + t[..., None] * d - np.array(c)) / r
    return t, n


def quad(o, d, p0, u, v):
    n = np.cross(v, u)
    n = n / np.linalg.norm(n)
    if n[2] > 0:
        n = -n
    den = (d * n).sum(-1)
    t = ((p0 - o) @ n) / np.where(np.abs(den) < 1e-9, 1e-9, den)
    p = o + t[..., None] * d - p0
    a, b = (p @ u) / (u @ u), (p @ v) / (v @ v)
    hit = (t > 1e-4) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
    return np.where(hit, t, np.inf), np.broadcast_to(n, d.shape), a, b


TILT = np.radians(14)
SCREEN = (np.array([-0.62, 0.02, 1.22]), np.array([0.40, 0, 0]), 0.26 * np.array([0, np.cos(TILT), np.sin(TILT)]))


def objects(o, d):
    return [
        (2, *box(o, d, (-0.62, 0, 0.95), (-0.22, 0.02, 1.22))),
        (2, *quad(o, d, *SCREEN)[:2]),
        (3, *cylinder(o, d, 0.02, 0.72, 0.042, 0, 0.095)),
        (4, *cylinder(o, d, 0.30, 1.05, 0.036, 0, 0.20)),
        (4, *cylinder(o, d, 0.30, 1.05, 0.014, 0.20, 0.265)),
        (5, *sphere(o, d, (-0.12, 0.042, 0.66), 0.042)),
        (6, *box(o, d, (0.14, 0, 0.58), (0.40, 0.032, 0.76))),
    ]


def trace(o, d):
    t = np.full(d.shape[:-1], np.inf)
    n = np.zeros_like(d)
    ids = np.full(d.shape[:-1], -1)
    for oid, (tt, nn) in [(0, plane_z(o, d, 1.9)), (1, plane_y(o, d, 0.0))] + [(i, (a, b)) for i, a, b in objects(o, d)]:
        m = tt < t
        t, ids = np.where(m, tt, t), np.where(m, oid, ids)
        n = np.where(m[..., None], nn, n)
    return t, n, ids


def shadow(p):
    flat = p.reshape(-1, 3)
    dd = np.broadcast_to(LIGHT, flat.shape)
    occ = np.zeros(len(flat), bool)
    for oid, lo, hi in [(2, (-0.62, 0, 0.95), (-0.22, 0.02, 1.22)), (6, (0.14, 0, 0.58), (0.40, 0.032, 0.76))]:
        occ |= np.isfinite(box(flat, dd, lo, hi)[0])
    occ |= np.isfinite(quad(flat, dd, *SCREEN)[0])
    occ |= np.isfinite(cylinder(flat, dd, 0.02, 0.72, 0.042, 0, 0.095)[0])
    occ |= np.isfinite(cylinder(flat, dd, 0.30, 1.05, 0.036, 0, 0.265)[0])
    occ |= np.isfinite(sphere(flat, dd, (-0.12, 0.042, 0.66), 0.042)[0])
    return ~occ.reshape(p.shape[:-1])


def albedo(ids, p, n):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    col = np.zeros(p.shape)
    emit = np.zeros(p.shape[:-1], bool)

    m = ids == 0
    paper = 0.78 + 0.16 * fbm(x, y, 9, 6)[..., None] + 0.08 * (noise(x, y, 160)[..., None] - 0.5)
    wall = np.array([0.80, 0.82, 0.86]) * paper
    pr = (x > 0.10) & (x < 0.52) & (y > 0.14) & (y < 0.48)
    rr = np.hypot(x - 0.31, y - 0.31)
    poster = np.stack([0.5 + 0.5 * np.sin(rr * 90), 0.35 + 0.3 * np.sin(rr * 90 + 2), 0.55 + 0.4 * np.cos(rr * 45)], -1) * 0.85
    pr2 = (x > -1.00) & (x < -0.66) & (y > 0.20) & (y < 0.44)
    grid = (((np.floor(x * 30) + np.floor(y * 30)) % 2)[..., None] * 0.5 + 0.35) * np.array([0.25, 0.5, 0.95])
    wall = np.where(pr[..., None], poster, np.where(pr2[..., None], grid, wall))
    col = np.where(m[..., None], wall, col)

    m = ids == 1
    plank = np.floor(z / 0.16)
    grain = np.sin(z * 260 + 9 * fbm(x * 1.5 + plank * 3.7, z * 6, 4) + plank * 2.1)
    seam = (np.abs((z / 0.16) % 1 - 0.5) > 0.49).astype(float)
    tone = 0.88 + 0.18 * noise(plank, 3.3, 1.0) + 0.10 * fbm(x * 3, z * 12, 10)
    knots = 0.12 * (noise(x * 2, z * 9, 30) - 0.5)
    wood = np.array([0.60, 0.41, 0.25])[None] * ((tone + knots) * (0.84 + 0.16 * grain) * (1 - 0.45 * seam))[..., None]
    col = np.where(m[..., None], wood, col)

    m = ids == 2
    keys = ((np.abs((x * 55) % 1 - 0.5) < 0.38) & (np.abs((z * 55) % 1 - 0.5) < 0.38) & (z > 1.0) & (z < 1.16) & (x > -0.58) & (x < -0.26) & (y > 0.019))
    alu = np.array([0.55, 0.57, 0.60]) * (0.85 + 0.2 * fbm(x, z, 40))[..., None]
    alu = np.where(keys[..., None], np.array([0.12, 0.12, 0.14]), alu)
    o, u, v = SCREEN
    rel = p - o
    a, b = (rel @ u) / (u @ u), (rel @ v) / (v @ v)
    on = (np.abs(rel @ (np.cross(v, u) / np.linalg.norm(np.cross(v, u)))) < 1e-3) & (a > 0.03) & (a < 0.97) & (b > 0.05) & (b < 0.95)
    row = np.floor(b * 26)
    line_len = (noise(row, row * 0.3, 0.9) * 0.7 + 0.1)
    indent = (noise(row * 3.1, 1.0, 0.5) * 4).astype(int) * 0.04
    code = (a > 0.06 + indent) & (a < 0.06 + indent + line_len) & ((b * 26) % 1 < 0.55)
    hue = noise(row, 5.0, 1.3)
    code_col = np.stack([0.4 + 0.6 * hue, 0.75 - 0.3 * hue, 1.0 - 0.6 * hue], -1)
    scr = np.where(code[..., None], code_col, np.array([0.07, 0.08, 0.13]))
    col = np.where(m[..., None], np.where(on[..., None], scr, alu), col)
    emit |= m & on

    m = ids == 3
    mug = np.array([0.78, 0.13, 0.16]) * (0.85 + 0.25 * fbm(x * 3, y * 3, 30))[..., None]
    top = (n[..., 1] > 0.9) & (np.hypot(x - 0.02, z - 0.72) < 0.036)
    mug = np.where(top[..., None], np.array([0.22, 0.12, 0.06]), mug)
    col = np.where(m[..., None], mug, col)

    m = ids == 4
    ang = np.arctan2(z - 1.05, x - 0.30)
    label = (y > 0.06) & (y < 0.14)
    stripes = (np.sin(ang * 14) > 0.2)
    bottle = np.where(label[..., None], np.where(stripes[..., None], np.array([0.95, 0.95, 0.92]), np.array([0.1, 0.35, 0.75])), np.array([0.12, 0.55, 0.52]) * (0.85 + 0.25 * fbm(ang, y, 6))[..., None])
    col = np.where(m[..., None], bottle, col)

    m = ids == 5
    orange = np.array([0.98, 0.52, 0.08]) * (0.8 + 0.3 * fbm(x * 4, y * 4 + z * 4, 60))[..., None]
    col = np.where(m[..., None], orange, col)

    m = ids == 6
    pages = (y < 0.03) & (n[..., 1] < 0.5)
    cover = np.array([0.16, 0.26, 0.62]) * (0.85 + 0.25 * fbm(x, z, 25))[..., None]
    title = (n[..., 1] > 0.5) & (x > 0.18) & (x < 0.36) & (z > 0.64) & (z < 0.70)
    cover = np.where(title[..., None], np.array([0.95, 0.80, 0.25]), cover)
    pg = np.array([0.93, 0.91, 0.86]) * (0.9 + 0.1 * (np.sin(y * 2500) > 0))[..., None]
    col = np.where(m[..., None], np.where(pages[..., None], pg, cover), col)
    return col, emit


def render(cam_x):
    w, h = W * SS, H * SS
    rgb = np.zeros((h, w, 3), np.float32)
    depth = np.zeros((h, w), np.float32)
    ids = np.zeros((h, w), np.int16)
    o = np.array([cam_x, CAM_Y, 0.0])
    us = (np.arange(w) + 0.5) / SS
    for r0 in range(0, h, 180):
        vs = (np.arange(r0, min(h, r0 + 180)) + 0.5) / SS
        U, V = np.meshgrid(us, vs)
        dx, dy = (U - W / 2) / F, (V - H / 2) / F
        d = dx[..., None] * np.array([1.0, 0, 0]) + (-dy)[..., None] * UP + FWD
        t, n, oid = trace(o, d)
        p = o + t[..., None] * d
        col, emit = albedo(oid, p, n)
        lit = shadow(p + n * 1e-3)
        diff = np.clip((n * LIGHT).sum(-1), 0, 1) * lit
        view = -d / np.linalg.norm(d, axis=-1, keepdims=True)
        hv = LIGHT + view
        hv /= np.linalg.norm(hv, axis=-1, keepdims=True)
        spec = np.clip((n * hv).sum(-1), 0, 1) ** 40 * lit * np.isin(oid, [2, 3, 4, 5]) * 0.35
        amb = 0.30 + 0.12 * n[..., 1]
        shade = col * (amb + 0.72 * diff)[..., None] + spec[..., None]
        shade = np.where(emit[..., None], col * 1.05, shade)
        fog = np.clip((t - 1.0) / 6, 0, 0.12)[..., None]
        shade = shade * (1 - fog) + fog * np.array([0.75, 0.78, 0.85])
        rgb[r0:r0 + len(vs)] = shade
        depth[r0:r0 + len(vs)] = t
        ids[r0:r0 + len(vs)] = oid
    rgb = rgb.reshape(H, SS, W, SS, 3).mean((1, 3))
    rgb = np.clip(rgb, 0, 1) ** (1 / 1.15)
    rgb = rgb + rng.normal(0, 0.005, rgb.shape)
    return np.clip(rgb, 0, 1), depth[SS // 2::SS, SS // 2::SS], ids[SS // 2::SS, SS // 2::SS]


def save(a, name):
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(OUT / name)


left, depth, ids = render(0.0)
right, _, _ = render(B)
save(left, "left_1280.png")
save(right, "right_1280.png")
for img, name in [(left, "left_640.png"), (right, "right_640.png")]:
    save(img.reshape(360, 2, 640, 2, 3).mean((1, 3)), name)
np.save(OUT / "depth_gt.npy", depth)
np.save(OUT / "ids.npy", ids)
json.dump({"f_1280": F, "baseline_m": B, "labels": LABELS}, open(OUT / "scene.json", "w"))
print("rendered", left.shape, "depth range", float(depth[np.isfinite(depth)].min()), float(depth[np.isfinite(depth)].max()))
