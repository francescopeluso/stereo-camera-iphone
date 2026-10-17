import colorsys
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(sys.argv[1])
PUB = Path(sys.argv[2])
IMG = PUB / "img"
IMG.mkdir(parents=True, exist_ok=True)

W, H = 640, 360
FB = 470.0 * 0.010
MINZ, MAXZ = 0.3, 2.2
scene = json.load(open(SRC / "scene.json"))
LABELS = {int(k): v for k, v in scene["labels"].items()}

rgb = np.asarray(Image.open(SRC / "left_1280.png")).astype(np.float32) / 255
rgbR = np.asarray(Image.open(SRC / "right_1280.png")).astype(np.float32) / 255
rgb640 = np.asarray(Image.open(SRC / "left_640.png")).astype(np.float32) / 255
rgbR640 = np.asarray(Image.open(SRC / "right_640.png")).astype(np.float32) / 255
gt = np.load(SRC / "depth_gt.npy")[1::2, 1::2]
ids = np.load(SRC / "ids.npy")
ids640 = ids[1::2, 1::2]
disp = {n: np.fromfile(SRC / f"disp_{n}.f32", np.float32).reshape(H, W) for n in ["sgm_lr", "sgm", "block"]}


def median3(d):
    p = np.pad(np.where(d > 0, d, np.nan), 1, constant_values=np.nan)
    stack = np.stack([p[y:y + H, x:x + W] for y in range(3) for x in range(3)])
    n = np.isfinite(stack).sum(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(stack, 0)
    return np.where(n >= 5, med, 0)


def depth_of(d):
    d = median3(d)
    z = np.where(d > 0, FB / np.maximum(d, 1e-6), 0)
    return np.where((z >= 0.12) & (z <= 3.0), z, 0)


Z = depth_of(disp["sgm_lr"])


def up(a):
    return np.repeat(np.repeat(a, 2, 0), 2, 1)


def turbo(x):
    x = np.clip(x, 0, 1)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return np.clip(np.stack([r, g, b], -1), 0, 1)


def heat(z):
    return turbo(1 - (z - MINZ) / (MAXZ - MINZ))


def luma(a):
    return a @ np.array([0.299, 0.587, 0.114])


def save(a, name, q=86):
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(IMG / name, quality=q, method=6)


def label_hue(label):
    h = 7
    for ch in label:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % 360) / 360


def label_rgb(label):
    return np.array(colorsys.hsv_to_rgb(label_hue(label), 0.8, 0.95))


Zu = up(Z)
valid = Zu > 0
H2 = heat(Zu)

save(rgb, "mode-raw.webp")

std = rgb.copy()
for oid, name in LABELS.items():
    m = ids == oid
    std[m] = std[m] * 0.55 + label_rgb(name) * 0.45
save(std, "mode-standard.webp")
save(np.where(valid[..., None], H2, 0), "mode-depth.webp")
save(np.where(valid[..., None], rgb * 0.45 + H2 * 0.55, rgb * 0.7), "mode-overlay.webp")
g = np.where(valid, np.clip((1 / np.maximum(Zu, 1e-3) - 1 / MAXZ) / (1 / MINZ - 1 / MAXZ), 0, 1), 0)
save(np.repeat(g[..., None], 3, -1), "mode-disparity.webp")

pair = np.full((720, 1280, 3), [0.04, 0.04, 0.05], np.float32)
pair[180:540, :640] = rgb640
pair[180:540, 640:] = rgbR640
for k in range(12):
    y = 180 + int(k * 360 / 12)
    pair[y:y + 2] = pair[y:y + 2] * 0.15 + np.array([0.1, 1.0, 0.35]) * 0.85
pair[180:540, 639:641] = 1
save(pair, "mode-pair.webp")
tight = np.concatenate([rgb640, np.full((360, 4, 3), 1.0, np.float32), rgbR640], 1)
for k in range(1, 12):
    y = int(k * 360 / 12)
    tight[y:y + 2] = tight[y:y + 2] * 0.15 + np.array([0.1, 1.0, 0.35]) * 0.85
save(tight, "use-pair.webp")
save(np.stack([luma(rgb), luma(rgbR), luma(rgbR)], -1), "mode-anaglyph.webp")

step = 4
gw, gh = W // step, H // step
grid = np.zeros((gh, gw), np.float32)
for gy in range(gh):
    for gx in range(gw):
        cell = Z[gy * step:(gy + 1) * step, gx * step:(gx + 1) * step]
        v = cell[cell > 0]
        if len(v) * 2 >= step * step:
            grid[gy, gx] = v.mean()
lab = np.zeros((gh, gw), np.int32)
comps = []
cid = 0
for i in range(gw * gh):
    y0, x0 = divmod(i, gw)
    if grid[y0, x0] <= 0 or lab[y0, x0]:
        continue
    cid += 1
    lab[y0, x0] = cid
    stack = [(y0, x0)]
    cnt = 0
    while stack:
        y, x = stack.pop()
        cnt += 1
        z = grid[y, x]
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < gh and 0 <= xx < gw and not lab[yy, xx] and grid[yy, xx] > 0 and abs(grid[yy, xx] - z) < 0.02 + 0.04 * z:
                lab[yy, xx] = cid
                stack.append((yy, xx))
    comps.append((cnt, cid))
keep = {c: k + 1 for k, (n, c) in enumerate(sorted([c for c in comps if c[0] >= 40], reverse=True)[:24])}
seg = np.vectorize(lambda c: keep.get(c, 0))(lab)


def palette(i):
    h = (i * 0.61803398875) % 1
    k = np.abs(((h + np.array([1, 2 / 3, 1 / 3])) % 1) * 6 - 3) - 1
    return 1 + (np.clip(k, 0, 1) - 1) * 0.8


segu = np.repeat(np.repeat(seg, 8, 0), 8, 1)
gray = luma(rgb)[..., None]
segimg = np.where(segu[..., None] > 0, gray * 0.4 + np.array([palette(i) for i in range(30)])[segu] * 0.6, gray * 0.4)
save(segimg, "mode-segmentation.webp")


def font(size):
    for p in ["/System/Library/Fonts/SFNSRounded.ttf", "/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


objects = []
fx640, cx640, cy640 = 470.0, 320.0, 180.0
for oid, name in LABELS.items():
    m = ids == oid
    ys, xs = np.nonzero(m)
    m640 = ids640 == oid
    inner = m640.copy()
    inner[1:-1, 1:-1] = m640[1:-1, 1:-1] & m640[:-2, 1:-1] & m640[2:, 1:-1] & m640[1:-1, :-2] & m640[1:-1, 2:]
    vals = Z[inner & (Z > 0)]
    z = float(np.median(vals)) if len(vals) >= 6 else None
    ztrue = float(np.median(gt[inner]))
    x0, x1, y0, y1 = xs.min() / 1280, (xs.max() + 1) / 1280, ys.min() / 720, (ys.max() + 1) / 720
    o = {"label": name, "bbox": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)], "hue": label_hue(name), "trueDistance": round(ztrue, 3)}
    if z:
        o["distance"] = round(z, 3)
        o["x"] = round(((x0 + x1) / 2 * W - cx640) * z / fx640, 3)
        o["width"] = round((x1 - x0) * W * z / fx640, 3)
        o["height"] = round((y1 - y0) * H * z / fx640, 3)
    objects.append(o)

radar = rgb * 0.25
im = Image.fromarray((radar * 255).astype(np.uint8))
dr = ImageDraw.Draw(im, "RGBA")
ox, oy = 640, 720 - 28
s = (720 - 68) / MAXZ
half = np.arctan(W / 2 / fx640)
far = MAXZ * s
dr.polygon([(ox - np.tan(half) * far, oy - far), (ox, oy), (ox + np.tan(half) * far, oy - far)], fill=(255, 255, 255, 13), outline=(255, 255, 255, 90))
f_small, f_obj = font(18), font(22)
for r in np.arange(0.5, MAXZ + 1e-6, 0.5):
    rr = r * s
    dr.ellipse([ox - rr, oy - rr, ox + rr, oy + rr], outline=(255, 255, 255, 46), width=2)
    dr.text((ox + 8, oy - rr - 22), f"{r:.1f} m", fill=(255, 255, 255, 150), font=f_small)
for gx in range(gw):
    col = grid[int(gh * 0.3):int(gh * 0.7), gx]
    col = col[col > 0]
    if not len(col):
        continue
    z = col.min()
    u = (gx + 0.5) * step
    x = (u - cx640) * z / fx640
    px, py = ox + x * s, oy - z * s
    t = (z - MINZ) / (MAXZ - MINZ)
    c = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(0.66 * min(1, max(0, t)), 0.9, 1))
    dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=c)
for o in objects:
    if "distance" not in o:
        continue
    px, py = ox + o["x"] * s, oy - o["distance"] * s
    r = max(12, o["width"] * s / 2)
    c = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(o["hue"], 0.8, 0.95))
    dr.ellipse([px - r, py - r, px + r, py + r], fill=c + (90,), outline=c, width=4)
    text = f"{o['label']} {o['distance']:.2f} m"
    tw = dr.textlength(text, font=f_obj)
    dr.rounded_rectangle([px + r + 4, py - 16, px + r + 20 + tw, py + 16], 8, fill=c + (220,))
    dr.text((px + r + 12, py - 13), text, fill=(255, 255, 255), font=f_obj)
dr.polygon([(ox - 16, oy + 12), (ox, oy - 12), (ox + 16, oy + 12)], fill=(255, 214, 0))
im.convert("RGB").save(IMG / "mode-radar.webp", quality=88, method=6)


def cloud(depth, colors, step):
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    z = depth[ys, xs]
    m = z > 0
    X = (xs[m] + 0.5 - cx640) * z[m] / fx640
    Y = (ys[m] + 0.5 - cy640) * z[m] / fx640
    return np.stack([X, Y, z[m]], -1), colors[ys[m], xs[m]]


def splat(P, C, yaw, pitch, dist, size=(1280, 720), r=2, pivot=1.2):
    P = P - np.array([0, 0, pivot])
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    P = P @ np.array([[cy_, 0, -sy_], [0, 1, 0], [sy_, 0, cy_]]).T
    cp, sp = np.cos(pitch), np.sin(pitch)
    P = P @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]).T
    P = P + np.array([0, 0, dist])
    w_, h_ = size
    f = fx640 * w_ / W
    m = P[:, 2] > 0.05
    P, C = P[m], C[m]
    u = (P[:, 0] / P[:, 2] * f + w_ / 2).astype(int)
    v = (P[:, 1] / P[:, 2] * f + h_ / 2).astype(int)
    order = np.argsort(-P[:, 2])
    out = np.zeros((h_, w_, 3), np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            uu, vv = u[order] + dx, v[order] + dy
            ok = (uu >= 0) & (uu < w_) & (vv >= 0) & (vv < h_)
            out[vv[ok], uu[ok]] = C[order][ok]
    return out


P, Cc = cloud(Z, rgb640, 1)
save(splat(P, Cc, 0.38, 0.30, 1.45), "mode-pointcloud.webp")

Pg, Cg = cloud(Z, rgb640, 2)
q = np.round(Pg * 1000).astype(np.int16)
(PUB / "cloud.bin").write_bytes(np.int32(len(q)).tobytes() + q.tobytes() + (Cg * 255 + 0.5).astype(np.uint8).tobytes())


def stats(d):
    gtd = FB / gt
    sl = (slice(4, H - 4), slice(70, W - 4))
    dd, g = median3(d)[sl], gtd[sl]
    n = dd.size
    inv = (dd <= 0).sum()
    ok = ((dd > 0) & (np.abs(dd - g) <= 1)).sum()
    return {"correct": round(ok / n * 100, 1), "wrong": round((n - ok - inv) / n * 100, 1), "invalid": round(inv / n * 100, 1)}


bins = []
zr = depth_of(disp["sgm_lr"])
for b0 in np.arange(0.6, 2.0, 0.1):
    m = (gt >= b0) & (gt < b0 + 0.1) & (zr > 0)
    if m.sum() > 200:
        bins.append({"z": round(float(b0 + 0.05), 2), "medianError": round(float(np.median(np.abs(zr[m] - gt[m]))), 4), "n": int(m.sum())})

save(np.where(up(depth_of(disp["block"]))[..., None] > 0, heat(up(depth_of(disp["block"]))), 0), "compare-block.webp")
save(np.where(up(depth_of(disp["sgm"]))[..., None] > 0, heat(up(depth_of(disp["sgm"]))), 0), "compare-sgm.webp")
gtv = up(np.where((gt > 0.12) & (gt < 3), gt, 0))
save(heat(gtv), "compare-truth.webp")

removed = up((disp["sgm"] > 0) & (disp["sgm_lr"] <= 0))
dsg = up(depth_of(disp["sgm"]))
lrimg = np.where(dsg[..., None] > 0, heat(dsg) * 0.35, 0)
lrimg[removed] = [1.0, 0.18, 0.3]
save(lrimg, "step-lr.webp")
save(np.repeat(np.clip((up(median3(disp["sgm"])) - 1) / 8, 0, 1)[..., None], 3, -1), "step-disparity.webp")

gl = luma(rgb640)
pad = np.pad(gl, 3, mode="edge")
bits = np.zeros((H, W, 24), bool)
k = 0
for dy in range(-3, 4):
    for dx in range(-3, 4):
        if (dx, dy) == (0, 0) or k >= 24:
            continue
        bits[..., k] = pad[3 + dy:3 + dy + H, 3 + dx:3 + dx + W] < gl - 0.004
        k += 1
cen = np.stack([(bits[..., c * 8:(c + 1) * 8] * (1 << np.arange(8))).sum(-1) / 255 for c in range(3)], -1)
save(up(cen), "step-census.webp")

meta = {
    "scene": {"f": 470, "width": W, "height": H, "baseline_m": 0.010, "fB": FB, "range": [MINZ, MAXZ]},
    "objects": objects,
    "cloudPoints": int(len(q)),
    "centerDistance": round(float(np.median(Z[164:196, 304:336][Z[164:196, 304:336] > 0])), 3),
}
public_meta = {**meta, "objects": [{k: v for k, v in o.items() if k != "trueDistance"} for o in objects]}
json.dump(public_meta, open(PUB / "scene.json", "w"), indent=1)
print(json.dumps({"block": stats(disp["block"]), "sgm": stats(disp["sgm"]), "sgm_lr": stats(disp["sgm_lr"])}), len(q), "points")
print("error by distance", bins)
for o in objects:
    print(o["label"], o.get("distance"), o["trueDistance"])

og = Image.new("RGB", (1200, 630), (11, 16, 36))
pc = splat(P, Cc, 0.5, 0.42, 1.6, size=(760, 630), r=2)
mask = (pc.sum(-1) > 0)[..., None]
bg = np.array([11, 16, 36]) / 255
pc = np.where(mask, pc, bg)
fade = np.clip(np.linspace(-0.4, 1.6, 760), 0, 1)[None, :, None]
og.paste(Image.fromarray(((pc * fade + bg * (1 - fade)) * 255).astype(np.uint8)), (440, 0))
logo = Image.open(PUB / "logo.png").convert("RGBA")
og.paste(logo, (72, 150), logo)
d = ImageDraw.Draw(og)
title_font = font(78)
try:
    title_font.set_variation_by_name("Bold")
except Exception:
    pass
d.text((72, 290), "StereoVision", fill=(233, 237, 247), font=title_font)
d.text((72, 395), "Real-time stereo depth from\nthe two rear cameras of an iPhone", fill=(154, 165, 196), font=font(34), spacing=10)
og.save(PUB.parent / "app" / "opengraph-image.png", optimize=True)
