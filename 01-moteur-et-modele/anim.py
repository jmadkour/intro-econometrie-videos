import json, math, sys, io, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H, S, FPS = 1280, 720, 2, 30
TIM = json.load(open("timing.json"))
TOTAL = TIM[-1]["end"]

BG = (14, 22, 44); BG2 = (22, 33, 62); GRID = (34, 48, 84)
ORANGE = (255, 140, 50); TEAL = (40, 210, 200); WHITE = (240, 244, 252)
GREY = (140, 152, 180); GREEN = (60, 205, 110); RED = (240, 80, 80); YELLOW = (255, 210, 60)
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FC = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"

@functools.lru_cache(None)
def font(sz, bold=True, cond=False):
    return ImageFont.truetype(FC if cond else (FB if bold else FR), max(1, int(sz * S)))

def cl(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x):
    x = cl(x); return x * x * (3 - 2 * x)
def back(x):
    x = cl(x); c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def lerp(a, b, u): return a + (b - a) * u
def mix(c1, c2, u): return tuple(int(lerp(a, b, u)) for a, b in zip(c1, c2))
def P(*xy): return [v * S for v in xy]

@functools.lru_cache(None)
def math_img(tex, size, color):
    fig = plt.figure(figsize=(0.01, 0.01), dpi=100)
    c = tuple(v / 255 for v in color)
    fig.text(0, 0, tex, fontsize=size * S * 0.75, color=c,
             math_fontfamily="cm")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGBA")

def paste_c(img, im, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.01: return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    if alpha < 1:
        a = im.getchannel("A").point(lambda v: int(v * alpha)); im = im.copy(); im.putalpha(a)
    img.alpha_composite(im, (int(cx * S - im.width / 2), int(cy * S - im.height / 2)))

def text(d, xy, s, sz, fill, bold=True, anchor="mm", cond=False):
    d.text((xy[0] * S, xy[1] * S), s, font=font(sz, bold, cond), fill=fill, anchor=anchor)

def check(d, cx, cy, r, u=1.0, col=GREEN):
    if u <= 0: return
    rr = r * back(u)
    d.ellipse(P(cx - rr, cy - rr, cx + rr, cy + rr), fill=col)
    if u > 0.4:
        pts = [(cx - rr * .45, cy), (cx - rr * .1, cy + rr * .38), (cx + rr * .5, cy - rr * .35)]
        d.line([v * S for p in pts for v in p], fill=WHITE, width=int(max(2, rr * .22) * S), joint="curve")

def rrect(d, box, r, **kw):
    d.rounded_rectangle(P(*box), radius=r * S, **kw)

# ---------- drawing primitives ----------
def draw_engine(d, cx, cy, sc, t, run=True, xray=False, bolt_checks=None):
    col = (90, 160, 255) if xray else (120, 130, 150)
    line = (170, 210, 255) if xray else (60, 66, 80)
    fillb = (20, 40, 80) if xray else (98, 106, 124)
    w, h = 190 * sc, 130 * sc
    x0, y0 = cx - w / 2, cy - h / 2 + 20 * sc
    rrect(d, (x0, y0, x0 + w, y0 + h), 10 * sc, fill=fillb, outline=line, width=int(3 * S))
    # head
    rrect(d, (x0 - 8 * sc, y0 - 34 * sc, x0 + w + 8 * sc, y0 + 4 * sc), 8 * sc, fill=mix(fillb, (255, 255, 255), .12), outline=line, width=int(3 * S))
    # cylinders + pistons
    for i in range(3):
        cxi = x0 + w * (i + 1) / 4
        cw = 34 * sc
        d.rectangle(P(cxi - cw / 2, y0 + 8 * sc, cxi + cw / 2, y0 + h - 18 * sc), fill=(10, 20, 40) if xray else (50, 54, 66))
        ph = math.sin(t * (14 if run else 0) + i * 2.1)
        py = y0 + 30 * sc + (ph + 1) * 18 * sc
        d.rectangle(P(cxi - cw / 2 + 2 * sc, py, cxi + cw / 2 - 2 * sc, py + 18 * sc), fill=col)
        d.line(P(cxi, py + 18 * sc, cxi + 6 * sc * math.cos(t * 14 + i * 2.1), y0 + h - 10 * sc), fill=line, width=int(4 * S))
        # spark
        if run and ph > 0.85:
            d.ellipse(P(cxi - 6 * sc, y0 + 10 * sc, cxi + 6 * sc, y0 + 22 * sc), fill=YELLOW)
    # belt + pulleys on right side
    pa = (x0 + w + 30 * sc, y0 - 10 * sc); pb = (x0 + w + 30 * sc, y0 + h - 20 * sc)
    for (px, py), r in ((pa, 16 * sc), (pb, 24 * sc)):
        d.ellipse(P(px - r, py - r, px + r, py + r), outline=line, width=int(3 * S), fill=fillb)
        ang = t * (6 if run else 0)
        d.line(P(px, py, px + r * math.cos(ang), py + r * math.sin(ang)), fill=line, width=int(2 * S))
    d.line(P(pa[0] - 16 * sc, pa[1], pb[0] - 24 * sc, pb[1]), fill=ORANGE if not xray else (120, 200, 255), width=int(3 * S))
    d.line(P(pa[0] + 16 * sc, pa[1], pb[0] + 24 * sc, pb[1]), fill=ORANGE if not xray else (120, 200, 255), width=int(3 * S))
    # bolts
    bolts = [(x0 + 4 * sc, y0 - 15 * sc), (x0 + w - 4 * sc, y0 - 15 * sc), (x0 + 10 * sc, y0 + h - 12 * sc), (x0 + w - 10 * sc, y0 + h - 12 * sc), pa]
    for bx, by in bolts:
        d.regular_polygon((bx * S, by * S, 7 * sc * S), 6, fill=(200, 200, 210) if not xray else (200, 230, 255), outline=line)
    if bolt_checks is not None:
        for (bx, by), u in zip(bolts, bolt_checks):
            check(d, bx + 14 * sc, by - 14 * sc, 11 * sc, u)
    return (x0 + w / 2, y0 + h)

def draw_car(d, cx, cy, sc, t, speed=1.0, color=ORANGE):
    w = 200 * sc
    body = [(cx - w / 2, cy), (cx - w / 2, cy - 30 * sc), (cx - w / 2 + 30 * sc, cy - 36 * sc), (cx - 40 * sc, cy - 70 * sc),
            (cx + 35 * sc, cy - 70 * sc), (cx + 60 * sc, cy - 38 * sc), (cx + w / 2, cy - 30 * sc), (cx + w / 2, cy)]
    d.polygon([v * S for p in body for v in p], fill=color, outline=mix(color, (0, 0, 0), .4))
    d.polygon(P(cx - 32 * sc, cy - 64 * sc, cx - 5 * sc, cy - 64 * sc, cx - 5 * sc, cy - 40 * sc, cx - 52 * sc, cy - 40 * sc), fill=(170, 220, 255))
    d.polygon(P(cx + 2 * sc, cy - 64 * sc, cx + 30 * sc, cy - 64 * sc, cx + 50 * sc, cy - 40 * sc, cx + 2 * sc, cy - 40 * sc), fill=(170, 220, 255))
    d.ellipse(P(cx + w / 2 - 14 * sc, cy - 26 * sc, cx + w / 2 - 2 * sc, cy - 16 * sc), fill=YELLOW)
    for wx in (cx - 55 * sc, cx + 60 * sc):
        r = 22 * sc
        d.ellipse(P(wx - r, cy - r, wx + r, cy + r), fill=(30, 30, 36))
        d.ellipse(P(wx - r * .55, cy - r * .55, wx + r * .55, cy + r * .55), fill=(170, 175, 190))
        a = -t * 8 * speed
        for k in range(3):
            aa = a + k * 2.094
            d.line(P(wx, cy, wx + r * .5 * math.cos(aa), cy + r * .5 * math.sin(aa)), fill=(60, 60, 70), width=int(3 * S))
    return (cx - w / 2, cy - 12 * sc)  # exhaust position

def draw_road(d, x0, x1, y, t, speed, years=True, alpha_years=1.0, travel=None):
    d.rectangle(P(x0, y, x1, y + 34), fill=(48, 54, 70))
    off = (t * speed * 120) % 80
    x = x0 - off
    while x < x1:
        if x + 40 > x0:
            d.rectangle(P(max(x0, x), y + 15, min(x1, x + 40), y + 19), fill=(230, 230, 230))
        x += 80
    if years:
        step = 300; base = (t * speed if travel is None else travel) * 120
        k0 = int(base // step)
        for k in range(k0, k0 + 7):
            xx = x0 + k * step - base + 40
            if x0 + 10 < xx < x1 - 10:
                d.line(P(xx, y + 34, xx, y + 44), fill=GREY, width=int(2 * S))
                text(d, (xx, y + 56), str(2016 + k), 13, mix(BG, YELLOW, alpha_years))

def smoke(layer, ox, oy, t, col, opacity, start_t, n=40, spread=1.0):
    dd = ImageDraw.Draw(layer)
    for i in range(n):
        birth = start_t + i * 0.09
        age = t - birth
        if age < 0: continue
        age = (age % 3.6)
        x = ox - age * 70 - 10
        y = oy - age * 22 * spread + 6 * math.sin(i * 1.7 + age * 3)
        r = 6 + age * 14
        a = int(opacity * 255 * cl(1 - age / 3.6) * 0.75)
        if a <= 2: continue
        dd.ellipse(P(x - r, y - r, x + r, y + r), fill=col + (a,))

def clock(d, cx, cy, r, t, col=GREY):
    d.ellipse(P(cx - r, cy - r, cx + r, cy + r), outline=col, width=int(3 * S))
    for k in range(12):
        a = k * math.pi / 6
        d.line(P(cx + r * .82 * math.cos(a), cy + r * .82 * math.sin(a), cx + r * .95 * math.cos(a), cy + r * .95 * math.sin(a)), fill=col, width=int(2 * S))
    a1 = t * 2 - math.pi / 2; a2 = t * 0.17 - math.pi / 2
    d.line(P(cx, cy, cx + r * .75 * math.cos(a1), cy + r * .75 * math.sin(a1)), fill=col, width=int(3 * S))
    d.line(P(cx, cy, cx + r * .5 * math.cos(a2), cy + r * .5 * math.sin(a2)), fill=col, width=int(4 * S))

# deterministic series
rng = np.random.default_rng(7)
N = 40
xs = np.cumsum(rng.normal(0.4, 1, N)); ys = 2 + 0.7 * xs + rng.normal(0, 0.8, N)
eps = rng.normal(0, 1, 60)
hist = np.zeros(48); e = rng.normal(0, 1, 48)
for i in range(1, 48): hist[i] = 0.7 * hist[i - 1] + e[i] + 0.15
hist += 10 + np.arange(48) * 0.12

def mini_series(d, box, vals, prog, col, pts=True, zero=False):
    x0, y0, x1, y1 = box
    lo, hi = vals.min() - .5, vals.max() + .5
    n = len(vals); k = max(2, int(n * prog))
    if zero:
        ym = y1 - (0 - lo) / (hi - lo) * (y1 - y0)
        d.line(P(x0, ym, x1, ym), fill=GREY, width=int(1 * S))
    pp = [(x0 + (x1 - x0) * i / (n - 1), y1 - (v - lo) / (hi - lo) * (y1 - y0)) for i, v in enumerate(vals[:k])]
    d.line([v * S for p in pp for v in p], fill=col, width=int(2 * S), joint="curve")
    if pts:
        for (px, py) in pp: d.ellipse(P(px - 2.5, py - 2.5, px + 2.5, py + 2.5), fill=col)
    return pp

def line_at(seq, idx):
    return TIM[seq]["lines"][idx]

def local(t, seq, idx, dur=0.6, off=0.0):
    s = TIM[seq]["lines"][idx][0] + off
    return ease((t - s) / dur)

TITLES = ["L'initiale M & D… et le Temps", "Démarrage & Estimation", "Tests mécaniques vs Tests sur les coefficients",
          "La fumée vs les résidus : le Bruit Blanc", "Conclusion : cap sur la prévision"]

# ---------- scenes ----------
def scene1(img, d, t):
    s = TIM[0]; tt = t - s["start"]
    # time background
    clock(d, 640, 330, 110, t, col=(36, 52, 92))
    # years band
    base = tt * 40
    for k in range(-1, 12):
        xx = 40 + k * 120 - (base % 120)
        yr = 2010 + k + int(base // 120)
        text(d, (xx, 560), str(yr), 15, (60, 80, 125))
    uM = local(t, 0, 1); uD = local(t, 0, 3)
    pulseM = 0.5 + 0.5 * math.sin(t * 9) if line_at(0, 1)[0] < t < line_at(0, 1)[1] + 0.3 else 0
    pulseD = 0.5 + 0.5 * math.sin(t * 9) if line_at(0, 2)[0] < t < line_at(0, 3)[1] + 0.3 else 0
    a0 = ease(tt / 0.8)
    draw_engine(d, 300, 260, 0.55 + 0.4 * back(a0), t, run=False)
    eq = math_img(r"$Y_t=\beta_0+\beta_1X_t+\varepsilon_t$", 30, WHITE)
    paste_c(img, eq, 960, 260, a0)
    # M row
    for cx, word, colr in ((300, "oteur", ORANGE), (960, "odèle", TEAL)):
        cM = mix(colr, YELLOW, pulseM)
        if uM > 0:
            text(d, (cx - 70, 420), "M", 58 * (1 + .25 * pulseM), cM)
            text(d, (cx - 40, 432), word, 32, mix(BG, WHITE, uM), anchor="lm")
    tD = local(t, 0, 2)
    for cx, word, colr in ((300, "iesel", ORANGE), (960, "onnées", TEAL)):
        cD = mix(colr, YELLOW, pulseD)
        if tD > 0:
            text(d, (cx - 70, 500), "D", 58 * (1 + .25 * pulseD), cD)
            text(d, (cx - 40, 512), word, 32, mix(BG, WHITE, uD), anchor="lm")
    # icons: jerrycan & data table hints
    if uD > 0:
        rrect(d, (440, 470, 480, 525), 6, fill=mix(BG, (200, 40, 40), uD))
        text(d, (460, 498), "D", 18, mix(BG, WHITE, uD))
        for r in range(3):
            for c in range(3):
                d.rectangle(P(1100 + c * 22, 472 + r * 17, 1119 + c * 22, 486 + r * 17), fill=mix(BG, (40, 120, 140) if r else TEAL, uD))

def scene2(img, d, t):
    s = TIM[1]; tt = t - s["start"]
    go = local(t, 1, 0, 1.2, 0.8)
    # left: engine top, road + car bottom
    draw_engine(d, 300, 200, 0.75, t, run=go > 0.2)
    # fuel pouring
    if tt < 6:
        u = ease(tt / 0.8)
        cx, cy = 140, 140
        rrect(d, (cx - 25, cy - 30, cx + 25, cy + 30), 6, fill=mix(BG, (200, 40, 40), u))
        text(d, (cx, cy), "D", 22, mix(BG, WHITE, u))
        for i in range(6):
            fy = cy + 30 + ((tt * 90 + i * 18) % 70)
            d.ellipse(P(cx + 30 - 4, fy - 6, cx + 30 + 4, fy + 6), fill=mix(BG, (230, 200, 60), u))
    trav = max(0.0, t - (TIM[1]["lines"][0][0] + 0.8)) * 1.4
    draw_road(d, 20, 620, 470, t, speed=1.4 if go > .3 else 0, travel=trav)
    draw_car(d, 300 + 6 * math.sin(t * 2) * go, 466, 0.75, t, speed=go * 1.4)
    # right: data series -> equation -> betas
    u1 = ease(tt / 0.8)
    rrect(d, (700, 140, 1220, 300), 10, outline=mix(BG, GRID, u1), width=int(2 * S))
    text(d, (712, 152), "Données temporelles (2016 → 2025)", 13, mix(BG, GREY, u1), anchor="lt")
    prog = cl(tt / 6)
    mini_series(d, (720, 180, 1200, 290), ys, prog, TEAL)
    # flow particles
    fl = local(t, 1, 1, 0.5)
    if fl > 0:
        for i in range(8):
            ph = ((tt * 0.8 + i / 8) % 1)
            y = 300 + ph * 60
            d.ellipse(P(960 - 4, y - 4, 960 + 4, y + 4), fill=TEAL)
    eq = math_img(r"$Y_t=\beta_0+\beta_1X_t+\varepsilon_t$", 26, WHITE)
    paste_c(img, eq, 960, 385, u1)
    b = local(t, 1, 1, 0.8, 2.5)
    b0 = math_img(r"$\hat\beta_0 = 2{,}14$", 28, YELLOW)
    b1 = math_img(r"$\hat\beta_1 = 0{,}71$", 28, YELLOW)
    paste_c(img, b0, 860, 470, b, scale=0.6 + 0.4 * back(b))
    paste_c(img, b1, 1060, 470, local(t, 1, 1, 0.8, 3.3), scale=0.6 + 0.4 * back(local(t, 1, 1, 0.8, 3.3)))
    sig = local(t, 1, 1, 0.8, 6.0)
    text(d, (960, 530), "→ une signification économique", 17, mix(BG, WHITE, sig))
    # warning
    w = local(t, 1, 2, 0.4)
    if w > 0:
        fl = 0.6 + 0.4 * abs(math.sin(t * 6))
        for cx, cy in ((560, 140), (1200, 380)):
            r = 34 * back(w)
            d.polygon(P(cx, cy - r, cx + r, cy + r * .75, cx - r, cy + r * .75), fill=mix(BG, YELLOW, fl), outline=(60, 40, 0))
            text(d, (cx, cy + r * .2), "!", 30 * back(w), (40, 30, 0))

def scene3(img, d, t):
    s = TIM[2]; tt = t - s["start"]
    u = ease(tt / 0.8)
    # X-ray frame
    rrect(d, (60, 130, 580, 520), 14, fill=mix(BG, (8, 18, 40), u), outline=mix(BG, (90, 160, 255), u), width=int(2 * S))
    text(d, (80, 146), "VUE RAYONS X", 13, mix(BG, (120, 190, 255), u), anchor="lt")
    # scan line
    sy = 160 + ((tt * 120) % 340)
    d.line(P(70, sy, 570, sy), fill=mix(BG, (90, 160, 255), .5 * u), width=int(2 * S))
    st = TIM[2]["lines"][1][0]
    checks = [ease((t - st - 0.8 - i * 0.9) / 0.4) for i in range(5)]
    draw_engine(d, 300, 300, 1.1, t, run=True, xray=True, bolt_checks=checks)
    # wrench
    wx = 470 + 30 * math.sin(tt * 1.3); wy = 460
    d.line(P(wx - 50, wy + 20, wx + 10, wy - 20), fill=(210, 215, 225), width=int(9 * S))
    d.ellipse(P(wx, wy - 36, wx + 30, wy - 6), outline=(210, 215, 225), width=int(8 * S))
    # right: regression table
    rows = [("", "Coef.", "t de Student", "p-valeur", ""),
            (r"$\hat\beta_0$", "2,14", "4,87", "< 0,001", 0),
            (r"$\hat\beta_1$", "0,71", "9,32", "< 0,001", 1)]
    tb = local(t, 2, 2, 0.6)
    x0, y0 = 680, 160
    rrect(d, (x0, y0, 1240, y0 + 170), 10, fill=mix(BG, BG2, tb), outline=mix(BG, GRID, tb), width=int(2 * S))
    colx = [x0 + 45, x0 + 140, x0 + 275, x0 + 410, x0 + 510]
    for ri, row in enumerate(rows):
        yy = y0 + 30 + ri * 55
        for ci, cell in enumerate(row[:4]):
            if ri and ci == 0:
                paste_c(img, math_img(cell, 22, WHITE), colx[ci], yy, tb)
            else:
                text(d, (colx[ci], yy), cell, 16 if ri else 14, mix(BG, WHITE if ri else GREY, tb), bold=ri == 0)
        if ri:
            uu = local(t, 2, 3, 0.4, 1.0 + 0.9 * (ri - 1))
            check(d, colx[4], yy, 14, uu)
            if uu > 0.3:
                hl = cl(uu)
                d.rounded_rectangle(P(colx[2] - 55, yy - 18, colx[2] + 55, yy + 18), radius=8 * S, outline=mix(BG, GREEN, hl), width=int(2 * S))
    # Fisher
    fu = local(t, 2, 3, 0.5, 3.4)
    rrect(d, (x0, 360, 1240, 440), 10, fill=mix(BG, BG2, fu), outline=mix(BG, GRID, fu), width=int(2 * S))
    text(d, (x0 + 20, 384), "Test F de Fisher (significativité globale)", 15, mix(BG, GREY, fu), anchor="lm")
    paste_c(img, math_img(r"$F = 86{,}9 \quad p < 0{,}001$", 22, WHITE), x0 + 220, 415, fu)
    check(d, 1205, 400, 16, local(t, 2, 3, 0.4, 4.2))
    ok = local(t, 2, 3, 0.6, 5.5)
    text(d, (960, 490), "Variables significatives & bien à leur place", 17, mix(BG, GREEN, ok))

def scene4(img, d, t):
    s = TIM[3]; tt = t - s["start"]
    L = TIM[3]["lines"]
    # left: big exhaust close-up
    u = ease(tt / 0.8)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    rrect(d, (40, 125, 610, 420), 16, fill=mix(BG, (178, 196, 220), u))
    # pipe
    px, py = 470, 360
    d.rectangle(P(px, py - 22, 600, py + 22), fill=mix(BG, (150, 155, 165), u))
    d.ellipse(P(px - 14, py - 26, px + 14, py + 26), fill=mix(BG, (40, 40, 46), u), outline=mix(BG, (190, 195, 205), u), width=int(4 * S))
    # car rear hint
    rrect(d, (575, 215, 660, 345), 14, fill=mix(BG, ORANGE, u))
    rrect(d, (585, 235, 605, 275), 5, fill=mix(BG, RED, u))
    d.pieslice(P(560, 330, 660, 430), 180, 270, fill=mix(BG, (30, 30, 36), u))
    # phases
    t_black, t_oil, t_clear = L[1][0], L[2][0], L[3][0]
    pb = cl((t - t_black + 0.5) / 0.6) * (1 - cl((t - t_oil) / 0.8))
    po = cl((t - t_oil) / 0.6) * (1 - cl((t - t_clear) / 0.8))
    pc = cl((t - t_clear) / 0.6)
    pre = 1 - cl((t - t_black + 0.5) / 0.6)
    smoke(layer, px - 10, py, t, (120, 125, 140), 0.35 * pre * u, s["start"])
    smoke(layer, px - 10, py, t, (15, 15, 18), 0.95 * pb, t_black - 0.5)
    smoke(layer, px - 10, py, t, (95, 80, 40), 0.9 * po, t_oil - 0.2, spread=0.6)
    smoke(layer, px - 10, py, t, (255, 255, 255), 0.55 * pc, t_clear - 0.2)
    img.alpha_composite(layer)
    d2 = ImageDraw.Draw(img)
    # labels
    def tag(y, label, col, a, icon):
        if a <= 0: return
        rrect(d2, (60, y - 22, 480, y + 22), 22, fill=mix(BG, col, a * .9))
        text(d2, (95, y), icon, 20, mix(BG, WHITE, a))
        text(d2, (125, y), label, 15, mix(BG, WHITE, a), anchor="lm")
    tag(450, "Fumée noire : combustion incomplète", (90, 90, 100), local(t, 3, 1, .4), "✕")
    tag(500, "Fumée grasse : fuite d'huile", (140, 110, 40), local(t, 3, 2, .4), "✕")
    tag(550, "Fumée claire : moteur sain", GREEN, local(t, 3, 3, .4), "✓")
    if po > 0.2:  # oil drops
        for i in range(4):
            dy = ((t * 60 + i * 11) % 30)
            d2.ellipse(P(500 + i * 20 - 4, py + 24 + dy, 500 + i * 20 + 4, py + 36 + dy), fill=(110, 90, 30))
    # right
    u2 = local(t, 3, 4, .6)
    eq_full = math_img(r"$Y_t=\beta_0+\beta_1X_t+$", 26, mix(BG, WHITE, 1 - 0.55 * u2))
    eps_im = math_img(r"$\varepsilon_t$", 26 + 10 * u2, YELLOW if u2 > 0 else WHITE)
    ew = (eq_full.width + eps_im.width) / S + 8
    ex0 = 960 - ew / 2
    paste_c(img, eq_full, ex0 + eq_full.width / S / 2, 160, u)
    paste_c(img, eps_im, ex0 + eq_full.width / S + 8 + eps_im.width / S / 2, 160, u)
    if u2 > 0:
        text(d2, (1100, 200), "= les résidus", 15, mix(BG, YELLOW, u2))
        rrect(d2, (700, 225, 1230, 330), 10, outline=mix(BG, GRID, u2), width=int(2 * S))
        mini_series(d2, (715, 235, 1215, 320), eps, cl((t - L[4][0]) / 3), mix(BG, YELLOW, u2), zero=True)
    crit = [(r"$E[\varepsilon_t]=0$", "Espérance nulle"), (r"$V(\varepsilon_t)=\sigma^2$", "Homoscédasticité"),
            (r"$\mathrm{Cov}(\varepsilon_t,\varepsilon_{t-k})=0$", "Absence d'autocorrélation"),
            (r"$\varepsilon_t\sim\mathcal{N}(0,\sigma^2)$", "Loi normale")]
    for i, (tex, lab) in enumerate(crit):
        a = local(t, 3, 6 + i, .4)
        y = 360 + i * 48
        if a <= 0: continue
        imp = (i == 2)
        rrect(d2, (700 + 30 * (1 - a), y - 20, 1230, y + 20), 8, fill=mix(BG, BG2, a), outline=mix(BG, YELLOW if imp else GRID, a), width=int(2 * S))
        text(d2, (720, y), lab + (" ★" if imp else ""), 15, mix(BG, YELLOW if imp else WHITE, a), anchor="lm")
        paste_c(img, math_img(tex, 17, WHITE), 1060, y, a)
        check(d2, 1205, y, 13, local(t, 3, 6 + i, .4, .3))
        if imp and a > 0.5:
            text(d2, (965, 558), "★ L'autocorrélation : point crucial en séries temporelles !", 13, mix(BG, YELLOW, a))
    # stamp
    st = local(t, 3, 9, .45, 0.6)
    if st > 0:
        stamp = Image.new("RGBA", (420 * S, 110 * S), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stamp)
        sd.rounded_rectangle((6 * S, 6 * S, 414 * S, 104 * S), radius=14 * S, outline=GREEN + (235,), width=int(7 * S))
        sd.text((210 * S, 56 * S), "BRUIT BLANC", font=font(46), fill=GREEN + (235,), anchor="mm")
        stamp = stamp.rotate(10, expand=True, resample=Image.BICUBIC)
        paste_c(img, stamp, 965, 455, cl(st * 1.5), scale=lerp(2.2, 1.0, st))
    # same-color tie: residuals <-> smoke
    if u2 > 0:
        text(d2, (325, 145), "fumée  ≈  résidus", 16, mix(BG, (30, 40, 70), u2))

def scene5(img, d, t):
    s = TIM[4]; tt = t - s["start"]
    u = ease(tt / 0.8)
    # left: perspective road to horizon
    hy = 230
    d.rectangle(P(20, 120, 620, hy), fill=mix(BG, (40, 60, 110), u))
    sun = 0.5 + 0.5 * u
    sy = hy - 8 - 18 * sun
    d.ellipse(P(288, sy - 32, 352, sy + 32), fill=mix(BG, (255, 190, 90), u))
    d.rectangle(P(20, hy, 620, 590), fill=mix(BG, (24, 60, 50), u))
    d.polygon(P(310, hy, 330, hy, 600, 590, 40, 590), fill=mix(BG, (55, 60, 75), u))
    for k in range(8):
        z = ((k / 8) + tt * 0.35) % 1
        z2 = z ** 2
        y = hy + z2 * 360; hh = 6 + z2 * 30; ww = 1 + z2 * 6
        d.polygon(P(320 - ww, y, 320 + ww, y, 320 + ww * 1.2, y + hh, 320 - ww * 1.2, y + hh), fill=mix(BG, (235, 235, 235), u))
    # car from behind, shrinking toward horizon
    pr = ease(tt / 13)
    sc = lerp(1.0, 0.25, pr); cy = lerp(540, hy + 25, pr)
    cw, ch = 130 * sc, 60 * sc
    rrect(d, (320 - cw / 2, cy - ch, 320 + cw / 2, cy), 10 * sc, fill=ORANGE)
    rrect(d, (320 - cw * .35, cy - ch * 1.6, 320 + cw * .35, cy - ch * .9), 8 * sc, fill=mix(ORANGE, (0, 0, 0), .25))
    d.rectangle(P(320 - cw * .45, cy - ch * .55, 320 - cw * .3, cy - ch * .4), fill=RED)
    d.rectangle(P(320 + cw * .3, cy - ch * .55, 320 + cw * .45, cy - ch * .4), fill=RED)
    text(d, (320, 150), "→ l'avenir", 16, mix(BG, WHITE, u))
    # right: forecast chart
    x0, y0, x1, y1 = 700, 150, 1230, 470
    d.line(P(x0, y1, x1, y1), fill=mix(BG, GREY, u), width=int(2 * S))
    d.line(P(x0, y0, x0, y1), fill=mix(BG, GREY, u), width=int(2 * S))
    nh, nf = 48, 16
    fc = hist[-1] + np.cumsum(np.full(nf, 0.12)) * 1.0 + (hist[-1] - hist.mean()) * 0
    lo, hi = hist.min() - 3, hist.max() + 6
    X = lambda i: x0 + 10 + (x1 - x0 - 20) * i / (nh + nf - 1)
    Y = lambda v: y1 - 10 - (v - lo) / (hi - lo) * (y1 - y0 - 20)
    ph = cl(tt / 3)
    k = max(2, int(nh * ph))
    pts = [(X(i), Y(hist[i])) for i in range(k)]
    d.line([v * S for p in pts for v in p], fill=TEAL, width=int(3 * S), joint="curve")
    pf = cl((tt - 3) / 4)
    if pf > 0:
        kf = max(1, int(nf * pf))
        up = [(X(nh - 1 + i), Y(fc[i - 1] + 1.0 * math.sqrt(i) * 1.1 if i else hist[-1])) for i in range(kf + 1)]
        dn = [(X(nh - 1 + i), Y(fc[i - 1] - 1.0 * math.sqrt(i) * 1.1 if i else hist[-1])) for i in range(kf + 1)]
        band = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(band).polygon([v * S for p in up + dn[::-1] for v in p], fill=YELLOW + (60,))
        img.alpha_composite(band)
        d = ImageDraw.Draw(img)
        mid = [(X(nh - 1 + i), Y(fc[i - 1] if i else hist[-1])) for i in range(kf + 1)]
        for a, b in zip(mid[:-1], mid[1:]):
            if int((a[0]) / 6) % 2 == 0 or True:
                d.line(P(a[0], a[1], b[0], b[1]), fill=YELLOW, width=int(3 * S))
        d.line(P(X(nh - 1), y0, X(nh - 1), y1), fill=mix(BG, GREY, .6), width=int(1 * S))
        text(d, (X(nh - 1) - 8, y0 + 8), "aujourd'hui", 11, GREY, anchor="rt")
    lg = cl((tt - 6) / 0.6)
    text(d, (x0 + 10, 500), "— observé", 14, mix(BG, TEAL, u), anchor="lm")
    text(d, (x0 + 140, 500), "— prévision", 14, mix(BG, YELLOW, lg), anchor="lm")
    text(d, (x0 + 290, 500), "▮ intervalle de confiance 95 %", 14, mix(BG, (200, 180, 90), lg), anchor="lm")
    return d

SCENES = [scene1, scene2, scene3, scene4, scene5]

def subtitle(t):
    for s in TIM:
        for a, b, line in s["lines"]:
            if a - 0.05 <= t <= b + 0.25: return line
    return None

def wrap(s, f, maxw):
    words = s.split(); lines = []; cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if f.getlength(test) > maxw and cur:
            lines.append(cur); cur = w
        else: cur = test
    lines.append(cur); return lines

def frame(i):
    t = i / FPS
    img = Image.new("RGBA", (W * S, H * S), BG + (255,))
    d = ImageDraw.Draw(img)
    # subtle grid
    for gx in range(0, W, 40): d.line(P(gx, 0, gx, H), fill=(17, 26, 51))
    for gy in range(0, H, 40): d.line(P(0, gy, W, gy), fill=(17, 26, 51))
    si = 0
    for k, s in enumerate(TIM):
        if t >= s["start"] - 0.8: si = k
    s = TIM[si]
    # intro title card before first speech
    if t < TIM[0]["start"] - 0.0:
        pass
    # header
    rrect(d, (20, 16, 1260, 70), 14, fill=BG2)
    rrect(d, (32, 26, 160, 60), 17, fill=YELLOW)
    text(d, (96, 43), f"SÉQUENCE {si+1}", 14, BG)
    text(d, (180, 43), TITLES[si], 20, WHITE, anchor="lm")
    clock(d, 1225, 43, 17, t, col=GREY)
    # panel labels & divider
    d.line(P(640, 90, 640, 590), fill=GRID, width=int(2 * S))
    text(d, (320, 100), "LE MOTEUR", 16, ORANGE)
    text(d, (960, 100), "LE MODÈLE ÉCONOMÉTRIQUE", 16, TEAL)
    SCENES[si](img, d, t)
    d = ImageDraw.Draw(img)
    # scene fade
    fin = ease((t - (s["start"] - 0.8)) / 0.6) if si > 0 else ease(t / 0.6)
    fout = 1 - ease((t - (s["end"] - 0.9)) / 0.7) if si < len(TIM) - 1 else 1
    f = min(fin, fout)
    if f < 1:
        ov = Image.new("RGBA", (W * S, int(500 * S)), BG + (int(255 * (1 - f)),))
        img.alpha_composite(ov, (0, int(85 * S)))
        d = ImageDraw.Draw(img)
    # subtitle band
    rrect(d, (20, 600, 1260, 700), 14, fill=(8, 13, 28))
    sub = subtitle(t)
    if sub:
        ls = wrap(sub, font(21, False), 1180 * S)
        for j, l in enumerate(ls[:3]):
            yy = 650 + (j - (len(ls) - 1) / 2) * 28
            text(d, (640, yy), l, 21, WHITE, bold=False)
    # final card
    endt = TIM[-1]["lines"][-1][1] + 0.3
    if t > endt:
        a = ease((t - endt) / 0.8)
        ov = Image.new("RGBA", (W * S, H * S), BG + (int(235 * a),))
        img.alpha_composite(ov); d = ImageDraw.Draw(img)
        text(d, (640, 300), "Bonne route en économétrie !", 44, mix(BG, YELLOW, a))
        text(d, (640, 370), "Moteur sain + fumée claire  ⟺  coefficients testés + résidus bruit blanc", 18, mix(BG, WHITE, a), bold=False)
        text(d, (640, 420), "Économétrie des séries temporelles · Licence 3 Sciences Économiques", 15, mix(BG, GREY, a), bold=False)
    # progress bar
    d.rectangle(P(0, 714, W * t / TOTAL, 720), fill=YELLOW)
    out = img.convert("RGB").resize((W, H), Image.LANCZOS)
    return out.tobytes()

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "still":
        for ts in sys.argv[2:]:
            tt = float(ts)
            Image.frombytes("RGB", (W, H), frame(int(tt * FPS))).save(f"still_{ts}.png")
    else:
        import subprocess
        from multiprocessing import Pool
        n = int(TOTAL * FPS)
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                              "-i", "voice.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                              "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", sys.argv[2]], stdin=subprocess.PIPE)
        with Pool(2) as pool:
            for k, fr in enumerate(pool.imap(frame, range(n), chunksize=8)):
                p.stdin.write(fr)
                if k % 300 == 0: print(k, "/", n, flush=True)
        p.stdin.close(); p.wait(); print("done")
