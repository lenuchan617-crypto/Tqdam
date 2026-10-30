"""
Render «ОДИН НА ЗЕМЛЕ» — 1080x1920, 30 fps, 60 s vertical short.

Compositing: Ken Burns camera over 16 illustrated key frames, hard cuts,
kinetic stickers, chapter chips, progress rail, halftone grain, vignette,
one fully procedural map card (shot 12) and a CTA end card.

usage:  python3 build_video.py [--preview t1 t2 ...] [--out name.mp4]
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

import imageio_ffmpeg

W, H = 1080, 1920
FPS = 30
DUR = 60.0
NF = int(DUR * FPS)
FF = imageio_ffmpeg.get_ffmpeg_exe()

ROOT = os.path.dirname(os.path.abspath(__file__))
KF = os.path.join(ROOT, "keyframes")
OUT = os.path.join(ROOT, "out")
FONTS = os.path.join(ROOT, "fonts")
os.makedirs(OUT, exist_ok=True)

INK = (25, 25, 39)
ORANGE = (255, 107, 53)
YELLOW = (232, 255, 71)
TEAL = (66, 221, 224)
PURPLE = (105, 80, 161)
PAPER = (244, 238, 226)

F = {
    "oswald": os.path.join(FONTS, "Oswald-Bold.ttf"),
    "oswald600": os.path.join(FONTS, "Oswald-SemiBold.ttf"),
    "black": os.path.join(FONTS, "Rubik-Black.ttf"),
    "bold": os.path.join(FONTS, "Rubik-Bold.ttf"),
    "reg": os.path.join(FONTS, "Rubik-Regular.ttf"),
}
_fc = {}


def font(name, size):
    key = (name, size)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(F[name], size)
    return _fc[key]


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


def ease_in_out(p):
    return p * p * (3 - 2 * p)


def ease_out_back(p, s=1.7):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


# ----------------------------------------------------------------- base plates ---
BASE_MULT = 1.6  # extra resolution headroom for the camera move


def load_base(name):
    p = os.path.join(KF, name + ".png")
    im = Image.open(p).convert("RGB")
    return im.resize((int(W * BASE_MULT), int(H * BASE_MULT)), Image.LANCZOS)


def hsla_rect(d, box, color, alpha=255, radius=0):
    layer = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), color + (alpha,))
    mask = Image.new("L", layer.size, 0)
    md = ImageDraw.Draw(mask)
    if radius:
        md.rounded_rectangle([0, 0, layer.size[0] - 1, layer.size[1] - 1], radius, fill=255)
    else:
        md.rectangle([0, 0, layer.size[0] - 1, layer.size[1] - 1], fill=255)
    return layer, mask


# ------------------------------------------------------------- procedural map ---
def build_map_card():
    """Shot 12: hand-drawn planning map, city traps → river house."""
    Wb, Hb = int(W * BASE_MULT), int(H * BASE_MULT)
    img = Image.new("RGB", (Wb, Hb), PAPER)
    d = ImageDraw.Draw(img)

    # paper grid
    step = Wb // 16
    for x in range(0, Wb, step):
        d.line([(x, 0), (x, Hb)], fill=(232, 224, 208), width=2)
    for y in range(0, Hb, step):
        d.line([(0, y), (Wb, y)], fill=(232, 224, 208), width=2)

    # river (right side, diagonal band)
    river = [(Wb * 0.80, -40), (Wb * 0.70, Hb * 0.22), (Wb * 0.78, Hb * 0.44),
             (Wb * 0.66, Hb * 0.66), (Wb * 0.72, Hb * 0.88), (Wb * 0.60, Hb + 40)]
    left_edge = [(x - Wb * 0.16, y) for x, y in river]
    d.polygon(river + left_edge[::-1], fill=(120, 205, 224))
    d.line(river, fill=(40, 140, 170), width=9, joint="curve")
    d.line(left_edge, fill=(40, 140, 170), width=9, joint="curve")

    # countryside fields
    for (cx, cy, rx, ry, col) in [(0.16, 0.70, 0.15, 0.09, (206, 222, 168)),
                                  (0.30, 0.86, 0.12, 0.06, (196, 214, 158)),
                                  (0.13, 0.42, 0.10, 0.07, (214, 226, 178))]:
        d.ellipse([Wb * (cx - rx), Hb * (cy - ry), Wb * (cx + rx), Hb * (cy + ry)], fill=col)

    # city block grid (traps)
    gx, gy = Wb * 0.09, Hb * 0.10
    bw, bh, gap = Wb * 0.085, Hb * 0.030, Wb * 0.020
    for r in range(7):
        for c in range(5):
            x0 = gx + c * (bw + gap)
            y0 = gy + r * (bh + gap)
            shade = 168 + ((r * 5 + c * 3) % 4) * 12
            d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(shade, shade, shade + 6))
            d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(90, 90, 100), width=4)
    # warning frame around the city
    fx0, fy0 = gx - Wb * 0.03, gy - Hb * 0.025
    fx1, fy1 = gx + 5 * (bw + gap) + Wb * 0.01, gy + 7 * (bh + gap) + Hb * 0.01
    d.rectangle([fx0, fy0, fx1, fy1], outline=ORANGE, width=14)
    for i in range(9):
        ang = i * 8
        pass

    # route: city -> river house
    pts = []
    p0 = (fx1 - Wb * 0.02, (fy0 + fy1) / 2)
    p1 = (Wb * 0.80, Hb * 0.66)
    for i in range(40):
        t = i / 39
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t + math.sin(t * 3.14) * Hb * 0.05
        pts.append((x, y))
    for i in range(0, len(pts) - 1, 2):
        d.line([pts[i], pts[i + 1]], fill=(226, 176, 20), width=12)

    # destination: house by the water
    hx, hy = Wb * 0.745, Hb * 0.635
    hs = Wb * 0.072
    d.rectangle([hx - hs, hy - hs * 0.7, hx + hs, hy + hs * 0.7], fill=(250, 246, 236),
                outline=INK, width=6)
    d.polygon([(hx - hs * 1.2, hy - hs * 0.7), (hx, hy - hs * 1.5), (hx + hs * 1.2, hy - hs * 0.7)],
              fill=ORANGE, outline=INK)
    d.rectangle([hx - hs * 0.28, hy - hs * 0.05, hx + hs * 0.28, hy + hs * 0.7],
                fill=(90, 90, 110))
    # red cross marker
    d.ellipse([hx - hs * 2.2, hy - hs * 2.2, hx + hs * 2.2, hy + hs * 2.2],
              outline=(200, 40, 40), width=12)
    d.line([(hx - hs * 0.9, hy), (hx + hs * 0.9, hy)], fill=(200, 40, 40), width=14)
    d.line([(hx, hy - hs * 0.9), (hx, hy + hs * 0.9)], fill=(200, 40, 40), width=14)

    # legend
    legend = "× = ТОЧКА НАЗНАЧЕНИЯ"
    lf = font("oswald", 46)
    tw = d.textlength(legend, font=lf)
    x0, y0 = Wb * 0.07, Hb * 0.895
    d.rounded_rectangle([x0, y0, x0 + tw + 52, y0 + Hb * 0.058], 22,
                        fill=(255, 255, 255), outline=INK, width=6)
    d.text((x0 + 26, y0 + Hb * 0.029), legend, font=lf, fill=INK, anchor="lm")
    return img


# ------------------------------------------------------------------- shot data ---
# beat: (rel_t0, rel_t1, zoom0, zoom1, cx0, cy0, cx1, cy1)  centre in 0..1
SHOTS = [
    dict(id=1, t0=0.00, t1=2.30, img="k01",
         beats=[(0.0, 0.45, 1.42, 1.30, 0.62, 0.40, 0.60, 0.42),
                (0.45, 1.0, 1.26, 1.50, 0.60, 0.46, 0.60, 0.72)],
         chip=None, stickers=[], grade="cool"),
    dict(id=2, t0=2.30, t1=7.10, img="k02",
         beats=[(0.0, 0.44, 1.00, 1.22, 0.50, 0.44, 0.58, 0.40),
                (0.44, 1.0, 1.45, 1.20, 0.44, 0.82, 0.50, 0.80)],
         chip=None, grade="cool",
         stickers=[dict(t=2.55, t1=6.9, text="НАСЕЛЕНИЕ: 1", style="tag",
                        pos=(120, 300), anchor="lm", size=62, pop=True),
                   dict(t=3.10, t1=6.9, text="ОДИН НА ЗЕМЛЕ", style="big",
                        pos=(540, 1330), anchor="mm", size=150, rot=-2, pop=True),
                   dict(t=3.45, t1=6.9, text="ЛЮДЕЙ — НОЛЬ. ПЛАН — МАКСИМУМ.", style="sub",
                        pos=(540, 1450), anchor="mm", size=44, pop=True)]),
    dict(id=3, t0=7.10, t1=10.10, img="k03",
         beats=[(0.0, 1.0, 1.05, 1.30, 0.50, 0.52, 0.50, 0.48)],
         chip="03", shake=1.0, grade="cool",
         stickers=[dict(t=7.25, t1=10.05, text="ПЕРВЫЕ 5 МИНУТ", style="big",
                        pos=(540, 1420), anchor="mm", size=108, rot=-1.5, pop=True),
                   dict(t=7.55, t1=8.45, text="ПАНИКА", style="stk", pos=(200, 420), rot=-8),
                   dict(t=8.00, t1=8.90, text="КРИКИ", style="stk", pos=(700, 560), rot=7),
                   dict(t=8.45, t1=9.35, text="ОТРИЦАНИЕ", style="stk", pos=(420, 380), rot=-4)]),
    dict(id=4, t0=10.10, t1=13.00, img="k04",
         beats=[(0.0, 0.55, 1.28, 1.10, 0.50, 0.42, 0.50, 0.44),
                (0.55, 1.0, 1.16, 1.34, 0.50, 0.45, 0.50, 0.50)],
         chip="04", flash=[10.14, 10.62], grade="mid",
         stickers=[dict(t=10.25, t1=13.0, text="ПАНИКА / ВЫЖИВАНИЕ", style="tag",
                        pos=(120, 300), anchor="lm", size=58, pop=True),
                   dict(t=12.10, t1=12.95, text="РЕЖИМ: ВКЛ", style="big",
                        pos=(540, 1360), anchor="mm", size=112, rot=-2, pop=True)]),
    dict(id=5, t0=13.00, t1=16.00, img="k05",
         beats=[(0.0, 0.63, 1.30, 1.12, 0.52, 0.50, 0.50, 0.52),
                (0.63, 1.0, 1.10, 1.36, 0.50, 0.52, 0.52, 0.48)],
         chip="01 / ПРИПАСЫ", flash=[13.42], grade="mid",
         stickers=[dict(t=13.20, t1=16.0, text="ГАСТРОНОМ", style="plate",
                        pos=(540, 470), anchor="mm", size=86, rot=-2, pop=True),
                   dict(t=13.42, t1=14.45, text="БЭМ!", style="bam", pos=(690, 900), rot=9)]),
    dict(id=6, t0=16.00, t1=19.05, img="k06",
         beats=[(0.0, 0.30, 1.32, 1.18, 0.42, 0.45, 0.50, 0.48),
                (0.30, 0.62, 1.18, 1.34, 0.60, 0.40, 0.55, 0.52),
                (0.62, 1.0, 1.20, 1.05, 0.50, 0.55, 0.50, 0.50)],
         chip="01 / ПРИПАСЫ", grade="mid",
         stickers=[dict(t=16.30, t1=17.25, text="+ ВОДА", style="stk", pos=(180, 380), rot=-5),
                   dict(t=17.05, t1=18.00, text="+ ТУШЁНКА", style="stk", pos=(640, 480), rot=4),
                   dict(t=17.80, t1=19.0, text="+ КРУПЫ", style="stk", pos=(360, 330), rot=-3),
                   dict(t=18.15, t1=19.0, text="БРО НА ГОДЫ", style="tag",
                        pos=(120, 300), anchor="lm", size=58, pop=True)]),
    dict(id=7, t0=19.05, t1=23.10, img="k07",
         beats=[(0.0, 1.0, 1.34, 1.02, 0.52, 0.62, 0.50, 0.50)],
         chip="02 / ЭНЕРГИЯ", invert=[19.12], grade="dark",
         stickers=[dict(t=19.35, t1=23.05, text="СВЕТ — ВСЁ", style="big",
                        pos=(540, 1300), anchor="mm", size=124, rot=-2, pop=True),
                   dict(t=20.10, t1=23.05, text="ЧЕРЕЗ ПАРУ ДНЕЙ — ТИШИНА", style="sub",
                        pos=(540, 1420), anchor="mm", size=46, pop=True)]),
    dict(id=8, t0=23.10, t1=27.15, img="k08",
         beats=[(0.0, 0.45, 1.22, 1.06, 0.48, 0.55, 0.50, 0.50),
                (0.45, 1.0, 1.06, 1.30, 0.52, 0.48, 0.50, 0.42)],
         chip="02 / ЭНЕРГИЯ", grade="mid",
         stickers=[dict(t=23.30, t1=27.1, text="АВТОНОМНЫЙ РЕЖИМ", style="tag",
                        pos=(120, 300), anchor="lm", size=58, pop=True),
                   dict(t=25.10, t1=27.1, text="ГЕНЕРАТОР + СОЛНЦЕ", style="sub",
                        pos=(540, 1400), anchor="mm", size=48, pop=True)]),
    dict(id=9, t0=27.15, t1=31.00, img="k09",
         beats=[(0.0, 0.55, 1.18, 1.06, 0.50, 0.50, 0.52, 0.48),
                (0.55, 1.0, 1.12, 1.42, 0.55, 0.55, 0.56, 0.60)],
         chip="03 / АПТЕКА", grade="mid", health=True,
         stickers=[dict(t=28.95, t1=31.0, text="ЗУБ = ФИНАЛЬНЫЙ БОСС", style="big",
                        pos=(540, 1330), anchor="mm", size=92, rot=-2, pop=True)]),
    dict(id=10, t0=31.00, t1=34.40, img="k10",
         beats=[(0.0, 0.50, 1.20, 1.08, 0.50, 0.52, 0.52, 0.50),
                (0.50, 1.0, 1.10, 1.32, 0.50, 0.50, 0.50, 0.44)],
         chip="03 / АПТЕКА", grade="mid",
         stickers=[dict(t=31.30, t1=34.35, text="???", style="bam",
                        pos=(815, 470), rot=8, size=150),
                   dict(t=32.10, t1=34.35, text="РАЗМЕР ИМЕЕТ ЗНАЧЕНИЕ", style="tag",
                        pos=(120, 300), anchor="lm", size=58, pop=True),
                   dict(t=33.10, t1=34.35, text="НЕ СПРАШИВАЙ. ПРОСТО БЕРИ.", style="sub",
                        pos=(540, 1360), anchor="mm", size=46, pop=True)]),
    dict(id=11, t0=34.40, t1=40.30, img="k11",
         beats=[(0.0, 0.34, 1.28, 1.12, 0.48, 0.42, 0.52, 0.46),
                (0.34, 0.68, 1.14, 1.30, 0.56, 0.52, 0.50, 0.56),
                (0.68, 1.0, 1.26, 1.06, 0.50, 0.48, 0.50, 0.50)],
         chip="04 / ЗНАНИЯ", grade="mid",
         stickers=[dict(t=34.60, t1=40.25, text="КАЧАЙ МОЗГИ", style="big",
                        pos=(540, 1330), anchor="mm", size=112, rot=-2, pop=True),
                   dict(t=35.30, t1=40.25, text="ТОЛЬКО БУМАГА. WI-FI НЕТ.", style="sub",
                        pos=(540, 1445), anchor="mm", size=46, pop=True)]),
    dict(id=12, t0=40.30, t1=46.60, img="MAP",
         beats=[(0.0, 0.34, 1.30, 1.16, 0.34, 0.28, 0.40, 0.34),
                (0.34, 0.70, 1.18, 1.30, 0.46, 0.44, 0.64, 0.60),
                (0.70, 1.0, 1.34, 1.60, 0.74, 0.64, 0.755, 0.655)],
         chip="05 / БАЗА", grade="warm",
         stickers=[dict(t=40.55, t1=46.55, text="К ВОДЕ. ЗА ГОРОД.", style="big",
                        pos=(540, 1280), anchor="mm", size=96, rot=-2, pop=True),
                   dict(t=44.30, t1=46.55, text="ДОМ У ВОДЫ", style="stk",
                        pos=(600, 620), rot=5)]),
    dict(id=13, t0=46.60, t1=49.80, img="k13",
         beats=[(0.0, 0.50, 1.24, 1.06, 0.52, 0.50, 0.50, 0.52),
                (0.50, 1.0, 1.20, 1.38, 0.46, 0.56, 0.52, 0.50)],
         chip="05 / БАЗА", grade="warm",
         stickers=[dict(t=46.85, t1=49.75, text="МЕХАНИКА > ПОНТЫ", style="tag",
                        pos=(120, 300), anchor="lm", size=56, pop=True),
                   dict(t=47.60, t1=49.75, text="ВРУУУМ", style="stk", pos=(700, 620), rot=6)]),
    dict(id=14, t0=49.80, t1=53.60, img="k14",
         beats=[(0.0, 0.45, 1.22, 1.08, 0.48, 0.55, 0.52, 0.52),
                (0.45, 1.0, 1.10, 1.28, 0.54, 0.48, 0.52, 0.42)],
         chip=None, grade="warm",
         stickers=[dict(t=51.10, t1=53.55, text="+1 ДРУГ", style="heart",
                        pos=(640, 560), rot=4),
                   dict(t=50.10, t1=53.55, text="КАРТОШКА · ВОДА · ПЁС", style="sub",
                        pos=(540, 1545), anchor="mm", size=48, pop=True)]),
    dict(id=15, t0=53.60, t1=55.75, img="k15",
         beats=[(0.0, 1.0, 1.02, 1.20, 0.50, 0.50, 0.50, 0.46)],
         chip=None, grade="warm",
         stickers=[dict(t=53.85, t1=56.25, text="ТВОЙ МИР. ТВОИ ПРАВИЛА.", style="plate",
                        pos=(540, 470), anchor="mm", size=64, rot=0, pop=True)]),
    dict(id=16, t0=55.75, t1=60.00, img="k16",
         beats=[(0.0, 0.40, 1.20, 1.06, 0.50, 0.46, 0.50, 0.48),
                (0.40, 1.0, 1.06, 1.22, 0.50, 0.48, 0.50, 0.44)],
         chip=None, grade="warm", cta=True,
         stickers=[dict(t=55.95, t1=60.0, text="А ТЫ БЫ ВЫЖИЛ?", style="big",
                        pos=(540, 440), anchor="mm", size=104, rot=-2, pop=True)]),
]


# ---------------------------------------------------------------- composition ---
class Renderer:
    def __init__(self):
        self.cache = {}
        self.grain = []
        rng = np.random.RandomState(4)
        for i in range(8):
            n = rng.normal(128, 7, (H, W, 3)).clip(0, 255).astype(np.uint8)
            self.grain.append(Image.fromarray(n))
        # vignette
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = W / 2, H / 2
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / 1.414
        v = (1 - 0.55 * np.clip(r, 0, 1) ** 2.2)
        v = (v * 255).astype(np.uint8)
        self.vign_l = Image.fromarray(v, "L")
        self.vign = Image.merge("RGB", (self.vign_l, self.vign_l, self.vign_l))
        self.tints = {
            "cool": Image.new("RGB", (W, H), (232, 242, 255)),
            "mid": Image.new("RGB", (W, H), (255, 252, 246)),
            "dark": Image.new("RGB", (W, H), (226, 224, 248)),
            "warm": Image.new("RGB", (W, H), (255, 244, 228)),
        }

    def base(self, name):
        if name not in self.cache:
            self.cache[name] = build_map_card() if name == "MAP" else load_base(name)
        return self.cache[name]

    # -- camera ---------------------------------------------------------------
    def camera(self, shot, rel_p, t):
        beats = shot["beats"]
        b = beats[0]
        for i, cand in enumerate(beats):
            if rel_p >= cand[0]:
                b = cand
        span = max(1e-6, b[1] - b[0])
        p = min(1.0, max(0.0, (rel_p - b[0]) / span))
        e = ease_in_out(p)
        z = b[2] + (b[3] - b[2]) * e
        cx = b[4] + (b[6] - b[4]) * e
        cy = b[5] + (b[7] - b[5]) * e
        shake = shot.get("shake", 0.0)
        if shake:
            amp = 8 * shake
            cx += math.sin(t * 47.3) * amp / W
            cy += math.cos(t * 61.7) * amp / H
        return z, cx, cy

    def frame(self, idx):
        t = idx / FPS
        shot = SHOTS[-1]
        for s in SHOTS:
            if s["t0"] <= t < s["t1"]:
                shot = s
                break
        base = self.base(shot["img"])
        bw, bh = base.size
        rel = (t - shot["t0"]) / (shot["t1"] - shot["t0"])
        z, cx, cy = self.camera(shot, rel, t)
        # zoom 1.0 = whole plate visible, BASE_MULT = native pixels
        z = min(BASE_MULT, max(1.0, z))
        cw, ch = bw / z, bh / z
        cx = min(max(cx, cw / 2 / bw), 1 - cw / 2 / bw)
        cy = min(max(cy, ch / 2 / bh), 1 - ch / 2 / bh)
        x0 = cx * bw - cw / 2
        y0 = cy * bh - ch / 2
        crop = base.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        img = crop.resize((W, H), Image.LANCZOS if cw > W else Image.BICUBIC)

        # grade
        img = ImageChops.multiply(img, self.tints.get(shot.get("grade", "mid"), self.tints["mid"]))
        img = img.point(lambda v: min(255, int(v * 1.03 + 6)))

        img = self.draw_graphics(img, shot, t, rel)

        # impacts
        for ft in shot.get("flash", []):
            if 0 <= t - ft < 1.5 / FPS:
                img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), 0.55)
            elif 1.5 / FPS <= t - ft < 4 / FPS:
                img = Image.blend(img, Image.new("RGB", (W, H), (255, 240, 200)), 0.2)
        for ft in shot.get("invert", []):
            if 0 <= t - ft < 2 / FPS:
                img = ImageChops.invert(img)

        # grain + vignette
        img = ImageChops.add(img, self.grain[idx % 8], 1.0, -128)
        img = ImageChops.multiply(img, self.vign)
        if idx < 6:  # fade up from black
            img = Image.blend(Image.new("RGB", (W, H), INK), img, idx / 6)
        return img

    # -- graphics -------------------------------------------------------------
    def draw_graphics(self, img, shot, t, rel):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        parts = []

        self.rail(d, shot, t)

        if shot.get("chip"):
            parts.append(self.chip(shot["chip"], shot["t0"], t))

        if shot.get("health"):
            p = min(1.0, max(0.0, (t - 28.95) / 0.5))
            self.health(d, 1.0 - 0.62 * ease_out_cubic(p))

        for st in shot.get("stickers", []):
            if st["t"] <= t <= st.get("t1", 99):
                parts.append(self.sticker(st, t))

        if shot.get("cta"):
            parts.append(self.cta(t))

        for piece, pos in parts:
            if piece is None:
                continue
            layer.alpha_composite(piece, (int(pos[0]), int(pos[1])))

        img = img.convert("RGBA")
        img.alpha_composite(layer)
        return img.convert("RGB")

    def rail(self, d, shot, t):
        n = 16
        x0, x1 = 120, 900
        y = 150
        gap = 8
        wseg = (x1 - x0 - gap * (n - 1)) / n
        p = min(1.0, max(0.0, (t - shot["t0"]) / max(0.01, shot["t1"] - shot["t0"])))
        for i in range(n):
            sx = x0 + i * (wseg + gap)
            done = i + 1 < shot["id"]
            active = i + 1 == shot["id"]
            col = YELLOW + (255,) if done else (ORANGE + (255,) if active else INK + (95,))
            d.rounded_rectangle([sx, y, sx + wseg, y + 7], 3, fill=col)
            if active:
                d.rounded_rectangle([sx, y, sx + wseg * p, y + 7], 3, fill=YELLOW + (255,))

    def chip(self, text, t0, t):
        p = ease_out_back(min(1.0, (t - t0) / 0.28))
        if p <= 0.02:
            return None, (0, 0)
        f = font("oswald", 44)
        probe = Image.new("RGBA", (8, 8))
        tw = ImageDraw.Draw(probe).textlength(text, font=f)
        pad = 26
        w0, h0 = int(tw + pad * 2), 76
        lay = Image.new("RGBA", (w0 + 20, h0 + 20), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        ld.rounded_rectangle([10, 10, w0 + 10, h0 + 10], 14, fill=INK + (238,),
                             outline=YELLOW + (255,), width=3)
        ld.text((10 + pad, 10 + h0 / 2), text, font=f, fill=YELLOW + (255,), anchor="lm")
        s = 0.86 + 0.14 * p
        lay = lay.resize((int(lay.width * s), int(lay.height * s)), Image.BICUBIC)
        return lay, (120, 196)

    def health(self, d, frac):
        x, y, w, h = 560, 392, 340, 34
        d.rounded_rectangle([x - 8, y - 8, x + w + 8, y + h + 8], 12, fill=INK + (235,))
        d.rounded_rectangle([x, y, x + max(6, w * frac), y + h], 8, fill=(226, 62, 62, 255))
        d.text((x, y - 40), "HP ГЕРОЯ", font=font("oswald", 34), fill=(255, 255, 255, 245))

    def sticker(self, st, t):
        age = t - st["t"]
        style = st["style"]
        rot = st.get("rot", 0)
        txt = st["text"]

        size_default = {"tag": 60, "big": 100, "stk": 58, "bam": 120, "plate": 80,
                        "heart": 84, "sub": 46}
        fkey = "oswald" if style == "tag" else ("oswald600" if style == "sub" else "black")
        size = st.get("size", size_default[style])
        maxw = 900 if style != "tag" else 780
        f = font(fkey, size)
        probe = Image.new("RGBA", (8, 8))
        pd = ImageDraw.Draw(probe)
        while size > 30:
            bb = pd.textbbox((0, 0), txt, font=f, stroke_width=12)
            if bb[2] - bb[0] <= maxw:
                break
            size -= 4
            f = font(fkey, size)
        st = dict(st)
        st["_fit"] = size

        pop = ease_out_back(min(1.0, age / 0.26))
        alpha = int(255 * min(1.0, age / 0.16))
        scale = 0.84 + 0.16 * pop if style in ("big", "bam", "stk", "plate", "heart") else 1.0
        drop = (1 - pop) * 46 if style == "big" else 0

        bbox = pd.textbbox((0, 0), txt, font=f, stroke_width=12)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 26
        cw, chh = int(tw + pad * 2), int(th + pad * 2)
        lay = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        cx, cy = cw / 2, chh / 2

        if style == "plate":
            ld.rounded_rectangle([4, 4, cw - 5, chh - 5], 18, fill=INK + (232,),
                                 outline=YELLOW + (255,), width=4)
            ld.text((cx, cy), txt, font=f, fill=YELLOW + (alpha,), anchor="mm")
        elif style == "tag":
            ld.rounded_rectangle([2, 2, cw - 3, chh - 3], 12, fill=(255, 255, 255, 242),
                                 outline=INK + (255,), width=5)
            ld.text((cx, cy), txt, font=f, fill=INK + (alpha,), anchor="mm")
        elif style == "stk":
            ld.text((cx, cy), txt, font=f, fill=ORANGE + (alpha,), anchor="mm",
                    stroke_width=9, stroke_fill=INK + (255,))
        elif style == "bam":
            ld.text((cx, cy), txt, font=f, fill=YELLOW + (alpha,), anchor="mm",
                    stroke_width=12, stroke_fill=INK + (255,))
        elif style == "heart":
            ld.text((cx, cy), txt, font=f, fill=TEAL + (alpha,), anchor="mm",
                    stroke_width=10, stroke_fill=INK + (255,))
        elif style == "sub":
            ld.text((cx, cy), txt, font=f, fill=(255, 255, 255, alpha), anchor="mm",
                    stroke_width=7, stroke_fill=INK + (255,))
        else:
            ld.rounded_rectangle([6, 6, cw - 7, chh - 7], 24, fill=INK + (205,))
            ld.text((cx, cy), txt, font=f, fill=YELLOW + (alpha,), anchor="mm",
                    stroke_width=11, stroke_fill=INK + (255,))

        if rot:
            lay = lay.rotate(rot, expand=True, resample=Image.BICUBIC)
        if abs(scale - 1.0) > 0.01:
            lay = lay.resize((max(1, int(lay.width * scale)),
                              max(1, int(lay.height * scale))), Image.BICUBIC)

        ax, ay = st["pos"]
        anchor = st.get("anchor", "mm")
        px = ax if anchor[0] == "l" else (ax - lay.width if anchor[0] == "r" else ax - lay.width / 2)
        py = ay - lay.height / 2
        px = min(max(px, 46), W - lay.width - 46)
        py = min(max(py, 300), H - 300 - lay.height)
        return lay, (px, py + drop)

    def cta(self, t):
        p = ease_out_back(min(1.0, (t - 57.15) / 0.35))
        if p <= 0.02:
            return None, (0, 0)
        f = font("black", 62)
        lay = Image.new("RGBA", (940, 280), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        bx0, by0, bx1, by1 = 30, 40, 680, 190
        ld.rounded_rectangle([bx0, by0, bx1, by1], 26, fill=ORANGE + (255,),
                             outline=INK + (255,), width=8)
        ld.text(((bx0 + bx1) / 2, (by0 + by1) / 2), "ПОДПИСАТЬСЯ", font=f,
                fill=(255, 255, 255, 255), anchor="mm")
        cx, cy = bx1 + 90, (by0 + by1) / 2
        ld.rounded_rectangle([cx - 54, cy - 48, cx + 54, cy + 34], 22,
                             fill=(255, 255, 255, 255), outline=INK + (255,), width=7)
        ld.polygon([(cx - 16, cy + 30), (cx + 8, cy + 30), (cx - 4, cy + 66)],
                   fill=(255, 255, 255, 255), outline=INK + (255,))
        ld.line([(cx - 16, cy + 30), (cx + 8, cy + 30)], fill=(255, 255, 255, 255), width=7)
        for i in range(3):
            ld.ellipse([cx - 36 + i * 26, cy - 12, cx - 18 + i * 26, cy + 6],
                       fill=INK + (255,))
        s = 0.82 + 0.18 * p
        lay = lay.resize((int(lay.width * s), int(lay.height * s)), Image.BICUBIC)
        return lay, (540 - lay.width / 2, 1420 - lay.height / 2)


# --------------------------------------------------------------------- render ---
def render(out_path, frames=None, audio=True):
    r = Renderer()
    cmd = [FF, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", os.path.join(OUT, "mix.wav"),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
            "-movflags", "+faststart", "-shortest", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    idlist = frames if frames is not None else range(NF)
    for i in idlist:
        img = r.frame(i)
        proc.stdin.write(img.tobytes())
        if i % 120 == 0:
            print(f"  frame {i}/{NF}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", out_path)


if __name__ == "__main__":
    if "--preview" in sys.argv:
        i = sys.argv.index("--preview")
        times = [float(x) for x in sys.argv[i + 1:]]
        os.makedirs(os.path.join(OUT, "prev"), exist_ok=True)
        r = Renderer()
        for t in times:
            im = r.frame(int(t * FPS))
            p = os.path.join(OUT, "prev", f"t{t:05.2f}.png")
            im.save(p)
            print("saved", p)
    else:
        name = "master.mp4"
        if "--out" in sys.argv:
            name = sys.argv[sys.argv.index("--out") + 1]
        render(os.path.join(OUT, name), audio="--noaudio" not in sys.argv)
