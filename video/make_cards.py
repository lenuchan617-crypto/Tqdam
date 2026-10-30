"""
Derived assets for shots 8-16 of «ОДИН НА ЗЕМЛЕ».

Sources: the two generated design sheets (hero / dog) + procedural vector-ish
comic cards drawn with PIL, so every beat of the 60 s cut has its own image
in one consistent palette, halftone and inked-outline language.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont

W, H = 1080, 1920
M = 1.6
BW, BH = int(W * M), int(H * M)

ROOT = os.path.dirname(os.path.abspath(__file__))
KF = os.path.join(ROOT, "keyframes")
REF = os.path.join(ROOT, "ref")
FONTS = os.path.join(ROOT, "fonts")

INK = (25, 25, 39)
ORANGE = (255, 107, 53)
YELLOW = (232, 255, 71)
TEAL = (66, 221, 224)
PURPLE = (105, 80, 161)
PAPER = (244, 238, 226)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


# ------------------------------------------------------------------ utilities ---
def halftone(size, dot=9, color=(0, 0, 0), angle=27, density=0.55, radius_max=4.6):
    """Transparent halftone dot sheet."""
    w, h = size
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    th = math.radians(angle)
    step = dot
    n = int((w + h) / step) + 2
    for i in range(-n, n):
        for j in range(-n, n):
            x = i * step * math.cos(th) - j * step * math.sin(th)
            y = i * step * math.sin(th) + j * step * math.cos(th)
            if -10 <= x <= w + 10 and -10 <= y <= h + 10:
                d.ellipse([x - radius_max, y - radius_max, x + radius_max, y + radius_max],
                          fill=color + (int(255 * density),))
    return lay


def vgrad(size, top, bottom):
    w, h = size
    t = np.linspace(0, 1, h)[:, None]
    arr = (np.array(top)[None, :] * (1 - t) + np.array(bottom)[None, :] * t)
    arr = np.repeat(arr[:, None, :], w, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def burst(size, center, color, rays=26, inner=0.06, seed=3, alpha=255, spread=1.0):
    w, h = size
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    cx, cy = center
    R = math.hypot(w, h) * 1.2 * spread
    step = 2 * math.pi / rays
    for i in range(rays):
        if i % 2:
            continue
        a0 = i * step - math.pi / 2
        a1 = a0 + step
        d.polygon([(cx, cy),
                   (cx + R * math.cos(a0), cy + R * math.sin(a0)),
                   (cx + R * math.cos(a1), cy + R * math.sin(a1))],
                  fill=color + (alpha,))
    return lay


def ink_round(d, box, fill, width=8, r=28, outline=INK):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def drop_shadow(img, blur=18, offset=(12, 16), darkness=0.35):
    alpha = img.split()[-1]
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh.putalpha(alpha.point(lambda v: int(v * darkness)))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    base = Image.new("RGBA", img.size, (0, 0, 0, 0))
    base.alpha_composite(sh, offset)
    base.alpha_composite(img)
    return base


# --------------------------------------------------------------------- cutouts ---
def cutout(path, box, tol=26, feather=2, grow=True):
    im = Image.open(path).convert("RGB").crop(box)
    arr = np.asarray(im).astype(np.int16)
    # background colour = median of the four corners
    corners = np.concatenate([arr[:8, :8].reshape(-1, 3), arr[:8, -8:].reshape(-1, 3),
                              arr[-8:, :8].reshape(-1, 3), arr[-8:, -8:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.abs(arr - bg).sum(axis=2)
    mask = (dist > tol * 3).astype(np.uint8) * 255
    m = Image.fromarray(mask, "L")
    if grow:
        m = m.filter(ImageFilter.MaxFilter(5))
    m = m.filter(ImageFilter.MinFilter(7))   # eat the pale background rim
    m = m.filter(ImageFilter.MaxFilter(3))
    m = m.filter(ImageFilter.GaussianBlur(feather))
    out = im.convert("RGBA")
    out.putalpha(m)
    bbox = m.getbbox()
    if bbox:
        out = out.crop(bbox)
    return out


HERO_SHEET = os.path.join(REF, "hero_sheet.png")
DOG_SHEET = os.path.join(REF, "dog_sheet.png")


# ---------------------------------------------------------------------- cards ---
def card_pharmacy():
    """Shot 09 — medicine cabinet, comic shelf straight-on."""
    img = Image.new("RGB", (BW, BH), (226, 243, 238))
    img = ImageChops.multiply(img, vgrad((BW, BH), (255, 255, 255), (206, 228, 222)))
    img = img.convert("RGBA")
    d = ImageDraw.Draw(img)

    # cabinet body
    ink_round(d, [BW * 0.05, BH * 0.06, BW * 0.95, BH * 0.94], (247, 250, 249), width=12, r=40)
    # shelves
    shelves = [0.20, 0.40, 0.60, 0.80]
    for i, sy in enumerate(shelves):
        y = BH * sy
        d.rectangle([BW * 0.08, y, BW * 0.92, y + BH * 0.012], fill=(200, 214, 210),
                    outline=INK, width=8)

    # products per shelf (unbranded)
    def box(cx, cy, w, h, col, cross=True):
        ink_round(d, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], col, width=7, r=14)
        if cross:
            s = w * 0.22
            d.rectangle([cx - s, cy - s / 3, cx + s, cy + s / 3], fill=(226, 62, 62))
            d.rectangle([cx - s / 3, cy - s, cx + s / 3, cy + s], fill=(226, 62, 62))

    cols = [TEAL, PAPER, (255, 205, 120), (198, 226, 168), (240, 226, 246)]
    xs = [0.18, 0.36, 0.55, 0.73, 0.87]
    for r, sy in enumerate(shelves):
        for i, x in enumerate(xs):
            if (r + i) % 3 == 2:
                continue
            box(BW * x, BH * (sy - 0.055), BW * 0.11, BH * 0.075, cols[(r * 5 + i) % 5])
    # bandage rolls
    for k in range(3):
        cx, cy = BW * (0.22 + 0.16 * k), BH * 0.36
        d.ellipse([cx - BW * 0.05, cy - BW * 0.05, cx + BW * 0.05, cy + BW * 0.05],
                  fill=(255, 232, 208), outline=INK, width=8)
        d.ellipse([cx - BW * 0.018, cy - BW * 0.018, cx + BW * 0.018, cy + BW * 0.018],
                  fill=(214, 170, 140), outline=INK, width=5)

    # the boss tooth
    tx, ty, s = BW * 0.52, BH * 0.60, BW * 0.26
    tooth = [(tx - s * 0.5, ty - s * 0.55), (tx - s * 0.55, ty + s * 0.05),
             (tx - s * 0.30, ty + s * 0.62), (tx - s * 0.08, ty + s * 0.18),
             (tx + s * 0.12, ty + s * 0.62), (tx + s * 0.40, ty + s * 0.05),
             (tx + s * 0.5, ty - s * 0.55)]
    d.polygon(tooth, fill=(255, 255, 255), outline=INK)
    d.line(tooth + [tooth[0]], fill=INK, width=12, joint="curve")
    # crown
    crown = [(tx - s * 0.42, ty - s * 0.55), (tx - s * 0.46, ty - s * 0.95),
             (tx - s * 0.20, ty - s * 0.78), (tx, ty - s * 1.05),
             (tx + s * 0.20, ty - s * 0.78), (tx + s * 0.46, ty - s * 0.95),
             (tx + s * 0.42, ty - s * 0.55)]
    d.polygon(crown, fill=YELLOW, outline=INK)
    d.line(crown + [crown[0]], fill=INK, width=12, joint="curve")
    # angry eyes
    for sgn in (-1, 1):
        ex = tx + sgn * s * 0.17
        d.line([(ex - s * 0.09, ty - s * 0.26), (ex + s * 0.09, ty - s * 0.15)],
               fill=INK, width=11)
        d.ellipse([ex - s * 0.05, ty - s * 0.12, ex + s * 0.05, ty + s * 0.02],
                  fill=INK)
    # grimace
    d.arc([tx - s * 0.16, ty - s * 0.02, tx + s * 0.16, ty + s * 0.22], 200, 340,
          fill=INK, width=10)

    img = img.convert("RGBA")
    ht = halftone((BW, BH), dot=13, color=(30, 80, 70), density=0.20, radius_max=5)
    img = Image.alpha_composite(img, ht)
    return img.convert("RGB")


def card_reaction():
    """Shot 10 — bewildered close-up (design-sheet cutout) + giant bulb."""
    hero = cutout(HERO_SHEET, (390, 880, 730, 1290))
    bg = vgrad((BW, BH), (28, 30, 56), (12, 12, 24)).convert("RGBA")
    bg.alpha_composite(burst((BW, BH), (BW * 0.5, BH * 0.42), TEAL, rays=30, alpha=70))
    bg.alpha_composite(burst((BW, BH), (BW * 0.5, BH * 0.42), YELLOW, rays=14, alpha=40))

    d = ImageDraw.Draw(bg)
    # rubber bulb – pear silhouette, held at chest height on the right
    bx, by, bs = BW * 0.70, BH * 0.80, BW * 0.19
    d.ellipse([bx - bs, by - bs * 0.80, bx + bs, by + bs * 0.80], fill=(240, 138, 150),
              outline=INK, width=12)
    d.ellipse([bx - bs * 0.42, by - bs * 0.40, bx + bs * 0.12, by + bs * 0.16],
              fill=(250, 180, 190))
    d.rounded_rectangle([bx - bs * 0.24, by - bs * 1.12, bx + bs * 0.24, by - bs * 0.74],
                        16, fill=(216, 216, 228), outline=INK, width=12)
    d.ellipse([bx - bs * 0.11, by - bs * 1.36, bx + bs * 0.11, by - bs * 1.08],
              fill=(216, 216, 228), outline=INK, width=10)
    # neutral product label (text is added by the compositor)
    lw, lh = bs * 1.30, bs * 0.52
    d.rounded_rectangle([bx - lw / 2, by - lh * 0.10, bx + lw / 2, by + lh * 0.90],
                        14, fill=(252, 250, 244), outline=INK, width=8)

    hw = int(BW * 0.56)
    hh = int(hero.height * hw / hero.width)
    hero = hero.resize((hw, hh), Image.LANCZOS)
    hero = drop_shadow(hero)
    bg.alpha_composite(hero, (int(BW * 0.44 - hw / 2), int(BH * 0.28 - hh / 2)))
    return bg.convert("RGB")


def card_books():
    """Shot 11 — three survival manuals."""
    bg = vgrad((BW, BH), (58, 42, 96), (26, 20, 42)).convert("RGBA")
    bg.alpha_composite(halftone((BW, BH), dot=14, color=(255, 255, 255), density=0.10,
                                radius_max=4))
    d = ImageDraw.Draw(bg)

    def book(cx, cy, w, h, col, angle, title, icon):
        lay = Image.new("RGBA", (int(w * 1.4), int(h * 1.4)), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        x0, y0 = w * 0.2, h * 0.2
        ld.rounded_rectangle([x0, y0, x0 + w, y0 + h], 18, fill=col, outline=INK, width=12)
        ld.rectangle([x0, y0, x0 + w * 0.09, y0 + h], fill=(0, 0, 0, 45))
        # icon plate
        ld.rounded_rectangle([x0 + w * 0.18, y0 + h * 0.12, x0 + w * 0.82, y0 + h * 0.44],
                             14, fill=(255, 255, 255, 225), outline=INK, width=6)
        icon(ld, x0 + w * 0.5, y0 + h * 0.28, w * 0.22)
        # title
        f = font("Oswald-Bold.ttf", int(w * 0.13))
        lines = title.split("\n")
        ty = y0 + h * 0.56
        for ln in lines:
            bb = ld.textbbox((0, 0), ln, font=f)
            ld.text((x0 + w / 2 - (bb[2] - bb[0]) / 2, ty), ln, font=f, fill=INK)
            ty += (bb[3] - bb[1]) * 1.35
        lay = lay.rotate(angle, expand=True, resample=Image.BICUBIC)
        lay = drop_shadow(lay, blur=16, offset=(10, 14), darkness=0.4)
        bg.alpha_composite(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)))

    def icon_gear(ld, cx, cy, s):
        for i in range(8):
            a = i * math.pi / 4
            ld.rectangle([cx + s * 0.7 * math.cos(a) - s * 0.13,
                          cy + s * 0.7 * math.sin(a) - s * 0.13,
                          cx + s * 0.7 * math.cos(a) + s * 0.13,
                          cy + s * 0.7 * math.sin(a) + s * 0.13], fill=INK)
        ld.ellipse([cx - s * 0.62, cy - s * 0.62, cx + s * 0.62, cy + s * 0.62],
                   fill=INK)
        ld.ellipse([cx - s * 0.24, cy - s * 0.24, cx + s * 0.24, cy + s * 0.24],
                   fill=(255, 255, 255))

    def icon_sprout(ld, cx, cy, s):
        ld.line([(cx, cy + s * 0.8), (cx, cy - s * 0.1)], fill=(60, 130, 60), width=int(s * 0.16))
        ld.ellipse([cx - s * 0.8, cy - s * 0.55, cx, cy + s * 0.05], fill=(90, 170, 90),
                   outline=INK, width=5)
        ld.ellipse([cx, cy - s * 0.75, cx + s * 0.8, cy - s * 0.1], fill=(120, 195, 110),
                   outline=INK, width=5)

    def icon_drop(ld, cx, cy, s):
        ld.polygon([(cx, cy - s * 0.85), (cx + s * 0.6, cy + s * 0.15),
                    (cx + s * 0.3, cy + s * 0.8), (cx - s * 0.3, cy + s * 0.8),
                    (cx - s * 0.6, cy + s * 0.15)], fill=(110, 200, 235), outline=INK)
        ld.line([(cx, cy - s * 0.85), (cx + s * 0.6, cy + s * 0.15),
                 (cx + s * 0.3, cy + s * 0.8), (cx - s * 0.3, cy + s * 0.8),
                 (cx - s * 0.6, cy + s * 0.15), (cx, cy - s * 0.85)], fill=INK, width=7)

    book(BW * 0.30, BH * 0.24, BW * 0.46, BH * 0.32, ORANGE, 9, "МЕХА-\nНИКА", icon_gear)
    book(BW * 0.68, BH * 0.52, BW * 0.46, BH * 0.32, TEAL, -8, "ЗЕМЛЕ-\nДЕЛИЕ", icon_sprout)
    book(BW * 0.36, BH * 0.80, BW * 0.46, BH * 0.32, (250, 214, 120), 5, "ФИЛЬТР\nВОДЫ", icon_drop)

    # crossed-out wifi
    wx, wy, ws = BW * 0.80, BH * 0.12, BW * 0.09
    for r, wdt in ((1.0, 12), (0.66, 12), (0.32, 12)):
        d.arc([wx - ws * r, wy - ws * r * 0.9, wx + ws * r, wy + ws * r * 1.1],
              205, 335, fill=(255, 255, 255), width=wdt)
    d.ellipse([wx - 12, wy + ws * 0.62, wx + 12, wy + ws * 0.62 + 24], fill=(255, 255, 255))
    d.line([(wx - ws * 1.1, wy - ws * 1.15), (wx + ws * 1.1, wy + ws * 1.15)],
           fill=(226, 62, 62), width=18)
    return bg.convert("RGB")


def card_drive():
    """Shot 13 — dashboard gauge + night road."""
    bg = vgrad((BW, BH), (24, 26, 48), (10, 10, 18)).convert("RGBA")
    d = ImageDraw.Draw(bg)
    # perspective road stripes converging on the horizon
    vx, vy = BW * 0.5, BH * 0.55
    for i in range(14):
        t0 = 0.06 + i * 0.075
        t1 = t0 + 0.035
        hw0 = BW * 0.012 * (1 + t0 * 9)
        hw1 = BW * 0.012 * (1 + t1 * 9)
        y0 = vy + (BH - vy) * t0 ** 1.6
        y1 = vy + (BH - vy) * t1 ** 1.6
        d.polygon([(vx - hw0, y0), (vx + hw0, y0), (vx + hw1, y1), (vx - hw1, y1)],
                  fill=(240, 240, 200, 110))
    # speed lines
    for i in range(24):
        a = -math.pi / 2 + (i / 23 - 0.5) * 2.4
        r0, r1 = BW * 0.55, BW * 1.1
        d.line([(BW * 0.5 + r0 * math.cos(a), BH * 0.55 + r0 * math.sin(a)),
                (BW * 0.5 + r1 * math.cos(a), BH * 0.55 + r1 * math.sin(a))],
               fill=(255, 200, 120, 60), width=8)
    # gauge
    cx, cy, R = BW * 0.5, BH * 0.42, BW * 0.40
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(18, 18, 30), outline=ORANGE, width=18)
    d.ellipse([cx - R * 0.86, cy - R * 0.86, cx + R * 0.86, cy + R * 0.86],
              outline=(90, 80, 110), width=6)
    for i in range(21):
        a = math.radians(140 + i * (260 / 20))
        r0 = R * (0.72 if i % 2 == 0 else 0.79)
        col = (226, 62, 62) if i >= 16 else YELLOW
        d.line([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
                (cx + R * 0.86 * math.cos(a), cy + R * 0.86 * math.sin(a))],
               fill=col, width=10)
    a = math.radians(140 + 17.2 * (260 / 20))
    d.line([(cx, cy), (cx + R * 0.74 * math.cos(a), cy + R * 0.74 * math.sin(a))],
           fill=ORANGE, width=18)
    d.ellipse([cx - R * 0.09, cy - R * 0.09, cx + R * 0.09, cy + R * 0.09],
              fill=ORANGE, outline=INK, width=6)
    return bg.convert("RGB")


def card_garden():
    """Shot 14 — potato patch, shovel hero, the found dog, filter bottle."""
    bg = vgrad((BW, BH), (255, 222, 158), (255, 178, 112)).convert("RGBA")
    d = ImageDraw.Draw(bg)

    # sun + haze
    d.ellipse([BW * 0.66, BH * 0.03, BW * 0.66 + BW * 0.26, BH * 0.03 + BW * 0.26],
              fill=(255, 248, 208))
    bg.alpha_composite(burst((BW, BH), (BW * 0.79, BH * 0.10), (255, 236, 170), rays=20,
                             alpha=55))

    # distant hills
    d.ellipse([-BW * 0.35, BH * 0.52, BW * 0.75, BH * 0.90], fill=(172, 202, 124))
    d.ellipse([BW * 0.40, BH * 0.57, BW * 1.35, BH * 0.95], fill=(154, 190, 112))
    d.polygon([(0, BH * 0.66), (BW, BH * 0.62), (BW, BH), (0, BH)], fill=(186, 166, 116))

    # perspective potato rows
    vx, vy = BW * 0.52, BH * 0.62
    for k in range(6):
        t0 = k / 6.6
        t1 = min(1.0, t0 + 0.115)
        y0 = vy + (BH - vy) * t0 ** 1.5
        y1 = vy + (BH - vy) * t1 ** 1.5
        h0 = BW * 0.03 * (1 + t0 * 8)
        h1 = BW * 0.03 * (1 + t1 * 8)
        d.polygon([(vx - h0, y0), (vx + h0, y0), (vx + h1, y1), (vx - h1, y1)],
                  fill=(158, 112, 74))
        for j in range(5):
            ty = (t0 + t1) / 2
            py = vy + (BH - vy) * ty ** 1.5
            span = BW * 0.03 * (1 + ty * 8) * 0.78
            px = vx + span * ((j / 2) - 1) * 1.05
            rx = BW * 0.016 * (1 + ty * 2.4)
            d.ellipse([px - rx, py - rx * 0.68, px + rx, py + rx * 0.68],
                      fill=(206, 160, 102), outline=(126, 88, 54), width=4)

    # hero with straw hat + shovel
    hero = cutout(HERO_SHEET, (60, 40, 380, 880))
    hh = int(BH * 0.40)
    hw = int(hero.width * hh / hero.height)
    hero = hero.resize((hw, hh), Image.LANCZOS)
    hero = drop_shadow(hero, blur=16, offset=(10, 16), darkness=0.35)
    hx0, hy0 = int(BW * 0.30 - hw / 2), int(BH * 0.79 - hh)
    bg.alpha_composite(hero, (hx0, hy0))
    d = ImageDraw.Draw(bg)  # re-bind: alpha_composite swapped the buffer
    # straw hat
    cx, cy = hx0 + hw * 0.5, hy0 + hh * 0.040
    d.ellipse([cx - hw * 0.40, cy - hw * 0.09, cx + hw * 0.40, cy + hw * 0.10],
              fill=(240, 206, 120), outline=INK, width=9)
    d.polygon([(cx - hw * 0.26, cy), (cx + hw * 0.26, cy),
               (cx + hw * 0.18, cy - hw * 0.22), (cx - hw * 0.18, cy - hw * 0.22)],
              fill=(248, 216, 134), outline=INK)
    d.line([(cx - hw * 0.26, cy - hw * 0.05), (cx + hw * 0.26, cy - hw * 0.05)],
           fill=(196, 150, 80), width=6)
    # shovel
    sx, sy = cx + hw * 0.42, cy + hh * 0.52
    d.line([(sx, sy - hh * 0.30), (sx + BW * 0.05, BH * 0.955)],
           fill=(150, 104, 62), width=16)
    d.polygon([(sx + BW * 0.045, BH * 0.93), (sx + BW * 0.075, BH * 0.98),
               (sx + BW * 0.045, BH * 1.0), (sx + BW * 0.012, BH * 0.955)],
              fill=(210, 214, 220), outline=INK)
    # dirt clod flying
    d.ellipse([sx + BW * 0.06, BH * 0.90, sx + BW * 0.10, BH * 0.925],
              fill=(132, 92, 60), outline=INK, width=4)

    # the dog, on the ground, front-right
    dog = cutout(DOG_SHEET, (60, 40, 740, 470))
    dw = int(BW * 0.52)
    dh = int(dog.height * dw / dog.width)
    dog = dog.resize((dw, dh), Image.LANCZOS)
    dog = drop_shadow(dog, blur=14, offset=(8, 18), darkness=0.4)
    bg.alpha_composite(dog, (int(BW * 0.60 - dw / 2), int(BH * 0.80 - dh / 2)))
    d = ImageDraw.Draw(bg)

    # water filter bottle, top-right
    fx, fy = BW * 0.84, BH * 0.20
    d.rounded_rectangle([fx - BW * 0.055, fy - BH * 0.055, fx + BW * 0.055, fy + BH * 0.05],
                        22, fill=(212, 242, 242), outline=INK, width=9)
    d.rectangle([fx - BW * 0.04, fy + BH * 0.005, fx + BW * 0.04, fy + BH * 0.022],
                fill=(112, 196, 220), outline=INK, width=5)
    d.rounded_rectangle([fx - BW * 0.03, fy - BH * 0.075, fx + BW * 0.03, fy - BH * 0.052],
                        10, fill=TEAL, outline=INK, width=7)
    for k in range(2):
        d.ellipse([fx + BW * 0.05, fy + BH * (0.03 + k * 0.03) - 8,
                   fx + BW * 0.05 + 16, fy + BH * (0.03 + k * 0.03) + 8],
                  fill=(150, 220, 240), outline=INK, width=3)

    bg.alpha_composite(halftone((BW, BH), dot=12, color=(120, 70, 40), density=0.13,
                                radius_max=4))
    return bg.convert("RGB")


def card_sunset():
    """Shot 15 — the balcony: hero, the dog, a solar lamp, warm dusk."""
    bg = vgrad((BW, BH), (255, 168, 116), (96, 66, 132)).convert("RGBA")
    d = ImageDraw.Draw(bg)

    # sun disk + soft rings
    sx, sy, sr = BW * 0.46, BH * 0.34, BW * 0.15
    rings = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rings)
    for k in range(1, 5):
        rr = sr * (1 + k * 0.40)
        rd.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=(255, 226, 170, 70),
                   width=int(BW * 0.006))
    rings = rings.filter(ImageFilter.GaussianBlur(BW * 0.004))
    bg.alpha_composite(rings)
    d = ImageDraw.Draw(bg)
    d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 240, 184))

    # far hills + river
    d.ellipse([-BW * 0.4, BH * 0.44, BW * 0.8, BH * 0.62], fill=(150, 96, 132))
    d.ellipse([BW * 0.5, BH * 0.47, BW * 1.4, BH * 0.64], fill=(132, 84, 124))
    d.polygon([(0, BH * 0.55), (BW, BH * 0.52), (BW, BH * 0.66), (0, BH * 0.70)],
              fill=(104, 132, 186))
    for i in range(14):
        y = BH * (0.555 + i * 0.0075)
        x = BW * (0.06 + (i * 0.41) % 0.82)
        d.line([(x, y), (x + BW * 0.10, y)], fill=(226, 210, 226), width=4)

    # the house on the far bank
    hx, hy = BW * 0.66, BH * 0.520
    d.rectangle([hx, hy - BH * 0.045, hx + BW * 0.16, hy], fill=(52, 44, 74))
    d.polygon([(hx - BW * 0.018, hy - BH * 0.045), (hx + BW * 0.08, hy - BH * 0.082),
               (hx + BW * 0.178, hy - BH * 0.045)], fill=(38, 32, 56))
    d.rectangle([hx + BW * 0.050, hy - BH * 0.036, hx + BW * 0.095, hy - BH * 0.014],
                fill=(255, 218, 130))

    # balcony floor
    d.rectangle([0, BH * 0.560, BW, BH], fill=(78, 56, 82))
    for i in range(7):
        d.line([(0, BH * (0.60 + i * 0.055)), (BW, BH * (0.585 + i * 0.055))],
               fill=(64, 46, 68), width=6)

    # hero, standing at the rail
    hero = cutout(HERO_SHEET, (60, 40, 380, 880))
    hh = int(BH * 0.40)
    hw = int(hero.width * hh / hero.height)
    hero = hero.resize((hw, hh), Image.LANCZOS)
    hero = drop_shadow(hero, blur=18, offset=(0, 12), darkness=0.30)
    bg.alpha_composite(hero, (int(BW * 0.34), int(BH * 0.80 - hh)))

    # the dog, sitting beside him
    dog = cutout(DOG_SHEET, (250, 455, 560, 800))
    dw = int(BW * 0.29)
    dh = int(dog.height * dw / dog.width)
    dog = dog.resize((dw, dh), Image.LANCZOS)
    dog = drop_shadow(dog, blur=12, offset=(0, 8), darkness=0.30)
    bg.alpha_composite(dog, (int(BW * 0.64), int(BH * 0.80 - dh)))

    # the balcony rail, in front of them
    d = ImageDraw.Draw(bg)
    d.rectangle([0, BH * 0.655, BW, BH * 0.672], fill=(46, 34, 58))
    for i in range(8):
        x = BW * (0.03 + i * 0.132)
        d.rectangle([x, BH * 0.672, x + BW * 0.022, BH * 0.80], fill=(46, 34, 58))
    d.rectangle([0, BH * 0.792, BW, BH * 0.812], fill=(38, 28, 50))

    # small table + solar lamp on the left
    lx, ly = BW * 0.86, BH * 0.585
    d.rectangle([lx - BW * 0.012, ly - BH * 0.075, lx + BW * 0.012, ly], fill=(58, 42, 60))
    d.ellipse([lx - BW * 0.075, ly - BH * 0.088, lx + BW * 0.075, ly - BH * 0.066],
              fill=(70, 52, 72), outline=INK, width=6)
    d.polygon([(lx - BW * 0.040, ly - BH * 0.088), (lx + BW * 0.040, ly - BH * 0.088),
               (lx + BW * 0.026, ly - BH * 0.138), (lx - BW * 0.026, ly - BH * 0.138)],
              fill=(255, 240, 190), outline=INK, width=5)
    lamp_glow = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    gd = ImageDraw.Draw(lamp_glow)
    gd.ellipse([lx - BW * 0.13, ly - BH * 0.175, lx + BW * 0.13, ly - BH * 0.105],
               fill=(255, 226, 150, 150))
    lamp_glow = lamp_glow.filter(ImageFilter.GaussianBlur(BW * 0.03))
    bg = Image.alpha_composite(bg, lamp_glow)

    # stars
    rng = np.random.RandomState(9)
    for _ in range(26):
        x, y = rng.uniform(0, BW), rng.uniform(0, BH * 0.30)
        r = rng.uniform(1.5, 3.5)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 250, 235, 170))

    bg.alpha_composite(halftone((BW, BH), dot=12, color=(70, 36, 60), density=0.11,
                                radius_max=4))
    return bg.convert("RGB")


def card_portrait():
    """Shot 16 — frontal portrait for the CTA."""
    hero = cutout(HERO_SHEET, (390, 880, 730, 1290))
    bg = vgrad((BW, BH), (255, 152, 92), (216, 96, 60)).convert("RGBA")
    bg.alpha_composite(burst((BW, BH), (BW * 0.5, BH * 0.40), (255, 210, 120), rays=28,
                             alpha=90))
    bg.alpha_composite(halftone((BW, BH), dot=14, color=(120, 50, 30), density=0.14,
                               radius_max=4))
    hw = int(BW * 1.06)
    hh = int(hero.height * hw / hero.width)
    hero = hero.resize((hw, hh), Image.LANCZOS)
    hero = drop_shadow(hero, blur=26, offset=(0, 22), darkness=0.30)
    bg.alpha_composite(hero, (int(BW * 0.5 - hw / 2), int(BH * 0.46 - hh / 2)))
    return bg.convert("RGB")


def card_rooftop_day():
    """Shot 8 — the night rooftop from shot 7, re-lit at dawn."""
    base = Image.open(os.path.join(KF, "k07.png")).convert("RGB").resize((BW, BH), Image.LANCZOS)
    arr = np.asarray(base).astype(np.float32) / 255.0
    # lift shadows, warm the sky, keep the ink lines
    lum = arr.mean(axis=2)
    lift = np.clip(lum * 1.45 + 0.30, 0, 1)[:, :, None]
    warm = np.array([1.18, 1.02, 0.80])[None, None, :]
    arr = np.clip(arr * lift * warm, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGBA")
    # warm morning sun with a soft bloom in the upper right
    glow = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy, r = BW * 0.80, BH * 0.10, BW * 0.42
    gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 226, 150, 170))
    gd.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45],
               fill=(255, 246, 205, 235))
    glow = glow.filter(ImageFilter.GaussianBlur(BW * 0.05))
    img = Image.alpha_composite(img, glow)
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    arr = np.clip(arr * np.array([1.10, 1.04, 0.90])[None, None, :], 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGBA")

    # autonomy kit on the roof: generator + folded-open panel
    d = ImageDraw.Draw(img)
    gx, gy, gw, gh = BW * 0.10, BH * 0.735, BW * 0.24, BH * 0.095
    d.rectangle([gx + gw * 0.16, gy + gh, gx + gw * 0.22, gy + gh + BH * 0.020],
                fill=(44, 38, 60))
    d.rectangle([gx + gw * 0.76, gy + gh, gx + gw * 0.82, gy + gh + BH * 0.020],
                fill=(44, 38, 60))
    d.rounded_rectangle([gx, gy, gx + gw, gy + gh], 22, fill=(74, 66, 92), outline=INK,
                        width=11)
    d.rounded_rectangle([gx + gw * 0.10, gy + gh * 0.20, gx + gw * 0.58, gy + gh * 0.68],
                        12, fill=(38, 34, 52), outline=INK, width=6)
    d.ellipse([gx + gw * 0.72, gy + gh * 0.32, gx + gw * 0.90, gy + gh * 0.54],
              fill=(226, 62, 62), outline=INK, width=6)
    # solar panel standing behind the generator, two cells + stand
    px, py = BW * 0.60, BH * 0.665
    cw_, ch_ = BW * 0.115, BH * 0.085
    for k in range(2):
        x0 = px + k * cw_ * 1.04
        d.polygon([(x0, py), (x0 + cw_, py + BH * 0.012),
                   (x0 + cw_ * 0.94, py + ch_), (x0, py + ch_ * 0.96)],
                  fill=(62, 98, 148), outline=INK, width=8)
        d.line([(x0 + cw_ * 0.06, py + ch_ * 0.50), (x0 + cw_ * 0.92, py + ch_ * 0.52)],
               fill=(160, 200, 240), width=4)
    d.line([(px + cw_ * 0.5, py + ch_ * 0.96), (px + cw_ * 0.5, BH * 0.745)],
           fill=INK, width=8)
    return img.convert("RGB")


def main():
    jobs = {
        "k08": card_rooftop_day,
        "k09": card_pharmacy,
        "k10": card_reaction,
        "k11": card_books,
        "k13": card_drive,
        "k14": card_garden,
        "k15": card_sunset,
        "k16": card_portrait,
    }
    for name, fn in jobs.items():
        im = fn()
        if im.size != (BW, BH):
            im = im.resize((BW, BH), Image.LANCZOS)
        p = os.path.join(KF, name + ".png")
        im.save(p)
        print("wrote", p, im.size)


if __name__ == "__main__":
    main()
