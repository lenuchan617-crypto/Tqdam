"""Render the 5-minute Shadow Slave animatic: AI keyframes + camera moves,
film grain, colour grade, subtitles and burned-in titles.

Usage:
    python renderVideo.py            # full render -> ../shadowSlave.mp4
    python renderVideo.py stats      # per-shot luminance check, no output
"""
import math
import os
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
TOTAL = 300.0
NF = int(TOTAL * FPS)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FRAMES = os.path.join(ROOT, "frames")
AUDIO = os.path.join(ROOT, "audio", "ambience.wav")
OUT = os.path.join(ROOT, "shadowSlave.mp4")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTS = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

rng = np.random.default_rng(31337)


def lerp(a, b, u):
    return a + (b - a) * u


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------ shots
def S(img, ts, te, fi, fo, z, c, **kw):
    d = dict(img=img, ts=ts, te=te, fi=fi, fo=fo, z=z, c=c)
    d.update(sat=0.6, con=1.06, exp=1.0, tint=(1.0, 1.0, 1.0), grain=0.05, lift=0.0,
             vig=0.34, flicker=0, dark=1.0, shake=(), flash=(), kind="img")
    d.update(kw)
    return d


SHOTS = [
    # --- scene 1: the last cup of coffee
    S("s01_sunny_bench", 0.0, 14.0, 1.2, 1.5, (1.00, 1.12), ((0.50, 0.55), (0.46, 0.50)),
      sat=0.62, con=1.06, exp=1.00, tint=(1.00, 0.99, 1.00)),
    S("s02_face_circles", 12.5, 26.0, 1.5, 1.5, (1.00, 1.18), ((0.50, 0.48), (0.50, 0.44)),
      sat=0.58, con=1.08, exp=0.98, tint=(0.99, 0.99, 1.01)),
    S("s01_sunny_bench", 24.5, 40.0, 1.5, 1.2, (1.38, 1.02), ((0.42, 0.58), (0.48, 0.50)),
      sat=0.60, con=1.05, exp=1.00, tint=(1.00, 0.99, 1.00)),
    # --- scene 2: the chair
    S("s03_basement_chair", 38.8, 57.0, 1.2, 1.2, (1.02, 1.16), ((0.50, 0.50), (0.52, 0.48)),
      sat=0.48, con=1.12, exp=0.98, lift=0.018, tint=(0.93, 0.98, 1.08), flicker=1, grain=0.06, vig=0.40),
    S("s04_eyes_closing", 55.8, 71.0, 1.2, 1.0, (1.00, 1.20), ((0.50, 0.50), (0.50, 0.46)),
      sat=0.44, con=1.14, exp=0.94, lift=0.018, tint=(0.92, 0.97, 1.10), flicker=2, grain=0.06, vig=0.42),
    S("s03_basement_chair", 70.0, 76.5, 1.0, 1.4, (1.30, 1.36), ((0.60, 0.45), (0.62, 0.44)),
      sat=0.46, con=1.12, exp=0.96, lift=0.018, tint=(0.93, 0.98, 1.08), flicker=1, grain=0.06, vig=0.42),
    # --- darkness, a point of light, the starry sky
    S(None, 76.3, 83.0, 1.5, 1.5, (1.0, 1.06), ((0.5, 0.5), (0.5, 0.5)),
      kind="star", mode="reveal", sat=0.55, con=1.15, exp=1.08, lift=0.004, tint=(0.88, 0.94, 1.12),
      grain=0.05, vig=0.30),
    # --- scene 3: waking in chains
    S("s05_black_moon_road", 81.5, 98.0, 1.5, 1.5, (1.00, 1.14), ((0.50, 0.52), (0.54, 0.46)),
      sat=0.44, con=1.12, exp=0.92, tint=(0.90, 0.96, 1.12), grain=0.055, vig=0.38),
    S("s06_shackles", 96.5, 110.0, 1.5, 1.5, (1.05, 1.20), ((0.50, 0.50), (0.48, 0.52)),
      sat=0.46, con=1.14, exp=0.88, tint=(0.92, 0.97, 1.10), grain=0.06, vig=0.40),
    S("s07_berries_scientist", 108.5, 122.0, 1.5, 1.2, (1.00, 1.16), ((0.52, 0.52), (0.50, 0.50)),
      sat=0.58, con=1.10, exp=0.92, tint=(0.94, 0.96, 1.06), grain=0.055, vig=0.36),
    S("s05_black_moon_road", 120.8, 130.0, 1.2, 1.0, (1.20, 1.45), ((0.58, 0.40), (0.58, 0.36)),
      sat=0.42, con=1.12, exp=0.86, tint=(0.90, 0.96, 1.13), grain=0.055, vig=0.40),
    # --- scene 4: the mountain king
    S("s08_rockfall_panic", 129.0, 146.0, 1.0, 0.4, (1.00, 1.15), ((0.50, 0.50), (0.48, 0.52)),
      sat=0.44, con=1.12, exp=0.80, tint=(0.92, 0.97, 1.10), grain=0.065, vig=0.42,
      shake=((130.9, 7), (132.4, 11), (134.1, 8), (136.0, 14), (138.2, 9), (141.0, 17), (143.4, 6))),
    S("s09_mountain_king", 145.6, 162.0, 0.4, 1.2, (1.02, 1.22), ((0.50, 0.52), (0.50, 0.56)),
      sat=0.40, con=1.14, exp=0.72, tint=(0.95, 0.95, 1.08), grain=0.06, vig=0.48,
      shake=((147.0, 10), (151.0, 12), (155.0, 13), (159.2, 15))),
    S("s10_chain_trap_hands", 160.8, 178.0, 1.2, 1.5, (1.00, 1.14), ((0.50, 0.50), (0.50, 0.48)),
      sat=0.46, con=1.12, exp=0.82, tint=(0.93, 0.97, 1.09), grain=0.06, vig=0.42),
    S("s09_mountain_king", 176.5, 190.0, 1.5, 1.2, (1.50, 1.08), ((0.50, 0.58), (0.50, 0.50)),
      sat=0.38, con=1.14, exp=0.66, tint=(0.94, 0.95, 1.08), grain=0.06, vig=0.50),
    # --- scene 5: poison and the bell
    S("s11_hero_falls_dawn", 188.8, 206.0, 1.2, 1.5, (1.00, 1.14), ((0.50, 0.50), (0.52, 0.48)),
      sat=0.55, con=1.08, exp=0.96, tint=(1.01, 0.99, 0.98), grain=0.05, vig=0.34),
    S("s11_hero_falls_dawn", 204.5, 220.0, 1.5, 1.2, (1.55, 1.28), ((0.44, 0.56), (0.46, 0.54)),
      sat=0.52, con=1.10, exp=0.92, tint=(1.00, 0.99, 1.00), grain=0.05, vig=0.36),
    S("s12_bell_shadow", 218.8, 232.0, 1.2, 1.5, (1.00, 1.18), ((0.50, 0.50), (0.50, 0.48)),
      sat=0.42, con=1.16, exp=1.22, lift=0.026, tint=(0.90, 0.95, 1.12), grain=0.055, vig=0.36),
    S("s12_bell_shadow", 230.5, 240.0, 1.5, 1.2, (1.55, 1.34), ((0.50, 0.48), (0.50, 0.48)),
      sat=0.40, con=1.20, exp=1.10, lift=0.038, tint=(0.88, 0.94, 1.14), grain=0.05, vig=0.38, dark=0.55,
      flash=((233.0, 0.10, 0.5), (236.2, 0.08, 0.6), (238.6, 0.06, 0.5))),
    # --- scene 6: the evaluation
    S("s13_temple_altar", 238.8, 254.0, 1.2, 1.5, (1.00, 1.16), ((0.50, 0.52), (0.50, 0.48)),
      sat=0.44, con=1.12, exp=0.82, tint=(0.92, 0.96, 1.10), grain=0.055, vig=0.42),
    S("s14_sunny_on_altar", 252.5, 268.0, 1.5, 1.2, (1.00, 1.18), ((0.50, 0.50), (0.50, 0.52)),
      sat=0.55, con=1.10, exp=0.88, tint=(0.93, 0.96, 1.08), grain=0.055, vig=0.44),
    S("s15_shadows_tear", 266.8, 282.0, 1.2, 1.5, (1.05, 1.30), ((0.50, 0.50), (0.50, 0.54)),
      sat=0.42, con=1.16, exp=1.06, lift=0.012, tint=(0.90, 0.95, 1.12), grain=0.06, vig=0.40,
      shake=((268.5, 9), (271.6, 11), (274.8, 13), (278.2, 15)),
      flash=((270.5, 0.16, 0.35), (273.2, 0.18, 0.35), (276.0, 0.20, 0.35), (279.0, 0.22, 0.4))),
    S(None, 280.5, 290.0, 1.5, 0.6, (1.08, 0.98), ((0.5, 0.5), (0.5, 0.5)),
      kind="star", mode="fade", sat=0.55, con=1.14, exp=1.10, lift=0.004, tint=(0.88, 0.94, 1.12),
      grain=0.05, vig=0.28),
    S("s02_face_circles", 289.4, 300.0, 0.6, 1.0, (1.62, 1.76), ((0.50, 0.42), (0.50, 0.40)),
      sat=0.40, con=1.16, exp=0.98, tint=(0.90, 0.96, 1.12), grain=0.055, vig=0.44),
]

# ------------------------------------------------------------------ text
SUBS = [
    (3.0, 6.8, "Горький. Как и вся моя жизнь.", "mono"),
    (7.5, 12.7, "Но я за это заплатил. Значит, допью.", "mono"),
    (12.0, 16.6, "Неделю держался. Дольше не выйдет.", "mono"),
    (18.0, 23.2, "Если засну на улице, превращусь в монстра.", "mono"),
    (27.0, 30.8, "Так что… сдаюсь.", "mono"),
    (52.0, 58.0, "Ещё один. Ладно, грузите.", "spoke"),
    (67.5, 71.1, "Хотя бы попробовал настоящий кофе.", "mono"),
    (84.0, 89.7, "Это не сон. Слишком холодно. Слишком реально.", "mono"),
    (112.5, 121.3, "Это Кровавые ягоды. Растут только там, где пролилась человеческая кровь. Не ешь.", "spoke"),
    (138.0, 141.4, "Беги! Беги же!", "spoke"),
    (152.0, 155.1, "Вот значит, что такое Кошмар.", "mono"),
    (196.0, 200.4, "Сначала старик. Потом ты.", "spoke"),
    (212.0, 219.5, "Нет ничего более жалкого, чем раб, который начал доверять своему хозяину.", "mono"),
    (272.8, 278.8, "Претендент. Твоё испытание окончено.", "spell"),
    (278.3, 284.7, "Ты получил благословение Бога Теней.", "spell"),
    (284.2, 288.9, "Окончательная оценка: Великолепно.", "spell"),
    (288.4, 294.9, "Твоё истинное имя: Лишённый Света.", "spell"),
    (295.0, 297.7, "…Лишённый Света.", "whisper"),
]

TITLES = [
    (1.0, 5.6, "ТЕНЕВОЙ РАБ", "title"),
    (1.6, 5.6, "главы 1–5", "subtitle"),
    (292.4, 300.0, "ТЕНЕВОЙ РАБ", "endmain"),
    (292.4, 300.0, "Глава 1. Конец.", "endsub"),
    (292.4, 300.0, "Он выжил. В этот раз.", "endsub2"),
]


def loadFont(path, size):
    return ImageFont.truetype(path, size)


F_SUB = loadFont(FONT, 46)
F_SUB_S = loadFont(FONT, 40)
F_SPELL = loadFont(FONTS, 50)
F_TITLE = loadFont(FONTB, 96)
F_TITLE_S = loadFont(FONT, 42)
F_END_S = loadFont(FONT, 44)


def textWidth(d, s, font):
    return d.textbbox((0, 0), s, font=font)[2]


def drawSpaced(d, xy, text, font, fill, spacing, anchor="mm"):
    x, y = xy
    total = sum(textWidth(d, ch, font) for ch in text) + spacing * (len(text) - 1)
    cx = x - total / 2 if anchor == "mm" else x
    for ch in text:
        d.text((cx, y), ch, font=font, fill=fill, anchor="lm")
        cx += textWidth(d, ch, font) + spacing


def wrap(d, text, font, maxw):
    words = text.split()
    lines, cur = [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if textWidth(d, t, font) > maxw and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines[:2]


def drawTexts(im, t):
    d = ImageDraw.Draw(im)
    # ---- burned-in titles
    for t0, t1, text, kind in TITLES:
        if not (t0 <= t < t1):
            continue
        u = (t - t0) / (t1 - t0)
        a = min(smooth((t - t0) / 1.1), smooth((t1 - t) / 0.9))
        a = max(0.0, min(1.0, a))
        if a <= 0.01:
            continue
        if kind == "title":
            fill = (int(226 * a), int(226 * a), int(220 * a))
            drawSpaced(d, (W // 2, 452), text, F_TITLE, fill, 14)
        elif kind == "subtitle":
            fill = (int(178 * a), int(178 * a), int(175 * a))
            drawSpaced(d, (W // 2, 552), text.upper(), F_TITLE_S, fill, 8)
        elif kind == "endmain":
            fill = (int(222 * a), int(222 * a), int(216 * a))
            drawSpaced(d, (W // 2, 430), text, F_TITLE, fill, 14)
        elif kind == "endsub":
            fill = (int(196 * a), int(196 * a), int(192 * a))
            drawSpaced(d, (W // 2, 546), text, F_END_S, fill, 4)
        else:
            fill = (int(150 * a), int(150 * a), int(152 * a))
            drawSpaced(d, (W // 2, 606), text, F_END_S, fill, 4)
    # ---- subtitles
    for t0, t1, text, kind in SUBS:
        if not (t0 <= t < t1):
            continue
        a = min(smooth((t - t0) / 0.35), smooth((t1 - t) / 0.45))
        a = max(0.0, min(1.0, a))
        if a <= 0.01:
            continue
        font = F_SUB
        if kind == "spell":
            font = F_SPELL
        lines = wrap(d, text, font, 1520)
        if len(lines) > 1:
            font = F_SUB_S if kind != "spell" else F_SPELL
            lines = wrap(d, text, font, 1400)
        if kind == "spell":
            col = (int(214 * a), int(226 * a), int(238 * a))
        elif kind == "mono":
            col = (int(198 * a), int(206 * a), int(212 * a))
        else:
            col = (int(222 * a), int(220 * a), int(212 * a))
        y = 1006 - (len(lines) - 1) * 54
        for ln in lines:
            shadow = (0, 0, 0, int(210 * a))
            d.text((W // 2 + 2, y + 2), ln, font=font, fill=shadow, anchor="mm")
            d.text((W // 2, y), ln, font=font, fill=col, anchor="mm")
            y += 54


# ------------------------------------------------------------------ image fx
IMAGES = {}


def loadImages():
    for s in SHOTS:
        if s["img"] and s["img"] not in IMAGES:
            p = os.path.join(FRAMES, s["img"] + ".png")
            im = Image.open(p).convert("RGB")
            IMAGES[s["img"]] = im
            print("  loaded", s["img"], im.size)


def buildVignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx / W - 0.5) * 2
    ny = (yy / H - 0.5) * 2
    r = np.sqrt(nx ** 2 + ny ** 2) / 1.414
    m = np.clip((r - 0.42) / 0.78, 0, 1) ** 1.35
    return m.astype(np.float32)


def buildHaze():
    small = (rng.random((24, 42, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(small).resize((W, H), Image.BICUBIC)
    im = im.filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(50))
    return (np.asarray(im).astype(np.float32) / 255 - 0.5).astype(np.float32)


def buildStars():
    im = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    rr = np.random.default_rng(5150)
    for _ in range(2400):
        x, y = int(rr.integers(0, W)), int(rr.integers(0, H))
        rad = float(rr.choice([0.7, 0.9, 1.1, 1.5, 2.1]))
        b = float(rr.uniform(0.18, 1.0) ** 2.0)
        c = (int(255 * b * rr.uniform(0.82, 1.0)), int(255 * b * rr.uniform(0.88, 1.0)), int(255 * b))
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=c)
    for _ in range(26):
        x, y = int(rr.integers(0, W)), int(rr.integers(0, H))
        b = float(rr.uniform(0.5, 1.0))
        for k in range(7, 0, -1):
            v = int(210 * b * (0.13 ** (1 - k / 7.0)) * (k / 7.0))
            d.ellipse([x - k * 1.6, y - k * 1.6, x + k * 1.6, y + k * 1.6],
                      fill=(int(v * 0.92), int(v * 0.96), v))
    small = (rr.random((30, 54, 3)) * 255).astype(np.uint8)
    neb = Image.fromarray(small).resize((W, H), Image.BICUBIC).filter(
        __import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(70))
    arr = np.asarray(im).astype(np.float32) / 255 + np.asarray(neb).astype(np.float32) / 255 * 0.13
    return np.clip(arr, 0, 1).astype(np.float32)


STAR = None
VIG = None
HAZE = None
RDIST = None


def init():
    global STAR, VIG, HAZE, RDIST
    loadImages()
    VIG = buildVignette()
    HAZE = buildHaze()
    STAR = buildStars()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    RDIST = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2).astype(np.float32)


def cropFrame(im, z, cx, cy, shake=(0.0, 0.0)):
    sw, sh = im.size
    if sw / sh > W / H:
        fh, fw = sh, sh * W / H
    else:
        fw, fh = sw, sw * H / W
    cw, ch = min(fw / z, sw), min(fh / z, sh)
    ch = min(ch, cw * H / W)
    cw = ch * W / H
    x0 = cx * sw - cw / 2 + shake[0]
    y0 = cy * sh - ch / 2 + shake[1]
    x0 = max(0.0, min(sw - cw, x0))
    y0 = max(0.0, min(sh - ch, y0))
    box = (int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch)))
    return im.resize((W, H), Image.LANCZOS, box=box)


def cropArr(arr, z, cx, cy):
    ah, aw = arr.shape[:2]
    fh, fw = ah, ah * W / H
    if aw / ah > W / H:
        fh, fw = ah, ah * W / H
    ch = fh / z
    cw = ch * W / H
    x0 = max(0.0, min(aw - cw, cx * aw - cw / 2))
    y0 = max(0.0, min(ah - ch, cy * ah - ch / 2))
    box = (int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch)))
    sub = arr[box[1]:box[3], box[0]:box[2]]
    return np.asarray(Image.fromarray((sub * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                      np.float32) / 255.0


def grade(a, sat, con, exp, tint, lift=0.0):
    l = a[:, :, 0] * 0.2126 + a[:, :, 1] * 0.7152 + a[:, :, 2] * 0.0722
    a = l[..., None] + (a - l[..., None]) * sat
    a = (a - 0.5) * con + 0.5 + lift
    a *= exp
    a[..., 0] *= tint[0]
    a[..., 1] *= tint[1]
    a[..., 2] *= tint[2]
    np.clip(a, 0, 1, out=a)
    return a


FLICK_SEED = rng.random(4096)


def flickerAt(t, level):
    i = int(t * 30) % 4096
    n = 0.5 * FLICK_SEED[i] + 0.5 * FLICK_SEED[(i * 7 + 13) % 4096]
    base = 1.0 + 0.10 * math.sin(2 * math.pi * 7.3 * t) + 0.06 * (n - 0.5) * 2
    drops = (44.2, 51.7, 60.4, 63.1, 68.9, 74.1) if level == 1 else (57.6, 62.2, 66.8, 70.4)
    for dt in drops:
        d = t - dt
        if 0 <= d < 0.26:
            base *= 0.22 + 0.78 * smooth(d / 0.26)
        elif -0.05 < d < 0:
            base *= 0.30
    return max(0.15, base)


def shakeAt(sh, t):
    sx = sy = 0.0
    for (te, amp) in sh["shake"]:
        d = t - te
        if 0 <= d < 1.1:
            e = math.exp(-d * 3.4)
            sx += amp * e * math.sin(2 * math.pi * 8.5 * d)
            sy += amp * e * math.cos(2 * math.pi * 6.7 * d) * 0.7
    return sx, sy


def flashAt(sh, t):
    v = np.zeros(3, np.float32)
    for (te, amp, dec) in sh["flash"]:
        d = t - te
        if 0 <= d < dec * 3:
            e = math.exp(-d / dec)
            v += np.array([0.72, 0.84, 1.0], np.float32) * (amp * e)
    return v


def renderShot(sh, t):
    u = (t - sh["ts"]) / (sh["te"] - sh["ts"])
    z = lerp(sh["z"][0], sh["z"][1], u)
    cx = lerp(sh["c"][0][0], sh["c"][1][0], u)
    cy = lerp(sh["c"][0][1], sh["c"][1][1], u)
    if sh["kind"] == "star":
        a = cropArr(STAR, z, cx, cy)
        if sh["mode"] == "reveal":
            p = smooth((t - (sh["ts"] + 1.15)) / 2.5)
            rmax = float(np.sqrt((W / 2) ** 2 + (H / 2) ** 2))
            R = (0.01 + 1.25 * p) * rmax
            mask = np.clip((R - RDIST) / (0.20 * rmax), 0, 1).astype(np.float32)
            a *= mask[..., None]
            glow = np.exp(-(RDIST ** 2) / (2 * (24 + 300 * p) ** 2)).astype(np.float32)
            a += (glow * (1 - p) * 1.5)[..., None] * np.array([0.86, 0.93, 1.0], np.float32)
        else:
            tw = 1.0 + 0.05 * math.sin(2 * math.pi * 0.43 * t) + 0.03 * math.sin(2 * math.pi * 1.7 * t)
            a *= tw
    else:
        im = IMAGES[sh["img"]]
        sx, sy = shakeAt(sh, t)
        pim = cropFrame(im, z, cx, cy, (sx, sy))
        a = np.asarray(pim, np.float32) / 255.0
    a = grade(a, sh["sat"], sh["con"], sh["exp"], sh["tint"], sh["lift"])
    if sh["flicker"]:
        a *= flickerAt(t, sh["flicker"])
    if sh["dark"] != 1.0:
        a *= sh["dark"]
    return a


def alphaOf(sh, t):
    a = 1.0
    if sh["fi"] > 0:
        a = min(a, smooth((t - sh["ts"]) / sh["fi"]))
    if sh["fo"] > 0:
        a = min(a, smooth((sh["te"] - t) / sh["fo"]))
    return max(0.0, min(1.0, a))


def makeFrame(i):
    t = i / FPS
    out = np.zeros((H, W, 3), np.float32)
    flash = np.zeros(3, np.float32)
    best = None
    bestA = -1
    for sh in SHOTS:
        if sh["ts"] <= t < sh["te"]:
            a = alphaOf(sh, t)
            if a <= 0.002:
                continue
            if a > bestA:
                bestA, best = a, sh
            img = renderShot(sh, t)
            out += img * a
            flash += flashAt(sh, t) * a
    if best is None:
        return np.zeros((H, W, 3), np.uint8)
    if flash.max() > 0:
        out += flash
    l = out[:, :, 0] * 0.2126 + out[:, :, 1] * 0.7152 + out[:, :, 2] * 0.0722
    out *= (1.0 - best["vig"] * VIG)[..., None]
    out += HAZE * 0.035
    g = rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8)
    g = np.asarray(Image.fromarray(g, "L").resize((W, H), Image.NEAREST), np.float32) / 255.0 - 0.5
    out += g[..., None] * best["grain"] * (1.15 - 0.55 * l)[..., None]
    if t > TOTAL - 0.8:
        out *= max(0.0, (TOTAL - t) / 0.8)
    np.clip(out, 0, 1, out=out)
    im = Image.fromarray((out * 255).astype(np.uint8))
    drawTexts(im, t)
    return im.tobytes()


def stats():
    init()
    print("frame stats (mean luma per shot middle):")
    for k, sh in enumerate(SHOTS):
        t = (sh["ts"] + sh["te"]) / 2
        i = int(t * FPS)
        b = np.frombuffer(makeFrame(i), np.uint8).reshape(H, W, 3).astype(np.float32) / 255
        l = (b[:, :, 0] * 0.2126 + b[:, :, 1] * 0.7152 + b[:, :, 2] * 0.0722)
        print("  %2d %-22s t=%6.2f  luma=%.3f  max=%.3f" % (k, sh["img"] or sh["mode"], t, l.mean(), b.max()))
    for t in (0.5, 76.0, 76.8, 78.5, 81.0, 290.0, 299.5):
        b = np.frombuffer(makeFrame(int(t * FPS)), np.uint8).reshape(H, W, 3).astype(np.float32) / 255
        l = (b[:, :, 0] * 0.2126 + b[:, :, 1] * 0.7152 + b[:, :, 2] * 0.0722)
        print("  t=%6.2f luma=%.4f max=%.3f" % (t, l.mean(), b.max()))


def main():
    init()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
           "-i", "-", "-i", AUDIO,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT]
    print(" ".join(cmd))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.time()
    try:
        for i in range(NF):
            p.stdin.write(makeFrame(i))
            if i % 150 == 0:
                el = time.time() - t0
                eta = el / max(1, i + 1) * (NF - i - 1)
                print("frame %5d/%d  %.1fs elapsed, eta %.0fs" % (i, NF, el, eta), flush=True)
    finally:
        p.stdin.close()
        err = p.stderr.read().decode(errors="ignore")
        p.wait()
    print("ffmpeg rc", p.returncode)
    if p.returncode != 0:
        print(err[-3000:])
    else:
        print("done in %.1fs ->" % (time.time() - t0), OUT, os.path.getsize(OUT) / 1e6, "MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    else:
        main()
