"""
Renderer for «POV: минус сорок тысяч» — 1080x1920, 30 fps, 360 s.

Dark cinematic comic look: slow push on illustrated plates, film grain, deep
vignette, POV captions timed from the narration text, chapter chips, editorial
beats, graphic intermissions and a checklist outro.

usage:
  python3 render_money.py --from 0 --to 3600 --out out/seg0.mp4
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

from script_money import BEATS, TEXTS

ROOT = os.path.dirname(os.path.abspath(__file__))
KF = os.path.join(ROOT, "kf")
FONTS = os.path.join(os.path.dirname(ROOT), "fonts")
OUT = os.path.join(ROOT, "out")
FF = imageio_ffmpeg.get_ffmpeg_exe()

with open(os.path.join(OUT, "schedule.json")) as f:
    SCHED = json.load(f)

W, H = SCHED["size"]
FPS = SCHED["fps"]
TOTAL = SCHED["total"]
NF = int(TOTAL * FPS)

PAPER = (240, 236, 228)
ORANGE = (255, 107, 53)
TEAL = (74, 178, 186)
INK = (12, 12, 20)

FONTS_MAP = {
    "cond": "Oswald-Bold.ttf",
    "cond6": "Oswald-SemiBold.ttf",
    "black": "Rubik-Black.ttf",
    "bold": "Rubik-Bold.ttf",
    "reg": "Rubik-Regular.ttf",
}
_fc = {}


def font(key, size):
    k = (key, size)
    if k not in _fc:
        from PIL import ImageFont
        _fc[k] = ImageFont.truetype(os.path.join(FONTS, FONTS_MAP[key]), size)
    return _fc[k]


def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_in_out(p):
    return p * p * (3 - 2 * p)


def ease_back(p, s=1.9):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


# ------------------------------------------------------------------ captions ---
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
COMMA_SPLIT = re.compile(r"(?<=[,;:])\s+")
MAX_CHARS = 108


def chunk_text(text):
    """Split narration into caption-sized chunks (<= MAX_CHARS), dropping nothing."""
    out = []
    for sent in SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= MAX_CHARS:
            out.append(sent)
            continue
        cur = ""
        for part in COMMA_SPLIT.split(sent):
            trial = (cur + " " + part).strip()
            if len(trial) <= MAX_CHARS or not cur:
                cur = trial
            else:
                out.append(cur)
                cur = part
        if cur:
            out.append(cur)
    # merge tiny tails into the previous chunk
    merged = []
    for c in out:
        if merged and len(c) < 28:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return merged


def build_captions():
    out = []
    for ch in SCHED["chapters"]:
        if ch.get("kind"):
            continue
        text = TEXTS.get(ch["id"], "")
        if not text:
            continue
        chunks = chunk_text(text)
        total_chars = sum(len(c) for c in chunks) or 1
        span = ch["vo_end"] - ch["vo_start"]
        t = ch["vo_start"]
        for c in chunks:
            d = span * len(c) / total_chars
            out.append(dict(text=c, start=t, end=t + d, chapter=ch["id"]))
            t += d
    return out


CAPTIONS = build_captions()


def find_chapter(t):
    for c in SCHED["chapters"]:
        if c["start"] <= t < c["end"]:
            return c
    return SCHED["chapters"][-1]


# --------------------------------------------------------------- base plates ---
SCALE = 1.28          # pre-upscale so the slow push stays sharp


def load_plate(name):
    p = os.path.join(KF, name + ".png")
    im = Image.open(p).convert("RGB")
    im = im.resize((int(W * SCALE), int(H * SCALE)), Image.LANCZOS)
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=45, threshold=3))


class Renderer:
    def __init__(self):
        self.plates = {}
        self.cache = {}
        self.blurred = {}
        self.grain = []
        rng = np.random.RandomState(7)
        for _ in range(10):
            n = rng.normal(126, 6.5, (H, W)).clip(0, 255).astype(np.uint8)
            self.grain.append(Image.fromarray(n, "L"))

        # vignette + subtle top/bottom falloff, baked into one grayscale mask
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W / 2, H / 2
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / 1.42
        v = 1.0 - 0.62 * np.clip(r, 0, 1) ** 1.9
        band = np.ones(H, dtype=np.float32)
        band[: int(H * 0.10)] = np.linspace(0.72, 1.0, int(H * 0.10))
        band[-int(H * 0.14):] = np.linspace(1.0, 0.62, int(H * 0.14))
        v = v * band[:, None]
        self.vign = Image.fromarray((v * 255).astype(np.uint8), "L")
        self.vign_rgb = Image.merge("RGB", (self.vign, self.vign, self.vign))
        soft = np.clip(0.55 + 0.45 * v, 0, 1)
        self.vign_card = Image.merge("RGB", [Image.fromarray((soft * 255).astype(np.uint8), "L")] * 3)

        self.cool = Image.new("RGB", (W, H), (196, 214, 232))
        self.warm = Image.new("RGB", (W, H), (255, 226, 198))
        self.tints = {"cool": self.cool, "warm": self.warm}

    # -- assets ---------------------------------------------------------------
    def plate(self, name):
        if name not in self.plates:
            self.plates[name] = load_plate(name)
        return self.plates[name]

    def soft(self, name):
        if name not in self.blurred:
            p = self.plate(name).resize((W, H), Image.LANCZOS)
            p = p.filter(ImageFilter.GaussianBlur(26))
            arr = np.asarray(p).astype(np.float32) * 0.62 + 26
            self.blurred[name] = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return self.blurred[name]

    # -- camera ---------------------------------------------------------------
    def camera(self, shot_img, p, t, warm=False):
        base = self.plate(shot_img)
        bw, bh = base.size
        z = 1.03 + 0.12 * ease_in_out(p)
        # handheld drift
        dx = math.sin(t * 0.37) * 6 + math.sin(t * 1.13) * 2
        dy = math.cos(t * 0.29) * 5 + math.sin(t * 0.91) * 2
        cw, ch = bw / z, bh / z
        x0 = (bw - cw) / 2 + dx
        y0 = (bh - ch) / 2 + dy
        x0 = min(max(x0, 0), bw - cw)
        y0 = min(max(y0, 0), bh - ch)
        crop = base.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
        img = crop.resize((W, H), Image.BILINEAR)
        img = ImageChops.multiply(img, self.tints["warm" if warm else "cool"])
        img = ImageChops.multiply(img, self.vign_rgb)
        return img

    # -- primitives -----------------------------------------------------------
    @staticmethod
    def wrap(draw, text, fnt, maxw):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=fnt) <= maxw or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def chip(self, text, age, fade_out=0.0):
        """Top-left chapter marker with a hairline rule."""
        a = clamp01(age / 0.35)
        if fade_out:
            a *= clamp01(fade_out)
        if a <= 0.01:
            return None, (0, 0)
        f = font("cond", 46)
        probe = Image.new("RGBA", (8, 8))
        tw = ImageDraw.Draw(probe).textlength(text, font=f)
        while tw > 830 and f.size > 30:
            f = font("cond", f.size - 2)
            tw = ImageDraw.Draw(probe).textlength(text, font=f)
        w = int(tw + 130)
        lay = Image.new("RGBA", (w, 120), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        fa = int(255 * a)
        d.rectangle([0, 96, 90, 100], fill=ORANGE + (fa,))
        d.rectangle([98, 97, min(w, 98 + tw + 10), 99], fill=(255, 255, 255, int(fa * 0.42)))
        d.text((0, 22), text, font=f, fill=(255, 255, 255, fa))
        return lay, (78, 196)

    def caption(self, text, alpha):
        if alpha <= 0.02:
            return None, (0, 0)
        f = font("bold", 56)
        probe = Image.new("RGBA", (8, 8))
        pd = ImageDraw.Draw(probe)
        lines = self.wrap(pd, text, f, 880)
        lh = 70
        wid = max(pd.textlength(l, font=f) for l in lines)
        lay = Image.new("RGBA", (int(wid + 64), int(lh * len(lines) + 48)), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        a = int(240 * alpha)
        d.rounded_rectangle([0, 0, lay.width - 1, lay.height - 1], 18,
                            fill=INK + (int(150 * alpha),))
        for i, ln in enumerate(lines):
            d.text((32, 24 + i * lh), ln, font=f, fill=(255, 255, 255, a))
        return lay, (W / 2 - lay.width / 2, 1380 - lay.height)

    def accent_line(self, text, age, at=(90, 1330), size=96, color=ORANGE):
        p = ease_back(clamp01(age / 0.32))
        if p <= 0.02:
            return None, (0, 0)
        f = font("black", size)
        probe = Image.new("RGBA", (8, 8))
        pd = ImageDraw.Draw(probe)
        while size > 40 and pd.textlength(text, font=f) > 880:
            size -= 4
            f = font("black", size)
        tw = pd.textlength(text, font=f)
        lay = Image.new("RGBA", (int(tw + 40), int(size * 1.5)), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.text((20, lay.height * 0.5), text, font=f, fill=color + (255,), anchor="lm")
        s = 0.9 + 0.1 * p
        lay = lay.resize((int(lay.width * s), int(lay.height * s)), Image.BICUBIC)
        return lay, (at[0] - 20, at[1] - lay.height / 2)

    # -- scenes ---------------------------------------------------------------
    def photo_scene(self, ch, t, idx):
        span = ch["end"] - ch["start"]
        p = clamp01((t - ch["start"]) / span)
        warm = ch["id"] >= 9
        img = self.camera(ch["img"], p, t, warm=warm)
        img = ImageChops.add(img, Image.merge("RGB", [self.grain[idx % 10]] * 3), 1.0, -126)

        parts = []
        age = t - ch["start"]
        fade = (ch["end"] - t) / 0.4
        pieces = [self.chip(f'{ch["kicker"]}  ·  {ch["headline"]}', age, fade)]

        # editorial beat in the middle of the chapter
        beats = BEATS.get(ch["id"], [])
        if beats:
            bo = (ch["vo_end"] - ch["vo_start"]) * 0.58
            ba = age - (0.75 + bo)
            if 0 <= ba <= 4.2:
                for k, (txt, kind) in enumerate(beats):
                    bage = ba - k * 1.5
                    if 0 <= bage <= 4.2:
                        size = 128 if kind == "number" else 84
                        pieces.append(self.accent_line(txt, bage, at=(84, 1010), size=min(size, 84)))
        # POV captions
        for cap in CAPTIONS:
            if cap["start"] <= t <= cap["end"] + 0.35 and cap["chapter"] == ch["id"]:
                a = clamp01((t - cap["start"]) / 0.12) * clamp01((cap["end"] + 0.35 - t) / 0.2)
                pieces.append(self.caption(cap["text"], a))
                break
        img = self.composite(img, pieces)
        return img

    def headline(self, ch, age, fade):
        p = ease_back(clamp01(age / 0.4))
        if p <= 0.02:
            return None, (0, 0)
        a = int(255 * clamp01(age / 0.25) * ((fade if fade < 1 else 1)))
        f = font("black", 82)
        probe = Image.new("RGBA", (8, 8))
        pd = ImageDraw.Draw(probe)
        lines = self.wrap(pd, ch["headline"], f, 860)[:2]
        sub = ch.get("subline", "")
        lay_h = int(100 * len(lines) + (70 if sub else 0) + 40)
        lay = Image.new("RGBA", (940, lay_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for i, ln in enumerate(lines):
            d.text((10, 10 + i * 98), ln, font=f, fill=(255, 255, 255, a))
        if sub:
            d.text((14, 24 + len(lines) * 98), sub, font=font("cond", 46),
                   fill=ORANGE + (a,))
        s = 0.94 + 0.06 * p
        lay = lay.resize((int(lay.width * s), int(lay.height * s)), Image.BICUBIC)
        return lay, (80, 330)

    def paper_bg(self, name):
        """Cream notebook page with a photo strip — the graphic intermission look."""
        key = "paper_" + name
        if key in self.cache:
            return self.cache[key]
        page = Image.new("RGB", (W, H), (238, 232, 220))
        d = ImageDraw.Draw(page)
        # ruled lines
        for y in range(140, H, 74):
            d.line([(0, y), (W, y)], fill=(224, 216, 202), width=2)
        # margin
        d.line([(96, 0), (96, H)], fill=(226, 150, 130), width=3)
        # photo strip at the bottom: the blurred plate, cropped
        strip = self.plate(name).resize((int(W * 0.86), int(H * 0.26)), Image.LANCZOS)
        strip = strip.convert("RGB")
        sa = np.asarray(strip).astype(np.float32)
        lum = sa.mean(axis=2, keepdims=True) / 255.0
        sa = sa * (0.55 + 0.75 * lum)
        sa = sa * np.array([1.06, 1.0, 0.94])[None, None, :]
        strip = Image.fromarray(np.clip(sa, 0, 255).astype(np.uint8))
        sw, sh = strip.size
        px, py = int(W * 0.07), int(H * 0.665)
        frame = Image.new("RGB", (sw + 36, sh + 36), (250, 248, 244))
        frame.paste(strip, (18, 18))
        page.paste(frame, (px, py))
        d.rectangle([px, py, px + sw + 35, py + sh + 35], outline=(26, 24, 22), width=4)
        self.cache[key] = page
        return page

    def card_scene(self, ch, t, idx):
        nxt = next((c for c in SCHED["chapters"]
                    if not c.get("kind") and c["start"] >= ch["end"]), None)
        bg_img = nxt["img"] if nxt else "m10"
        img = self.paper_bg(bg_img).convert("RGB")
        img = ImageChops.add(img, Image.merge("RGB", [self.grain[idx % 10]] * 3), 1.0, -134)
        parts = []
        y = 430
        for b in ch["beats"]:
            age = t - b["start"]
            if age < 0:
                continue
            vis = clamp01((b["end"] - t) / 0.3)
            a = int(255 * clamp01(age / 0.3) * vis)
            if b["style"] == "kicker":
                lay = Image.new("RGBA", (960, 90), (0, 0, 0, 0))
                d = ImageDraw.Draw(lay)
                d.rectangle([0, 58, 84, 64], fill=ORANGE + (a,))
                d.text((100, 20), b["text"], font=font("cond", 48), fill=(24, 22, 20, a))
                parts.append((lay, (120, 250)))
            elif b["style"] == "number":
                s_ = 1.0 + 0.06 * (1 - ease_out(clamp01(age / 0.5)))
                f = font("black", 150)
                probe = Image.new("RGBA", (8, 8))
                tw = ImageDraw.Draw(probe).textlength(b["text"], font=f)
                lay = Image.new("RGBA", (int(tw + 60), 230), (0, 0, 0, 0))
                d = ImageDraw.Draw(lay)
                d.text((30, 115), b["text"], font=f, fill=(20, 18, 16, a), anchor="lm")
                lay = lay.resize((int(lay.width * s_), int(lay.height * s_)), Image.BICUBIC)
                parts.append((lay, (112, y + 20)))
                y += 280
            else:
                f = font("black", 76)
                probe = Image.new("RGBA", (8, 8))
                pd = ImageDraw.Draw(probe)
                while f.size > 40 and pd.textlength(b["text"], font=f) > 830:
                    f = font("black", f.size - 4)
                lay = Image.new("RGBA", (900, 108), (0, 0, 0, 0))
                d = ImageDraw.Draw(lay)
                dy = (1 - ease_out(clamp01(age / 0.4))) * 24
                d.text((12, 54 + dy), b["text"], font=f, fill=(24, 22, 20, a), anchor="lm")
                parts.append((lay, (118, y)))
                y += 112
        # progress dots
        cards = [c for c in SCHED["chapters"] if c.get("kind") not in (None, "outro")]
        lay = Image.new("RGBA", (W, 80), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        gap = 34
        x0 = W / 2 - (len(cards) - 1) * gap / 2
        for i, c in enumerate(cards):
            on = c["start"] <= t < c["end"]
            r = 8 if on else 4
            col = ORANGE if on else (40, 36, 32)
            a = 255 if on else 120
            d.ellipse([x0 + i * gap - r, 20 - r, x0 + i * gap + r, 20 + r], fill=col + (a,))
        parts.append((lay, (0, 1560)))
        return self.composite(img, parts)

    def outro_scene(self, ch, t, idx):
        img = self.soft("m10")
        img = ImageChops.multiply(img, self.vign_card)
        ramp = np.zeros((H, W, 4), dtype=np.uint8)
        grad = np.clip(np.sin(np.linspace(0, math.pi, H)) ** 0.55, 0, 1) * 215
        ramp[:, :, 3] = grad[:, None].astype(np.uint8)
        scrim = Image.fromarray(ramp, "RGBA").filter(ImageFilter.GaussianBlur(30))
        img = Image.alpha_composite(img.convert("RGBA"), scrim).convert("RGB")
        img = ImageChops.add(img, Image.merge("RGB", [self.grain[idx % 10]] * 3), 1.0, -126)
        parts = []
        y = 470
        for b in ch["beats"]:
            age = t - b["start"]
            if age < 0:
                continue
            a = int(255 * clamp01(age / 0.3))
            if b["style"] == "kicker":
                lay = Image.new("RGBA", (960, 90), (0, 0, 0, 0))
                d = ImageDraw.Draw(lay)
                d.rectangle([0, 62, 92, 68], fill=ORANGE + (a,))
                d.text((108, 24), b["text"], font=font("cond", 46), fill=(255, 255, 255, a))
                parts.append((lay, (70, 250)))
            else:
                f = font("bold", 58)
                lay = Image.new("RGBA", (960, 92), (0, 0, 0, 0))
                d = ImageDraw.Draw(lay)
                dy = (1 - ease_out(clamp01(age / 0.35))) * 18
                d.text((70, 46 + dy), b["text"], font=f, fill=(255, 255, 255, a), anchor="lm")
                d.rectangle([20, 22 + dy, 32, 70 + dy], fill=ORANGE + (a,))
                parts.append((lay, (60, y)))
                y += 96
        # subscribe button
        age = t - (ch["start"] + 11.0)
        if age > 0:
            p = ease_back(clamp01(age / 0.4))
            a = int(255 * clamp01(age / 0.2))
            lay = Image.new("RGBA", (900, 210), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
            d.rounded_rectangle([10, 30, 690, 180], 24, fill=ORANGE + (a,),
                                outline=(255, 255, 255, a), width=5)
            d.text((350, 105), "ПОДПИСАТЬСЯ", font=font("black", 62),
                   fill=(255, 255, 255, a), anchor="mm")
            d.rounded_rectangle([740, 50, 860, 160], 20, fill=(255, 255, 255, a),
                                outline=INK + (a,), width=5)
            for i in range(3):
                d.ellipse([762 + i * 30, 96, 780 + i * 30, 114], fill=INK + (a,))
            s = 0.86 + 0.14 * p
            lay = lay.resize((int(lay.width * s), int(lay.height * s)), Image.BICUBIC)
            parts.append((lay, (W / 2 - lay.width / 2, 1330)))
        img = self.composite(img, parts)
        return img

    @staticmethod
    def composite(img, pieces):
        if not pieces:
            return img
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for piece, pos in pieces:
            if piece is None:
                continue
            layer.alpha_composite(piece, (int(pos[0]), int(pos[1])))
        out = img.convert("RGBA")
        out.alpha_composite(layer)
        return out.convert("RGB")

    # -- frame ----------------------------------------------------------------
    def frame(self, idx):
        t = idx / FPS
        ch = find_chapter(t)
        if ch.get("kind") == "card":
            img = self.card_scene(ch, t, idx)
        elif ch.get("kind") == "outro":
            img = self.outro_scene(ch, t, idx)
        else:
            img = self.photo_scene(ch, t, idx)
        if idx < 12:  # fade in from black
            k = idx / 12
            img = Image.blend(Image.new("RGB", (W, H), INK), img, k)
        return img


# --------------------------------------------------------------------- render ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="stop", type=int, default=NF)
    ap.add_argument("--out", default=os.path.join(OUT, "seg.mp4"))
    ap.add_argument("--step", type=int, default=1, help="write every Nth frame (stills only)")
    a = ap.parse_args()

    r = Renderer()
    if a.step > 1:
        os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
        for i in range(a.start, a.stop, a.step):
            r.frame(i).save(os.path.join(OUT, "stills", f"f{i:05d}.png"))
        print("stills written")
        return

    cmd = [FF, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-g", str(FPS * 2), a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(a.start, a.stop):
        proc.stdin.write(r.frame(i).tobytes())
        if (i - a.start) % 300 == 0:
            print(f"  frame {i}/{a.stop}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
