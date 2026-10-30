"""Merge font subsets (cyrillic + latin + latin-ext) into full-coverage TTFs."""
import os

from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen

BASE = "/home/user/fonts"
DST = "/home/user/Tqdam/video/fonts"

JOBS = [
    ("Oswald-Bold.ttf", "fontsource-oswald-5.3.0/package/files/oswald-%s-700-normal.woff2",
     ["cyrillic", "latin", "latin-ext"]),
    ("Oswald-SemiBold.ttf", "fontsource-oswald-5.3.0/package/files/oswald-%s-600-normal.woff2",
     ["cyrillic", "latin", "latin-ext"]),
    ("Rubik-Black.ttf", "rubik/package/files/rubik-%s-900-normal.woff2",
     ["cyrillic", "latin", "latin-ext"]),
    ("Rubik-Bold.ttf", "rubik/package/files/rubik-%s-700-normal.woff2",
     ["cyrillic", "latin", "latin-ext"]),
    ("Rubik-Regular.ttf", "rubik/package/files/rubik-%s-400-normal.woff2",
     ["cyrillic", "latin", "latin-ext"]),
]


def open_ttf(path):
    f = TTFont(path)
    f.flavor = None
    return f


def merge(base, extra, tag):
    base_cmap = base.getBestCmap()
    extra_cmap = extra.getBestCmap()
    extra_glyphs = extra.getGlyphSet()
    glyf = base["glyf"]
    hmtx = base["hmtx"]
    order = base.getGlyphOrder()

    added = 0
    for cp, name in sorted(extra_cmap.items()):
        if cp in base_cmap:
            continue
        new_name = f"{tag}.{name}"
        if new_name in base_cmap.values():
            continue
        pen = TTGlyphPen(None)
        try:
            extra_glyphs[name].draw(pen)
            glyph = pen.glyph()
        except Exception:
            continue
        glyf.glyphs[new_name] = glyph
        order.append(new_name)
        adv, lsb = extra["hmtx"][name]
        hmtx.metrics[new_name] = (adv, lsb)
        for table in base["cmap"].tables:
            if table.isUnicode():
                table.cmap[cp] = new_name
        added += 1
    base.setGlyphOrder(order)
    return added


def main():
    for out_name, pat, subsets in JOBS:
        fonts = []
        for s in subsets:
            p = os.path.join(BASE, pat % s)
            if os.path.exists(p):
                fonts.append((s, open_ttf(p)))
        base = fonts[0][1]
        total = 0
        for tag, extra in fonts[1:]:
            total += merge(base, extra, tag)
        base["maxp"].numGlyphs = len(base.getGlyphOrder())
        path = os.path.join(DST, out_name)
        base.save(path)
        # coverage report
        chk = TTFont(path)
        cmap = chk.getBestCmap()
        need = ("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
                "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                "0123456789:.,!?%()—–-+·→✓/ ")
        missing = [c for c in need if ord(c) not in cmap]
        print(f"{out_name}: +{total} glyphs, missing: {''.join(missing) or 'none'}")


if __name__ == "__main__":
    main()
