"""Sound design + voice mix for the Shadow Slave animatic (300 s).

Everything is synthesized procedurally with numpy: wind, room tone, chains,
impacts, bell, monster, god drone. No samples, no external assets.
"""
import math
import os
import wave

import numpy as np

SR = 44100
DUR = 300.0
N = int(SR * DUR)
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(os.path.dirname(HERE), "audio")
rng = np.random.default_rng(20241030)

mixL = np.zeros(N, np.float32)
mixR = np.zeros(N, np.float32)


# ---------------------------------------------------------------- helpers
def norm(y):
    y = np.asarray(y, np.float32)
    m = np.max(np.abs(y)) + 1e-9
    return (y / m).astype(np.float32)


def shaped(n, mag):
    x = rng.standard_normal(int(n)).astype(np.float32)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X = np.fft.rfft(x) * mag(f).astype(np.float32)
    y = np.fft.irfft(X, len(x)).astype(np.float32)
    return norm(y)


def envAR(n, a=0.05, r=0.2):
    n = int(n)
    e = np.ones(n, np.float32)
    na, nr = int(a * SR), int(r * SR)
    if na > 0:
        e[:na] = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, na))
    if nr > 0:
        e[-nr:] = e[-nr:] * (0.5 + 0.5 * np.cos(np.linspace(0, math.pi, nr)))
    return e


def slowEnv(n, lo=0.35, hi=1.0, step=1.4):
    k = max(2, int(n / (step * SR)) + 2)
    pts = rng.uniform(lo, hi, k).astype(np.float32)
    t = np.linspace(0, k - 1, n)
    return np.interp(t, np.arange(k), pts).astype(np.float32)


def place(start, sig, amp=1.0, pan=0.0):
    """pan: -1 left .. +1 right"""
    sig = np.asarray(sig, np.float32)
    i = int(start * SR)
    if i < 0:
        sig = sig[-i:]
        i = 0
    n = min(len(sig), N - i)
    if n <= 0:
        return
    sig = sig[:n]
    gl = math.cos((pan + 1) * math.pi / 4)
    gr = math.sin((pan + 1) * math.pi / 4)
    mixL[i:i + n] += sig * amp * gl * 1.414
    mixR[i:i + n] += sig * amp * gr * 1.414


def placeStereo(start, sigL, sigR, amp=1.0):
    i = int(start * SR)
    if i < 0:
        sigL = sigL[-i:]
        sigR = sigR[-i:]
        i = 0
    n = min(len(sigL), len(sigR), N - i)
    if n <= 0:
        return
    mixL[i:i + n] += sigL[:n] * amp
    mixR[i:i + n] += sigR[:n] * amp


def bed(start, end, buf, amp, a=2.0, r=2.0, gust=(0.45, 1.0)):
    """Loop a noise buffer between start and end, decorrelated L/R."""
    dur = end - start
    nl = int((dur + a + r) * SR)
    outL = np.empty(nl, np.float32)
    outR = np.empty(nl, np.float32)
    for out in (outL, outR):
        i = 0
        while i < nl:
            off = int(rng.integers(0, len(buf)))
            m = min(nl - i, len(buf) - off)
            out[i:i + m] = buf[off:off + m]
            i += m
    e = envAR(nl, a, r)
    g = slowEnv(nl, *gust)
    placeStereo(start - a, outL * e * g, outR * e * g, amp)


def conv(x, ir):
    n = len(x) + len(ir) - 1
    nf = 1 << int(np.ceil(math.log2(n)))
    X = np.fft.rfft(x, nf)
    H = np.fft.rfft(ir, nf)
    y = np.fft.irfft(X * H, nf)[: len(x)]
    return norm(y) * (np.sqrt(np.mean(x ** 2) + 1e-12) / (np.sqrt(np.mean(y ** 2)) + 1e-12))


def ir(dur=3.0, decay=4.0):
    n = int(SR * dur)
    t = np.linspace(0, decay, n)
    y = rng.standard_normal(n) * np.exp(-t)
    y[: int(SR * 0.012)] *= 0.15
    return y


CAVE = ir(3.4, 4.2)
ROOM = ir(0.9, 7.0)
HALL = ir(5.5, 2.6)
GOD = ir(3.6, 3.8)


def pitch(x, factor):
    n = len(x)
    t = np.arange(int(n * factor), dtype=np.float64) / factor
    return np.interp(t, np.arange(n), x).astype(np.float32)


# ---------------------------------------------------------------- one-shots
def clink(dur=0.45, f0=1850.0, decay=7.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float32)
    for m, a in ((1.0, 1.0), (1.53, 0.62), (2.11, 0.42), (3.07, 0.26), (4.71, 0.14)):
        y += a * np.sin(2 * np.pi * f0 * m * (1 + rng.uniform(-0.01, 0.01)) * t + rng.uniform(0, 6.28))
    y *= np.exp(-t * decay)
    tr = shaped(int(SR * 0.02), lambda f: np.clip((f / 6000.0) ** 0.4, 0, 1))
    k = len(tr)
    y[:k] += tr * np.exp(-np.arange(k) / SR * 140) * 0.7
    return norm(y)


def bellRing(dur=6.5, f0=1046.5):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float64)
    for m, a, d in ((1.0, 1.0, 1.7), (2.01, 0.5, 1.3), (2.42, 0.34, 1.05),
                    (3.03, 0.26, 0.85), (4.51, 0.16, 0.62), (5.36, 0.11, 0.5),
                    (6.79, 0.06, 0.4), (0.5, 0.35, 0.9)):
        y += a * np.sin(2 * np.pi * f0 * m * t) * np.exp(-t / d)
    y[:400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 60.0) * 0.5
    y = y * (1 + 0.06 * np.sin(2 * np.pi * 3.1 * t)) * np.exp(-t * 0.12)
    return norm(y)


def thud(dur=1.1, f0=62.0, bright=0.35, dec=7.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * (f0 + 45 * np.exp(-t * 20)) * t) * np.exp(-t * dec)
    nb = shaped(n, lambda f: np.exp(-f / 800.0)) * np.exp(-t * 14) * bright
    return norm(y + nb)


def rockImpact(dur=2.2, size=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    body = shaped(n, lambda f: np.exp(-f / (700.0 * size))) * np.exp(-t * 6.5)
    crack = shaped(n, lambda f: np.clip(f / 2500.0, 0, 1) ** 1.2 * np.exp(-f / 9000.0)) * np.exp(-t * 22) * 1.6
    sub = np.sin(2 * np.pi * (48 * size + 40 * np.exp(-t * 12)) * t) * np.exp(-t * 5) * 1.2
    tail = shaped(n, lambda f: np.exp(-f / 3500.0)) * np.exp(-t * 2.2) * 0.25
    return norm(body + crack + sub + tail)


def growl(dur=2.6, f0=52.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    am = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * 17.0 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.3 * t))
    src = shaped(n, lambda f: np.exp(-f / 420.0))
    y = src * am * np.exp(-t * 0.8)
    y += np.sin(2 * np.pi * f0 * t) * am * 0.8
    y += np.sin(2 * np.pi * f0 * 1.5 * t) * am * 0.3
    y *= envAR(n, 0.12, 0.9)
    return norm(y)


def wetTear(dur=1.6):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = shaped(n, lambda f: np.clip(f / 400.0, 0, 1) * np.exp(-f / 2600.0))
    rip = np.zeros(n)
    for s, w in ((0.02, 0.30), (0.42, 0.26), (0.74, 0.22)):
        i0 = int(s * n)
        i1 = min(n, i0 + int(w * n))
        if i1 > i0:
            rip[i0:i1] += np.exp(-np.linspace(0, 6, i1 - i0))
    y = y * np.clip(rip + 0.15, 0, 1.2)
    y += np.sin(2 * np.pi * (70 - 30 * t / dur) * t) * np.exp(-t * 3) * 0.5
    return norm(y)


def heartBeat(amp=1.0):
    a = thud(0.42, 58, 0.12, 9) * 0.9
    b = thud(0.36, 46, 0.1, 10) * 0.6
    n = int(SR * 0.95)
    y = np.zeros(n, np.float32)
    y[: len(a)] += a
    y[int(SR * 0.26):int(SR * 0.26) + len(b)] += b
    return norm(y) * amp


def step(dur=0.35, hard=1.0, pan=0.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = shaped(n, lambda f: np.exp(-f / (900.0 / hard))) * np.exp(-t * 16)
    y += shaped(n, lambda f: np.clip(f / 3000.0, 0, 1) ** 1.5) * np.exp(-t * 40) * 0.35 * hard
    return norm(y), dur, pan


def breath(dur=1.5, out=True):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = shaped(n, lambda f: np.clip(f / 250.0, 0, 1) * np.exp(-f / 1400.0))
    e = np.exp(-((t - (0.15 if out else dur * 0.6)) / (dur * 0.42)) ** 2)
    return norm(y * e)


def pad(dur, f0=55.0, amp=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float64)
    for m, a in ((1, 1.0), (1.5, 0.5), (2, 0.35), (3, 0.18), (4.02, 0.1)):
        y += a * np.sin(2 * np.pi * f0 * m * t + m)
    for m, a in ((1180, 0.035), (1770, 0.022), (2360, 0.012)):
        y += a * np.sin(2 * np.pi * m * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.21 * t + m))
    y *= envAR(n, dur * 0.35, dur * 0.4)
    return norm(y) * amp


def hum(dur):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 100 * t) * 0.6 + np.sin(2 * np.pi * 150 * t) * 0.25
         + np.sin(2 * np.pi * 300 * t) * 0.12 + np.sin(2 * np.pi * 1200 * t) * 0.04)
    y *= 0.5 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    return norm(y)


def shimmer(dur, f0=1400.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float64)
    for m in (1.0, 1.5, 2.0, 2.51, 3.35, 4.2):
        y += (1 / m) * np.sin(2 * np.pi * f0 * m * t + m * 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * m * t))
    y *= envAR(n, dur * 0.4, dur * 0.45)
    return norm(y)


def drip():
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * (1400 - 900 * t / 0.5) * t) * np.exp(-t * 22)
    y += np.sin(2 * np.pi * 2600 * t) * np.exp(-t * 35) * 0.4
    return norm(y)


def metalShing():
    n = int(SR * 1.0)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for m, a in ((1.0, 1.0), (1.62, 0.5), (2.4, 0.3), (3.9, 0.15)):
        y += a * np.sin(2 * np.pi * 2400 * m * t) * np.exp(-t * (5 + m))
    return norm(y)


# ---------------------------------------------------------------- buffers
print("generating noise buffers ...")
BUF = 45.0
windLow = shaped(int(SR * BUF), lambda f: 1.0 / (1.0 + (f / 140.0) ** 1.7))
windMid = shaped(int(SR * BUF), lambda f: np.clip(f / 260.0, 0, 1) ** 0.6 * np.exp(-f / 2600.0))
windHigh = shaped(int(SR * BUF), lambda f: np.clip(f / 1800.0, 0, 1) ** 1.2 * np.exp(-f / 9000.0))
roomBuf = shaped(int(SR * BUF), lambda f: 1.0 / (1.0 + (f / 220.0) ** 2.0))
cityBuf = shaped(int(SR * BUF), lambda f: 1.0 / (1.0 + (f / 320.0) ** 1.4))
hissBuf = shaped(int(SR * BUF), lambda f: np.clip(f / 900.0, 0, 1) * np.exp(-f / 7000.0))
print("buffers done")

# ---------------------------------------------------------------- SCENE 1
bed(0.0, 40.0, cityBuf, 0.055, 3.0, 3.0, (0.5, 1.0))
bed(0.0, 40.0, windLow, 0.05, 3.0, 3.0, (0.35, 1.0))
bed(0.0, 40.0, windMid, 0.012, 3.0, 3.0)
place(4.3, clink(0.3, 2600, 12), 0.12, 0.15)          # cup on bench
place(9.2, breath(1.2), 0.05, -0.1)
place(16.6, breath(1.1), 0.05, 0.1)
for i, tt in enumerate((21.0, 24.0, 27.0)):           # distant cars
    place(tt, conv(cityBuf[: int(SR * 4)] * envAR(int(SR * 4), 1.5, 2.0), ROOM), 0.05, 0.5 - i * 0.5)
for tt in (30.5, 32.2, 33.9, 35.6, 37.3):             # sunny walks
    s, d, p = step(0.35, 0.7)
    place(tt, s, 0.16, -0.1)
place(30.0, heartBeat(), 0.10)
place(33.0, heartBeat(), 0.11)
place(36.2, heartBeat(), 0.09)

# ---------------------------------------------------------------- SCENE 2
bed(38.8, 76.5, roomBuf, 0.075, 2.5, 2.5, (0.6, 1.0))
bed(38.8, 76.5, hissBuf, 0.010, 2.5, 2.5)
humSig = hum(38.0) * 0.06
place(38.8, humSig, 1.0)
for tt in (44.2, 51.7, 60.4, 63.1, 68.9):             # lamp crackle
    place(tt, shaped(int(SR * 0.09), lambda f: np.clip(f / 4000.0, 0, 1)) *
          np.exp(-np.arange(int(SR * 0.09)) / SR * 90), 0.10, -0.2)
place(49.0, metalShing() * 0.5, 0.10, 0.3)            # chair creak / straps
place(51.2, clink(0.4, 1250, 9), 0.14, -0.3)
place(53.4, clink(0.4, 1150, 9), 0.12, 0.35)
place(58.0, breath(1.6), 0.045, 0.0)
place(62.5, heartBeat(), 0.09)
place(66.0, heartBeat(), 0.085)
place(70.0, heartBeat(), 0.08)
for tt in (68.4, 69.3, 70.2, 71.1):                   # guard walks away
    s, d, p = step(0.3, 1.2)
    place(tt, s, 0.13, 0.45)
place(74.6, conv(thud(1.6, 55, 0.5, 5), ROOM), 0.30, 0.0)   # heavy door

# ---------------------------------------------------------------- SCENE 3 (wake)
bed(76.3, 130.0, windLow, 0.10, 3.0, 2.0, (0.4, 1.0))
bed(83.0, 130.0, windMid, 0.055, 3.0, 2.0, (0.4, 1.0))
bed(76.3, 130.0, hissBuf, 0.008, 3.0, 2.0)
tt = 80.5
while tt < 130:                                        # chains of the caravan
    place(tt, clink(0.4, 1500 + rng.uniform(-260, 320), 8.0), rng.uniform(0.035, 0.085), rng.uniform(-0.5, 0.5))
    tt += rng.uniform(0.35, 1.5)
tt = 80.5
while tt < 130:                                        # shuffling steps
    s, d, p = step(0.3, 0.55)
    place(tt, s, rng.uniform(0.05, 0.10), rng.uniform(-0.4, 0.4))
    tt += rng.uniform(0.55, 1.1)
place(84.0, breath(1.4), 0.07, 0.0)
place(88.0, breath(1.2), 0.06, 0.1)
place(99.2, clink(0.55, 1350, 6.0), 0.22, -0.2)       # shackle close-up
place(103.6, clink(0.55, 1200, 6.0), 0.18, 0.25)
place(106.9, breath(1.5), 0.075, 0.0)
place(122.0, pad(9.0, 41.0, 1.0), 0.05)               # the black mountain drones

# ---------------------------------------------------------------- SCENE 4 (king)
bed(129.0, 190.0, windLow, 0.16, 1.5, 2.5, (0.45, 1.0))
bed(129.0, 190.0, windMid, 0.075, 1.5, 2.5, (0.4, 1.0))
bed(129.0, 190.0, windHigh, 0.012, 1.5, 2.5)
for tt, sz in ((130.9, 0.7), (132.4, 1.0), (134.1, 0.8), (136.0, 1.3), (138.2, 0.9), (141.0, 1.6), (143.4, 0.7)):
    place(tt, conv(rockImpact(2.2, sz), CAVE), 0.34 * min(1.0, sz), rng.uniform(-0.7, 0.7))
place(139.5, shaped(int(SR * 0.7), lambda f: np.clip(f / 700.0, 0, 1) * np.exp(-f / 2200.0)) *
      np.exp(-np.arange(int(SR * 0.7)) / SR * 5), 0.06, -0.6)   # distant shout
place(144.0, shaped(int(SR * 0.6), lambda f: np.clip(f / 800.0, 0, 1) * np.exp(-f / 2400.0)) *
      np.exp(-np.arange(int(SR * 0.6)) / SR * 6), 0.05, 0.6)
for tt in (147.0, 151.0, 155.0, 159.2):               # the king's steps
    place(tt, conv(thud(1.8, 44, 0.5, 4.0), CAVE), 0.42, rng.uniform(-0.3, 0.3))
place(149.6, growl(2.6), 0.20, 0.1)
place(157.4, growl(2.2, 47), 0.16, -0.15)
place(152.6, conv(wetTear(1.6), CAVE), 0.30, 0.2)     # a slave dies
place(152.9, shaped(int(SR * 1.1), lambda f: np.clip(f / 500.0, 0, 1) * np.exp(-f / 1800.0)) *
      np.exp(-np.arange(int(SR * 1.1)) / SR * 3.2), 0.14, 0.25)
for tt in (146.5, 148.1, 149.7, 151.4, 153.1, 154.8, 156.6, 158.4, 160.2):
    place(tt, heartBeat(), 0.13)
for tt in (163.4, 165.1, 166.9, 168.8, 171.0, 174.6, 176.1):   # chain trap
    place(tt, clink(0.5, 1450 + rng.uniform(-200, 260), 7.0), rng.uniform(0.12, 0.2), rng.uniform(-0.4, 0.4))
place(170.4, metalShing() * 0.7, 0.12, -0.3)
place(174.9, conv(thud(1.2, 50, 0.6, 6), CAVE), 0.22, 0.1)
place(178.0, breath(1.5), 0.09, 0.0)
place(180.5, breath(1.4), 0.08, 0.0)

# ---------------------------------------------------------------- SCENE 5
bed(188.8, 240.0, windLow, 0.11, 2.5, 2.0, (0.4, 1.0))
bed(188.8, 240.0, windMid, 0.06, 2.5, 2.0, (0.4, 1.0))
bed(215.0, 240.0, windHigh, 0.010, 2.0, 2.0)
for tt in (191.0, 193.5, 196.2, 199.0):
    s, d, p = step(0.32, 0.8)
    place(tt, s, 0.10, rng.uniform(-0.4, 0.4))
place(194.6, metalShing(), 0.16, -0.2)                # a blade is drawn
place(205.4, conv(thud(1.2, 58, 0.45, 6), ROOM), 0.24, 0.0)   # the hero falls
for tt in (206.4, 207.6, 208.9, 210.4):               # coughing blood
    place(tt, (shaped(int(SR * 0.4), lambda f: np.clip(f / 380.0, 0, 1) * np.exp(-f / 1500.0)) *
               np.exp(-np.arange(int(SR * 0.4)) / SR * 9)), 0.17, 0.1)
place(213.0, conv(wetTear(0.9), ROOM), 0.12, 0.15)
place(218.0, breath(1.5), 0.08, 0.0)
bell = bellRing(6.5)
place(225.4, conv(bell, CAVE), 0.34, 0.0)
place(225.4, bell, 0.26, 0.0)
place(234.4, conv(bellRing(5.0, 880), CAVE), 0.16, 0.5)
place(236.0, breath(1.4), 0.06, -0.2)

# ---------------------------------------------------------------- SCENE 6
bed(238.8, 300.0, windLow, 0.20, 2.5, 3.0, (0.4, 1.0))
bed(238.8, 300.0, windMid, 0.09, 2.5, 3.0, (0.45, 1.0))
bed(238.8, 300.0, windHigh, 0.018, 2.5, 3.0)
for tt in (241.0, 243.2, 245.5, 247.9, 250.4):        # crawling over stone
    place(tt, (shaped(int(SR * 0.5), lambda f: np.clip(f / 900.0, 0, 1) * np.exp(-f / 4200.0)) *
               np.exp(-np.arange(int(SR * 0.5)) / SR * 8)), 0.11, rng.uniform(-0.3, 0.3))
    place(tt + 0.15, breath(1.3), 0.07, 0.0)
place(252.0, pad(50.0, 55.0, 1.0), 0.055)
for tt in (254.4, 258.7, 263.2):
    place(tt, drip(), 0.09, rng.uniform(-0.4, 0.4))
for tt in (256.0, 259.5, 263.0, 266.4):
    place(tt, heartBeat(), 0.075)
for tt in (268.5, 271.6, 274.8, 278.2):               # the king ascends
    place(tt, conv(thud(1.8, 40, 0.5, 3.4), CAVE), 0.40, rng.uniform(-0.3, 0.3))
place(269.0, growl(3.0, 44), 0.22, 0.0)
for tt in (270.5, 273.2, 276.0, 279.0):               # shadows tear it apart
    place(tt, conv(wetTear(1.6), CAVE), 0.34, rng.uniform(-0.6, 0.6))
    place(tt + 0.1, conv(thud(1.4, 55, 0.8, 6), CAVE), 0.24, 0.0)
place(272.0, pad(23.0, 55.0, 1.0), 0.12)
place(272.0, pad(23.0, 82.5, 1.0) * 0.6, 0.07)
place(281.0, shimmer(11.0, 1400.0), 0.075)
place(281.0, shimmer(11.0, 2100.0), 0.045)
place(291.5, pad(9.0, 55.0, 1.0), 0.07)

# ---------------------------------------------------------------- voices
def readWav(path):
    with wave.open(path) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        return data, w.getframerate()


VO = [
    # (file, start, kind)
    ("vo01", 3.0, "sunny"),
    ("vo02", 7.5, "sunny"),
    ("vo03", 12.0, "sunny"),
    ("vo04", 18.0, "sunny"),
    ("vo05", 27.0, "sunny"),
    ("vo06", 52.0, "other"),
    ("vo07", 67.5, "whisper"),
    ("vo09", 84.0, "sunny"),
    ("vo08", 112.5, "other"),
    ("vo10", 138.0, "shout"),
    ("vo11", 152.0, "sunny"),
    ("vo12", 196.0, "other"),
    ("vo13", 212.0, "sunny"),
    ("vo14", 272.8, "spell"),
    ("vo15", 278.3, "spell"),
    ("vo16", 284.2, "spell"),
    ("vo17", 288.4, "spell"),
    ("vo18", 295.9, "whisper"),
]

print("mixing voices ...")
for name, start, kind in VO:
    x, _ = readWav(os.path.join(AUDIO, name + ".wav"))
    if kind == "sunny":
        y = conv(x, ROOM) * 0.12 + x * 0.95
        y = norm(y) * 0.92
    elif kind == "whisper":
        y = conv(x, ROOM) * 0.14 + x * 0.9
        y = norm(y) * 0.95
    elif kind == "shout":
        y = conv(x, CAVE) * 0.35 + x
        y = norm(y) * 0.85
    elif kind == "other":
        y = conv(x, ROOM) * 0.22 + x
        y = norm(y) * 0.85
    else:  # spell
        low = pitch(x, 1.30)
        y = norm(low) * 0.9
        y = y + np.concatenate([np.zeros(int(SR * 0.42)), y[:-int(SR * 0.42)]]) * 0.30
        y = y + np.concatenate([np.zeros(int(SR * 0.85)), y[:-int(SR * 0.85)]]) * 0.16
        y = conv(y, GOD) * 0.42 + y * 0.85
        y = norm(y) * 0.95
    place(start, y, 1.0, 0.0)
    print("  %s @ %.1f (%.2fs, %s)" % (name, start, len(x) / 44100.0, kind))

# ---------------------------------------------------------------- master
print("mastering ...")
y = np.stack([mixL, mixR], axis=1)
peak = np.max(np.abs(y)) + 1e-9
y = y / peak * 1.35
y = np.tanh(y * 1.15) * 0.92
y *= envAR(N, 1.0, 1.2)[:, None]
pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
out = os.path.join(AUDIO, "ambience.wav")
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", out, os.path.getsize(out) / 1e6, "MB")
