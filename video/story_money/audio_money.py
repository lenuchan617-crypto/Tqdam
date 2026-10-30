"""
Soundtrack for «POV: минус сорок тысяч» (6-minute vertical storytime).

Noir bed at 96 BPM built from scratch: sparse kick, brushed snare, upright bass,
electric-piano stabs, vinyl crackle and tape hiss, plus the diegetic SFX the story
needs (ATM beeps, card slot, phone dial, coins, notification buzz, transition
whooshes). Narration is placed on the locked schedule and the music is ducked
under the voice. Master: about -14.5 LUFS / true peak under -1 dBFS.
"""
import json
import math
import os
import wave

import imageio_ffmpeg
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
SR = 48000
FF = imageio_ffmpeg.get_ffmpeg_exe()

with open(os.path.join(OUT, "schedule.json")) as f:
    SCHED = json.load(f)
TOTAL = SCHED["total"]
BPM = 96.0
BEAT = 60.0 / BPM

TENSION = {1: 0.55, 2: 0.55, 3: 0.35, 4: 0.70, 5: 0.75, 6: 0.60,
           7: 0.65, 8: 0.80, 9: 0.95, 10: 0.50, 11: 0.30}


# ------------------------------------------------------------------ helpers ---
def t_axis(d):
    return np.arange(max(1, int(d * SR))) / SR


def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf) or i < 0:
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def lowpass_fir(x, cutoff, taps=121):
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * (cutoff / SR) * np.sinc(2 * (cutoff / SR) * n) * np.blackman(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def saw(f, t, ph=0.0):
    return 2.0 * ((f * t + ph) % 1.0) - 1.0


# ------------------------------------------------------------------ voices ---
def kick(g=1.0):
    d = 0.34
    t = t_axis(d)
    f = 88 * np.exp(-t * 16) + 38
    body = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return np.tanh(body * np.exp(-t * 6.5) * 1.6) * g


def brush(g=1.0):
    d = 0.30
    t = t_axis(d)
    rng = np.random.RandomState(21)
    n = rng.randn(len(t)) * np.exp(-t * 13)
    n = lowpass_fir(n, 6500)
    return n * g * 0.5


def hat(g=1.0):
    d = 0.06
    t = t_axis(d)
    rng = np.random.RandomState(22)
    n = rng.randn(len(t)) * np.exp(-t * 90)
    return (n - lowpass_fir(n, 7000)) * g * 0.45


def bass(f, dur, g=1.0):
    t = t_axis(dur)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2 * t)
    x += 0.18 * saw(f, t)
    x = lowpass_fir(x, 420)
    env = np.minimum(1, t / 0.02) * np.exp(-t * 1.9)
    return x * env * g


def epiano(freqs, dur, g=1.0):
    t = t_axis(dur)
    x = np.zeros(len(t))
    for f in freqs:
        mod = np.sin(2 * np.pi * f * 2.0 * t) * 1.6
        x += np.sin(2 * np.pi * f * t + mod) * np.exp(-t * 3.2)
        x += 0.35 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 7)
    x /= len(freqs)
    env = np.minimum(1, t / 0.008)
    return x * env * g


def drone(f, dur, g=1.0):
    t = t_axis(dur)
    x = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.005 * t)
    env = np.minimum(1, t / 3.0) * np.minimum(1, (dur - t) / 3.0).clip(0, 1)
    return x * env * g


def crackle(dur, density=90, g=1.0):
    rng = np.random.RandomState(31)
    n = int(dur * SR)
    out = np.zeros(n)
    pops = rng.randint(0, n - 400, int(density * dur))
    for p in pops:
        amp = rng.uniform(0.2, 1.0) * rng.choice([1, -1])
        ln = rng.randint(60, 320)
        out[p:p + ln] += amp * np.exp(-np.arange(ln) / (ln * 0.25))
    out = lowpass_fir(out, 7500)
    hiss = rng.randn(n) * 0.02
    return (out * 0.5 + hiss) * g


# --------------------------------------------------------------------- sfx ---
def beep(f=1320, d=0.09, g=1.0):
    t = t_axis(d)
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 24) * g


def slot(g=1.0):
    d = 0.42
    t = t_axis(d)
    rng = np.random.RandomState(41)
    n = rng.randn(len(t))
    n = lowpass_fir(n, 2600) * np.exp(-t * 12)
    n += np.sin(2 * np.pi * 190 * t) * np.exp(-t * 26) * 0.5
    return n * g


def dial(g=1.0):
    t1 = beep(980, 0.11, 1.0)
    out = np.zeros(int(0.55 * SR))
    place(out, t1, 0.02)
    place(out, beep(1470, 0.11, 0.9), 0.17)
    return out * g


def coins(g=1.0):
    out = np.zeros(int(0.9 * SR))
    rng = np.random.RandomState(43)
    for k in range(9):
        f = rng.uniform(2100, 4600)
        d = rng.uniform(0.05, 0.16)
        t = t_axis(d)
        c = np.sin(2 * np.pi * f * t) * np.exp(-t * 30) * rng.uniform(0.25, 0.6)
        place(out, c, rng.uniform(0.0, 0.55))
    return out * g


def buzz(g=1.0):
    d = 0.55
    t = t_axis(d)
    x = (np.sin(2 * np.pi * 168 * t) + np.sin(2 * np.pi * 336 * t) * 0.4)
    gate = (np.sin(2 * np.pi * 12 * t) > 0).astype(float)
    gate = np.convolve(gate, np.ones(200) / 200, mode="same")
    return x * gate * np.exp(-t * 1.2) * 0.5 * g


def whoosh(g=1.0):
    d = 0.65
    t = t_axis(d)
    rng = np.random.RandomState(45)
    n = rng.randn(len(t))
    n = n - lowpass_fir(n, 800)
    return n * (t / d) ** 2.2 * np.exp(-t * 3.4) * 1.6 * g


def paper(g=1.0):
    d = 0.5
    t = t_axis(d)
    rng = np.random.RandomState(47)
    n = rng.randn(len(t))
    n = n - lowpass_fir(n, 3500)
    env = np.exp(-((t - 0.18) ** 2) / 0.004) + 0.6 * np.exp(-((t - 0.33) ** 2) / 0.003)
    return n * env * g


# -------------------------------------------------------------------- music ---
CHORDS = {
    "Am": [110.00, 164.81, 220.00, 261.63],
    "Fmaj7": [87.31, 130.81, 174.61, 220.00],
    "Dm": [73.42, 110.00, 146.83, 174.61],
    "Em": [82.41, 123.47, 164.81, 196.00],
}
PROG = ["Am", "Fmaj7", "Dm", "Em"]


def build_music():
    buf = np.zeros(int(TOTAL * SR) + SR)
    bar = 4 * BEAT
    nb = int(TOTAL / bar) + 1
    for b in range(nb):
        t0 = b * bar
        name = PROG[b % 4]
        notes = CHORDS[name]
        root = notes[0] / 2
        # which chapter are we in
        ch = next((c for c in SCHED["chapters"]
                   if c["start"] <= t0 < c["end"]), SCHED["chapters"][-1])
        if ch.get("kind") == "card":
            ten = 0.28
            dens = 0.30
        elif ch.get("kind") == "outro":
            ten, dens = 0.45, 0.55
        else:
            ten = TENSION.get(ch["id"], 0.6)
            dens = 0.55 + 0.45 * ten

        # sparse noir groove: kick 1 & 3, brushed 2 & 4, hats on 8ths
        if ten > 0.3:
            place(buf, kick(0.55 * ten * dens), t0)
            place(buf, kick(0.40 * ten * dens), t0 + 2 * BEAT)
        if ten > 0.45:
            place(buf, brush(0.55 * ten * dens), t0 + BEAT)
            place(buf, brush(0.50 * ten * dens), t0 + 3 * BEAT)
        for k in range(8):
            g = 0.10 + 0.10 * ten
            place(buf, hat(g * dens), t0 + k * BEAT / 2)
        # walking bass
        if ten > 0.35:
            walk = [root, root * 1.5, notes[1] / 2, root * 1.25]
            for k, f in enumerate(walk):
                place(buf, bass(f, BEAT * 0.82, 0.55 * ten), t0 + k * BEAT)
        # e-piano on the off-beat, softer in cards
        place(buf, epiano(notes[1:], bar * 0.9, 0.22 * (0.4 + ten)), t0 + BEAT * 0.5)
        if ch.get("kind") in ("card", "outro") or ten < 0.45:
            place(buf, drone(notes[0], bar * 1.05, 0.10), t0)

    # vinyl bed for the whole runtime, quieter in the bright ending
    buf[: int(TOTAL * SR)] += crackle(TOTAL, density=26, g=0.5)
    return buf[: int(TOTAL * SR)]


# ---------------------------------------------------------------------- sfx ---
def build_sfx():
    buf = np.zeros(int(TOTAL * SR))
    ch = {c["id"]: c for c in SCHED["chapters"] if "id" in c}
    # chapter 1 — ATM
    place(buf, beep(1180, 0.10, 0.5), ch[1]["vo_start"] - 0.55)
    place(buf, slot(0.6), ch[1]["vo_start"] - 0.30)
    place(buf, beep(720, 0.16, 0.55), ch[1]["vo_start"] + 15.6)
    # chapter 5 — dialling the bank
    place(buf, dial(0.55), ch[5]["vo_start"] - 0.35)
    # chapter 6 — selling things
    place(buf, coins(0.55), ch[6]["vo_start"] + 11.2)
    # chapter 9 — the phone glow
    place(buf, buzz(0.6), ch[9]["vo_start"] + 0.9)
    # intermissions + chapter turns: whoosh
    for c in SCHED["chapters"]:
        if c.get("kind") == "card":
            place(buf, whoosh(0.35), c["start"] - 0.30)
            place(buf, paper(0.35), c["start"] + 0.45)
    for i in range(1, len(SCHED["chapters"])):
        c = SCHED["chapters"][i]
        if not c.get("kind"):
            place(buf, whoosh(0.22), c["start"] - 0.22)
    place(buf, whoosh(0.30), SCHED["chapters"][-1]["start"] - 0.3)
    return buf


# ---------------------------------------------------------------------- mix ---
def read_wav(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main():
    voice = np.zeros(int(TOTAL * SR))
    cues = []
    for c in SCHED["chapters"]:
        if c.get("kind"):
            continue
        sig = read_wav(os.path.join(OUT, c["vo"] + "_t.wav"))
        place(voice, sig, c["vo_start"])
        cues.append(dict(id=c["id"], start=c["vo_start"],
                         end=round(c["vo_start"] + len(sig) / SR, 3)))
    voice = np.tanh(voice * 1.2) * 0.92

    music = build_music()
    sfx = build_sfx()

    # duck the bed under the voice
    env = np.abs(voice)
    k = int(0.05 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    k2 = int(0.35 * SR)
    env = np.convolve(env, np.ones(k2) / k2, mode="same")
    duck = 1.0 - 0.60 * np.clip(env / (env.max() + 1e-9) * 2.2, 0, 1)
    duck = np.convolve(duck, np.ones(int(0.15 * SR)) / int(0.15 * SR), mode="same")
    music *= duck[: len(music)]

    voice *= 10 ** ((-17.0 - 20 * math.log10(rms(voice))) / 20)
    music *= 10 ** ((-21.5 - 20 * math.log10(rms(music))) / 20)
    sfx *= 10 ** ((-20.0 - 20 * math.log10(rms(sfx))) / 20)

    mix = voice + music + sfx
    mix = lowpass_fir(mix, 15500, taps=161)     # tame inter-sample overshoot
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.98
    # gentle fades at both ends
    f = int(0.6 * SR)
    mix[:f] *= np.linspace(0, 1, f)
    mix[-f:] *= np.linspace(1, 0, f)

    stereo = np.stack([mix, mix], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    path = os.path.join(OUT, "mix_pre.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", path, "rms", round(rms(mix), 4), "peak", round(float(np.max(np.abs(mix))), 3))
    json.dump(cues, open(os.path.join(OUT, "voice_cues.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
