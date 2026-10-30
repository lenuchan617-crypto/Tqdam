"""
Build the soundtrack for «ОДИН НА ЗЕМЛЕ» (60 s vertical short).

- 180 BPM punk-funk / breakbeat groove synthesised from scratch (numpy)
- five mandatory SFX from the script (tick, whoosh, slap, BEM, click+hum, vroom)
- narration: trimmed TTS clips, time-compressed (pitch preserved) and placed
- side-chain ducking under the voice, master limited to about -14 LUFS / -1 dBTP

Outputs:  out/mix.wav, out/timings.json
"""
import json
import math
import os
import re
import subprocess
import wave

import imageio_ffmpeg
import numpy as np

SR = 48000
DUR = 60.0
BPM = 180.0
BEAT = 60.0 / BPM
FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- narration ---
# (file, start_seconds, tempo) — tempo is pitch-preserving compression
VO = [
    ("vo01", 0.15, 1.58),
    ("vo02", 4.60, 1.58),
    ("vo03", 7.25, 1.58),
    ("vo04", 10.15, 1.58),
    ("vo05", 13.05, 1.58),
    ("vo06", 19.10, 1.58),
    ("vo07", 27.25, 1.58),
    ("vo08", 34.45, 1.58),
    ("vo09", 40.40, 1.58),
    ("vo10", 46.70, 1.58),
    ("vo11", 55.80, 1.55),  # CTA line
]


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def load_vo():
    events = []
    for name, start, tempo in VO:
        src = os.path.join(ROOT, "vo_trim", name + ".wav")
        if not os.path.exists(src):
            print(f"  ! missing {name}, skipped")
            continue
        dst = os.path.join(OUT, f"{name}_t.wav")
        subprocess.run(
            [FF, "-y", "-loglevel", "error", "-i", src, "-af",
             f"atempo={tempo},aresample={SR}", "-ar", str(SR), "-ac", "1", dst],
            check=True)
        sig = read_wav(dst)
        events.append({"name": name, "start": start, "dur": len(sig) / SR})
        yield events[-1], sig


# ------------------------------------------------------------------- helpers ---
def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def env_ad(n, attack, decay, curve=2.0):
    a = max(1, int(attack * SR))
    d = max(1, n - a)
    e = np.concatenate([np.linspace(0, 1, a), np.linspace(1, 0, d) ** curve])
    return e[:n]


def one_pole_lp(x, cutoff_hz):
    a = math.exp(-2 * math.pi * cutoff_hz / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def lp_fast(x, cutoff_hz):
    """Vectorised-ish low-pass via cumulative smoothing (approximation)."""
    a = math.exp(-2 * math.pi * cutoff_hz / SR)
    return np.convolve(x, (1 - a) * a ** np.arange(0, 60), mode="same")


def saw(freq, t, phase=0.0):
    return 2.0 * ((freq * t + phase) % 1.0) - 1.0


def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


# ---------------------------------------------------------------- drum voices ---
def kick(gain=1.0):
    n = int(0.26 * SR)
    t = t_axis(0.26)
    f = 130 * np.exp(-t * 22) + 44
    body = np.sin(2 * np.pi * np.cumsum(f) / SR)
    click = np.random.RandomState(1).randn(n) * np.exp(-t * 260) * 0.25
    return (body * np.exp(-t * 7.5) + click) * gain


def snare(gain=1.0):
    n = int(0.19 * SR)
    t = t_axis(0.19)
    rng = np.random.RandomState(2)
    noise = rng.randn(n) * np.exp(-t * 26)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 34) * 0.5
    return (noise * 0.7 + tone) * gain


def hat(gain=1.0, open_=False):
    d = 0.055 if not open_ else 0.16
    n = int(d * SR)
    t = t_axis(d)
    rng = np.random.RandomState(3)
    x = rng.randn(n) * np.exp(-t * (95 if not open_ else 28))
    x = x - lp_fast(x, 5200)  # brighten
    return x * gain * 0.5


def clap(gain=1.0):
    out = np.zeros(int(0.22 * SR))
    for k, off in enumerate([0.0, 0.011, 0.024]):
        c = snare(0.5) * gain
        place(out, c[: int(0.09 * SR)], off * 1.0, 1.0 - k * 0.2)
    return out


# ------------------------------------------------------------------- melodic ---
SCALE = {  # A minor, one bar per chord, 4 beats
    "Am": [110.00, 130.81, 164.81],
    "F": [87.31, 130.81, 174.61],
    "C": [130.81, 164.81, 196.00],
    "G": [98.00, 123.47, 196.00],
}
PROG = ["Am", "F", "C", "G"]


def bass_note(freq, dur, gain=1.0):
    n = int(dur * SR)
    t = t_axis(dur)
    x = saw(freq, t) * 0.6 + np.sin(2 * np.pi * freq * 0.5 * t) * 0.5
    x = lp_fast(x, 320)
    e = np.minimum(1.0, np.linspace(0, 6, n)) * np.exp(-t * 5.0)
    return x * e * gain


def stab(freqs, dur=0.16, gain=1.0):
    n = int(dur * SR)
    t = t_axis(dur)
    x = np.zeros(n)
    for f in freqs:
        x += saw(f, t, phase=0.13) * 0.33
    x = lp_fast(x, 2400)
    return x * np.exp(-t * 12) * gain


def pad(freqs, dur, gain=1.0):
    n = int(dur * SR)
    t = t_axis(dur)
    x = np.zeros(n)
    for f in freqs:
        x += (np.sin(2 * np.pi * f * 2 * t) + 0.4 * np.sin(2 * np.pi * f * 3.01 * t))
    x /= len(freqs)
    e = np.minimum(1.0, np.linspace(0, 40, n)) * np.minimum(1.0, np.linspace(12, 0, n))
    return x * e * gain


def tick(gain=1.0):
    n = int(0.05 * SR)
    t = t_axis(0.05)
    rng = np.random.RandomState(5)
    x = rng.randn(n) * np.exp(-t * 190) + np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 150) * 0.5
    return lp_fast(x, 6000) * gain


def whoosh(gain=1.0):
    d = 0.55
    n = int(d * SR)
    t = t_axis(d)
    rng = np.random.RandomState(6)
    noise = rng.randn(n)
    cutoff = 700 + 5200 * (t / d) ** 1.4
    # cheap swept band: modulate with a time-varying gain on high-passed copy
    hp = noise - lp_fast(noise, 900)
    e = (t / d) ** 2.2 * np.exp(-t * 4.0) * 2.2
    return hp * e * gain


def slap(gain=1.0):
    d = 0.22
    n = int(d * SR)
    t = t_axis(d)
    rng = np.random.RandomState(7)
    x = rng.randn(n) * np.exp(-t * 60) * 0.9
    x += np.sin(2 * np.pi * 115 * t) * np.exp(-t * 30) * 0.7
    return x * gain


def bem(gain=1.0):
    d = 0.7
    n = int(d * SR)
    t = t_axis(d)
    f = 120 * np.exp(-t * 9) + 42
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5.5)
    rng = np.random.RandomState(8)
    crunch = rng.randn(n) * np.exp(-t * 22) * 0.5
    return np.tanh((body + crunch) * 2.1) * gain


def click_hum(gain=1.0):
    d = 3.0
    n = int(d * SR)
    t = t_axis(d)
    c = np.zeros(n)
    cl = tick(1.0)
    place(c, cl, 0.0, 1.0)
    hum = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 1.15) * 0.35
    hum += np.sin(2 * np.pi * 120 * t) * np.exp(-t * 1.6) * 0.12
    return (c + hum) * gain


def vroom(gain=1.0):
    d = 1.9
    n = int(d * SR)
    t = t_axis(d)
    f = 58 + 42 * np.minimum(1, t / 0.45) - 18 * np.maximum(0, (t - 0.9) / 1.0)
    ph = np.cumsum(f) / SR
    x = 2 * (ph % 1.0) - 1
    x += 0.5 * np.sin(2 * np.pi * np.cumsum(f * 2) / SR)
    x = lp_fast(x, 900)
    e = np.minimum(1.0, t / 0.05) * np.exp(-np.maximum(0, t - 0.8) * 1.4)
    return np.tanh(x * 2.2) * e * gain


def pop(gain=1.0):
    d = 0.18
    n = int(d * SR)
    t = t_axis(d)
    f = 520 + 900 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18) * gain


# -------------------------------------------------------------------- music ---
def build_music(total=DUR):
    buf = np.zeros(int(total * SR) + SR)
    rng = np.random.RandomState(11)

    # section table: (start_beat, end_beat, drums, bass, hats, stabs, pad)
    sections = [
        (0, 9, 0.0, 0.0, 0.0, 0.0, 0.25),      # 0-3 s: emptiness
        (9, 27, 0.55, 0.5, 0.45, 0.3, 0.35),   # 3-9 s: rhythm enters
        (27, 54, 0.95, 0.9, 0.85, 0.7, 0.0),   # 9-18 s
        (54, 63, 0.35, 0.45, 0.2, 0.0, 0.45),  # 18-21 s: breakdown (lights out)
        (63, 90, 1.0, 0.95, 0.9, 0.8, 0.0),    # 21-30 s
        (90, 99, 0.75, 0.8, 0.6, 0.35, 0.2),   # 30-33 s
        (99, 123, 1.0, 1.0, 0.95, 0.85, 0.0),  # 33-41 s
        (123, 138, 0.95, 0.95, 0.85, 0.7, 0.0),# 41-46 s drive
        (138, 153, 0.8, 0.8, 0.7, 0.5, 0.3),   # 46-51 s
        (153, 168, 0.6, 0.6, 0.5, 0.25, 0.55), # 51-56 s warm balcony
        (168, 178, 1.0, 1.0, 0.95, 0.9, 0.0),  # 56-59.3 s CTA
    ]

    def sec(beat):
        for s in sections:
            if s[0] <= beat < s[1]:
                return s
        return (0, 0, 0, 0, 0, 0, 0)

    for bar in range(int(total / (4 * BEAT)) + 1):
        chord = PROG[bar % 4]
        notes = SCALE[chord]
        root = notes[0] * 0.5  # bass an octave down
        b0 = bar * 4
        for beat_i in range(4):
            beat = b0 + beat_i
            t = beat * BEAT
            if t >= total:
                break
            _, _, drums, bass, hats, stabs, padlvl = sec(beat)
            # kick on 1 & 3
            if beat_i in (0, 2) and drums:
                place(buf, kick(drums), t)
            # snare on 2 & 4
            if beat_i in (1, 3) and drums:
                place(buf, clap(drums), t - 0.004)
                place(buf, snare(drums * 0.45), t)
            # hats on 8ths
            if hats:
                off = 0.25 if beat_i in (1, 3) else 0.32
                place(buf, hat(hats * 0.5), t)
                place(buf, hat(hats * 0.35), t + BEAT / 2)
                if beat_i == 3:
                    place(buf, hat(hats * 0.3, open_=True), t + BEAT * 0.75)
            # bass on 8ths
            if bass:
                place(buf, bass_note(root, BEAT * 0.46, bass * 0.85), t)
                place(buf, bass_note(root * (SCALE[chord][2] / SCALE[chord][0]), BEAT * 0.24,
                                    bass * 0.45), t + BEAT * 0.5)
            # stabs off-beat
            if stabs and beat_i in (1, 3):
                place(buf, stab([n * 2 for n in notes[:2]], 0.15, stabs * 0.4),
                      t + BEAT * 0.5)
        if padlvl:
            place(buf, pad(notes, 4 * BEAT * 1.02, padlvl * 0.5), b0 * BEAT)

    # drum fill before the CTA and before the breakdown
    for fill_beat in (52, 164):
        for k in range(4):
            place(buf, snare(0.35 - k * 0.05), fill_beat * BEAT + k * BEAT / 4)

    # riser into the groove at 3 s and into the CTA at 56 s
    for rt in (2.0, 55.0):
        d = 1.0
        n = int(d * SR)
        t = t_axis(d)
        nz = np.random.RandomState(13).randn(n) * (t / d) ** 2.5
        nz = nz - lp_fast(nz, 1200)
        place(buf, nz * 0.22, rt)

    return buf[: int(total * SR)]


def build_sfx(total=DUR):
    buf = np.zeros(int(total * SR))
    place(buf, tick(0.5), 0.35)
    place(buf, tick(0.35), 1.02)
    place(buf, tick(0.28), 1.69)
    place(buf, whoosh(0.55), 0.95)          # camera dive to the phone
    place(buf, slap(0.8), 8.42)             # cheek slap 1
    place(buf, slap(0.75), 8.92)            # cheek slap 2
    place(buf, whoosh(0.35), 11.35)         # run to the shop
    place(buf, bem(0.95), 12.88)            # kicked door
    place(buf, click_hum(0.6), 18.62)       # switch + dying hum
    place(buf, vroom(0.85), 41.55)          # engine
    place(buf, pop(0.5), 57.60)             # subscribe button
    return buf


# --------------------------------------------------------------------- mix ---
def lowpass_fir(x, cutoff_hz, taps=121):
    """Windowed-sinc low-pass (keeps inter-sample peaks under control)."""
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff_hz / SR
    h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def main():
    voice = np.zeros(int(DUR * SR))
    events = []
    for ev, sig in load_vo():
        place(voice, sig, ev["start"])
        events.append({"name": ev["name"], "start": round(ev["start"], 3),
                       "end": round(ev["start"] + ev["dur"], 3),
                       "text": VO_TEXT.get(ev["name"], "")})
        print(f"  vo {ev['name']:>4} {ev['start']:6.2f} → {ev['start'] + ev['dur']:6.2f}")

    # gentle voice shaping
    voice = np.tanh(voice * 1.15) * 0.9

    music = build_music()
    sfx = build_sfx()

    # side-chain: duck music under speech
    env = np.abs(voice)
    win = int(0.05 * SR)
    k = np.ones(win) / win
    env = np.convolve(env, k, mode="same")
    env = np.convolve(env, np.ones(int(0.30 * SR)) / int(0.30 * SR), mode="same")
    duck = 1.0 - 0.72 * np.clip(env / (env.max() + 1e-9) * 2.4, 0, 1)
    duck = np.convolve(duck, np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same")
    music = music * duck[: len(music)]

    # level targets
    vtarget = -17.0
    mtarget = -21.0
    starget = -19.0
    voice *= 10 ** ((vtarget - 20 * math.log10(rms(voice))) / 20)
    music *= 10 ** ((mtarget - 20 * math.log10(rms(music))) / 20)
    sfx *= 10 ** ((starget - 20 * math.log10(rms(sfx))) / 20)

    mix = voice + music + sfx
    # gentle saturation so loudness comes up before the peak ceiling does,
    # then hold the sample peak at -2.2 dBFS (inter-sample peak stays under -1 dBTP)
    # tame near-Nyquist transients: inter-sample overshoot was blowing the true-peak spec
    mix = lowpass_fir(mix, 15000.0)
    mix = mix / (np.max(np.abs(mix)) + 1e-9)
    mix = np.tanh(mix * 1.9) / np.tanh(1.9)
    cur = 20 * math.log10(rms(mix))
    mix *= 10 ** ((-14.6 - cur) / 20)
    peak = np.max(np.abs(mix))
    if peak > 0.78:
        mix = mix / peak * 0.78
    # fade the very last 0.25 s for a clean loop point
    fade = int(0.25 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)

    stereo = np.stack([mix, mix], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    path = os.path.join(OUT, "mix.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", path, round(rms(mix), 4), "peak", round(float(np.max(np.abs(mix))), 3))

    json.dump({"duration": DUR, "bpm": BPM, "vo": events},
              open(os.path.join(OUT, "timings.json"), "w"), ensure_ascii=False, indent=2)


VO_TEXT = {
    "vo01": "Твой будильник не звонит, потому что интернет умер. Выходишь на улицу — а там никого.",
    "vo02": "Вообще никого. Поздравляю, ты остался один на всей планете!",
    "vo03": "Первые пять минут: паника, крики, отрицание.",
    "vo04": "Но тормозить нельзя. Включаем режим выживания.",
    "vo05": "Шаг первый: Гастроном. Забудь про чипсы и торты. Твой бро на ближайшие годы — вода, тушёнка и крупы.",
    "vo06": "Шаг второй: Свету конец. Через пару дней электростанции встанут. Находишь генератор, закупаешь солярку и ищешь солнечные панели.",
    "vo07": "Шаг третий: Аптека. Забирай антибиотики, бинты и обезбол. Заболеть зуб — твоя главная угроза. Второго шанса подлечиться уже не будет!",
    "vo08": "Шаг четвёртый: Качай мозги. Качай бумажные инструкции по выживанию и фильтрации воды.",
    "vo09": "Шаг пятый: Вали из мегаполиса. Без людей города быстро превратятся в бетонные ловушки. Ищи дом у воды за городом.",
    "vo10": "Забирай самый надёжный внедорожник без лишней электроники и вали на природу. Сажай картоху, фильтруй воду и заводи собаку.",
    "vo11": "А ты бы выжил в таком раскладе? Пиши в комментах и подпишись.",
}

if __name__ == "__main__":
    main()
