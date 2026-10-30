"""
Timeline for «POV: минус сорок тысяч» — 6-minute vertical storytime.

Single source of truth: every other module (audio, video, srt) imports SCHEDULE.
Chapter narration is measured from the trimmed WAVs, so the picture follows the
voice instead of the other way round; the leftover time is distributed into
breathing beats and graphic intermissions and the whole cut is locked to 360.0 s.
"""
import json
import os
import subprocess
import wave

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 48000
FF = imageio_ffmpeg.get_ffmpeg_exe()
TOTAL = 360.0
FPS = 30
W, H = 1080, 1920

TEMPO = 1.0           # natural read: the new voice already carries the energy
LEAD = 0.75           # silence before the first word of a chapter
TAIL = 1.35           # hold after the last word
CARD_MAX = 11.7       # a graphic intermission never runs longer than this

CHAPTERS = [
    # (id, voice file, image, kicker, headline, second line)
    (1, "b01", "m01", "ГЛАВА 01", "МИНУС СОРОК", "26 дней до платежа"),
    (2, "b02", "m02", "ГЛАВА 02", "ТРИДЦАТЬ МАЛЕНЬКИХ", "Как я туда попал"),
    (3, "b03", "m03", "ГЛАВА 03", "ПЕРВЫЙ ДЕНЬ", "Таблица, а не паника"),
    (4, "b04", "m04", "ГЛАВА 04", "НЕ ЗАНИМАЙ, ЧТОБЫ ЗАКРЫТЬ", "Кредит на кредит"),
    (5, "b05", "m04", "ГЛАВА 05", "ЗВОНОК В БАНК", "Сам, первым, с цифрами"),
    (6, "b06", "m05", "ГЛАВА 06", "ПРОДАЙ СТАТУС", "31 000 вернулись"),
    (7, "b07", "m06", "ГЛАВА 07", "ТРИ КОНВЕРТА", "Дели сразу, не в конце"),
    (8, "b08", "m07", "ГЛАВА 08", "РАБОТА НА ЭТУ НЕДЕЛЮ", "Не мечта. Воздух"),
    (9, "b08", "m08", "ГЛАВА 09", "БЫСТРЫХ ДЕНЕГ НЕТ", "Ловушки дна"),
    (10, "b09", "m09", "ГЛАВА 10", "СПОКОЙНОЕ УТРО", "Минус 26 000"),
    (11, "b10", "m10", "ФИНАЛ", "Я НЕ СТАЛ БОГАТЫМ", "Я знаю, сколько у меня денег"),
]

# graphic intermissions: (after chapter, duration, [(text, style, at_rel, dur)])
INTERMISSIONS = [
    (2, 10.0, [
        ("МОЯ ТАБЛИЦА", "kicker", 0.15, 3.0),
        ("5 ДОЛГОВ · 4 ПЛАТЕЖА · 1 ЗАРПЛАТА", "line", 1.10, 4.2),
        ("−40 000", "number", 2.60, 5.0),
    ]),
    (4, 9.0, [
        ("ПРАВИЛО", "kicker", 0.15, 2.4),
        ("ДОЛГ ЗАКРЫВАЕТСЯ", "line", 1.00, 4.0),
        ("ТОЛЬКО ДЕНЬГАМИ", "line", 2.10, 4.4),
    ]),
    (6, 9.0, [
        ("ЧТО УШЛО С МОЛОТКА", "kicker", 0.15, 2.6),
        ("КАМЕРА · КОНСОЛЬ · ЧАСЫ", "line", 0.95, 3.8),
        ("КРОССОВКИ", "line", 1.95, 3.0),
        ("+31 000", "number", 3.20, 4.4),
    ]),
    (3, 9.0, [
        ("ЧТО В ТАБЛИЦЕ", "kicker", 0.15, 2.6),
        ("ДОЛГ · СТАВКА", "line", 0.95, 3.4),
        ("ПЛАТЁЖ · СРОК", "line", 2.10, 3.4),
        ("ЦИФРА ПУГАЕТ, ПОКА ЕЁ НЕ ВИДИШЬ", "line", 3.60, 4.4),
    ]),
    (10, 9.0, [
        ("ИТОГ ЗА 60 ДНЕЙ", "kicker", 0.15, 2.6),
        ("−14 000 К ДОЛГУ", "line", 1.00, 3.4),
        ("0 НОВЫХ КРЕДИТОВ", "line", 2.20, 3.4),
        ("ПОДУШКА: 10% КАЖДЫЙ МЕСЯЦ", "line", 3.80, 4.4),
    ]),
    (8, 9.0, [
        ("БЫЛО / СТАЛО", "kicker", 0.15, 2.4),
        ("ДОЛГ −40 000", "line", 1.00, 3.6),
        ("ДОЛГ −26 000", "line", 2.30, 3.6),
        ("0 НОВЫХ КРЕДИТОВ", "line", 4.10, 3.6),
    ]),
]

OUTRO = 15.0
FINAL_CHECK = [
    (2.0, "ЧЕКЛИСТ", "kicker", 3.0),
    (2.4, "1. ВЫПИШИ ВСЕ ДОЛГИ", "line", 4.0),
    (4.0, "2. ПОЗВОНИ В БАНК ПЕРВЫМ", "line", 4.0),
    (5.6, "3. ПРОДАЙ ЛИШНЕЕ", "line", 4.0),
    (7.2, "4. ТРИ КОНВЕРТА", "line", 4.0),
    (8.8, "5. НОЛЬ НОВЫХ КРЕДИТОВ", "line", 5.0),
]


def wav_dur(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def prepare_voice():
    """Trim + tempo-shift every clip once; return measured durations."""
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    out = {}
    for _, name, *_ in CHAPTERS:
        src = os.path.join(ROOT, "vo_trim", name + ".wav")
        dst = os.path.join(ROOT, "out", name + "_t.wav")
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, "-af",
                            f"atempo={TEMPO},aresample={SR}", "-ar", str(SR),
                            "-ac", "1", dst], check=True)
        out[name] = wav_dur(dst)
    return out


def build():
    durs = prepare_voice()

    def assemble(lead, tail, cards):
        out = []
        t = 0.0
        for idx, (cid, name, img, kicker, head, sub) in enumerate(CHAPTERS):
            start = t
            vo_start = start + lead
            vo_end = vo_start + durs[name]
            end = vo_end + tail
            out.append(dict(id=cid, vo=name, img=img, kicker=kicker, headline=head,
                            subline=sub, start=round(start, 3),
                            vo_start=round(vo_start, 3), vo_end=round(vo_end, 3),
                            end=round(end, 3)))
            t = end
            for (after, dur, beats) in INTERMISSIONS:
                if after == cid:
                    d = dur * cards
                    beats_out = []
                    for (text, style, rel, blen) in beats:
                        beats_out.append(dict(text=text, style=style,
                                              start=round(t + rel * cards, 3),
                                              end=round(t + (rel + blen) * cards, 3)))
                    out.append(dict(kind="card", after=cid, start=round(t, 3),
                                    end=round(t + d, 3), beats=beats_out))
                    t += d
        return out, t

    natural, t_nat = assemble(LEAD, TAIL, 1.0)
    target = TOTAL - OUTRO
    slack = target - t_nat
    card_base = sum(c[1] for c in INTERMISSIONS)
    # cards may stretch up to CARD_MAX each, the rest becomes breathing beats
    card_scale = max(1.0, min(CARD_MAX / 9.0, 1.0 + slack / card_base))
    used = card_base * (card_scale - 1.0)
    gap_total = len(CHAPTERS) * (LEAD + TAIL)
    k = 1.0 + (slack - used) / gap_total
    chapters, t_final = assemble(LEAD * k, TAIL * k, card_scale)

    outro_start = TOTAL - OUTRO
    outro = dict(kind="outro", start=round(outro_start, 3), end=TOTAL, beats=[
        dict(text=txt, style=st, start=round(outro_start + rel, 3),
             end=round(min(TOTAL, outro_start + rel + blen), 3))
        for (rel, txt, st, blen) in FINAL_CHECK
    ])
    chapters.append(outro)

    schedule = dict(total=TOTAL, fps=FPS, size=[W, H], tempo=TEMPO,
                    breathe=round(k, 4), chapters=chapters,
                    voice={k2: round(v, 3) for k2, v in durs.items()})
    with open(os.path.join(ROOT, "out", "schedule.json"), "w") as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)
    return schedule


if __name__ == "__main__":
    s = build()
    for ch in s["chapters"]:
        if ch.get("kind"):
            print(f"  [{ch['kind']:6}] {ch['start']:7.2f} → {ch['end']:7.2f}  "
                  f"({ch['end'] - ch['start']:5.2f}s)")
        else:
            print(f"  гл.{ch['id']:>2}  {ch['start']:7.2f} → {ch['end']:7.2f}  "
                  f"голос {ch['vo_start']:7.2f}–{ch['vo_end']:7.2f}")
    print("total", s["chapters"][-1]["end"])
