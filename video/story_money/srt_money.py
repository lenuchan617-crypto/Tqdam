"""Accessibility SRT built from the same caption chunks the video shows."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_money import CAPTIONS, SCHED  # noqa: E402


def ts(s):
    h = int(s // 3600)
    m = int(s % 3600 // 60)
    return f"{h:02d}:{m:02d}:{s % 60:06.3f}".replace(".", ",")


# merge chunks shorter than 1.2 s into the neighbour for readability
cues = []
for c in CAPTIONS:
    if cues and (c["end"] - c["start"] < 1.2 or len(c["text"]) < 34):
        cues[-1]["text"] += " " + c["text"]
        cues[-1]["end"] = c["end"]
    else:
        cues.append(dict(text=c["text"], start=c["start"], end=c["end"]))

lines = []
for i, c in enumerate(cues, 1):
    lines.append(f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}\n")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minus_40k_6min.srt")
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", out, len(cues), "cues, video", SCHED["total"], "s")
