"""Build the accessibility SRT from out/timings.json (verbatim VO windows)."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(ROOT, "out", "timings.json")))


def ts(sec):
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


lines = []
vo = data["vo"]
for i, ev in enumerate(vo, 1):
    end = ev["end"] + 0.25
    if i < len(vo):
        end = min(end, vo[i]["start"] - 0.05)
    lines.append(f"{i}\n{ts(ev['start'])} --> {ts(min(60.0, end))}\n{ev['text']}\n")
out = os.path.join(ROOT, "one_on_earth_60s.srt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("wrote", out, len(data["vo"]), "cues")
