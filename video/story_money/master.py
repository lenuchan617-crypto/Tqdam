"""Two-pass loudness master with ffmpeg loudnorm (target -14.5 LUFS, TP under -1 dBTP)."""
import json
import os
import subprocess

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = os.path.join(OUT, "mix_pre.wav")
DST = os.path.join(OUT, "mix_money.wav")
TARGET_I = -14.5
TARGET_TP = -1.6
TARGET_LRA = 9.0


def measure(path):
    r = subprocess.run([FF, "-i", path, "-af", "loudnorm=print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    block = r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1]
    return json.loads(block)


def run(src, dst, tp):
    d = measure(src)
    af = (f"loudnorm=I={TARGET_I}:TP={tp}:LRA={TARGET_LRA}:print_format=json:"
          f"measured_I={d['input_i']}:measured_TP={d['input_tp']}:"
          f"measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:"
          f"offset={d['target_offset']}:linear=true,aresample=48000")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, "-af", af,
                    "-ar", "48000", "-ac", "2", dst], check=True)
    return d


def main():
    tp = TARGET_TP
    for attempt in range(3):
        run(SRC, DST, tp)
        m = measure(DST)
        i, true_peak = float(m["input_i"]), float(m["input_tp"])
        print(f"pass {attempt + 1}: {i:.2f} LUFS, tp {true_peak:.2f} dBTP, "
              f"LRA {float(m['input_lra']):.1f} LU")
        if true_peak <= -1.0 and abs(i - TARGET_I) <= 0.4:
            break
        tp -= 1.0           # push the ceiling down and try again
    print("wrote", DST, round(os.path.getsize(DST) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
