#!/bin/sh
# mux silent picture with the mastered mix, then make a lighter delivery copy
set -e
FF=$(python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
$FF -y -loglevel error -i out/video_silent.mp4 -i out/mix_final.wav \
  -map 0:v -map 1:a -c:v libx264 -preset veryfast -crf 33 -pix_fmt yuv420p \
  -profile:v high -r 30 -movflags +faststart \
  -c:a aac -b:a 160k -ar 48000 -shortest minus_40k_6min.mp4
$FF -y -loglevel error -i minus_40k_6min.mp4 \
  -c:v libx264 -preset slow -crf 27 -pix_fmt yuv420p -movflags +faststart \
