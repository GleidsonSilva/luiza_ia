#!/bin/bash
set -euo pipefail
cd /Users/gleidson/Documents/Git/luiza_ia
echo "==> Base..."
python3 poc/poc_base.py
echo "==> Abordagem 1: scipy pitch..."
python3 poc/poc_pitch_scipy.py
echo "==> Abordagem 2: phase vocoder..."
python3 poc/poc_pitch_phasevocoder.py
echo "==> Abordagem 3: voice blend..."
/Users/gleidson/Documents/Git/luiza_ia/.venv/bin/python3 poc/poc_voice_blend.py
echo "==> Checando duracoes..."
ffprobe -v error -show_entries format=duration -of csv=p=0 poc/base_pf_dora.wav > /tmp/dur_base.txt 2>/dev/null || true
for f in poc/abordagem_1_scipy_pitch.wav poc/abordagem_2_phasevocoder_pitch.wav poc/abordagem_3_voice_blend_pf_dora_af_heart.wav; do ffprobe -v error -show_entries format=duration -of csv=p=0 "$f"; done
ls -lh poc/abordagem_*.wav poc/base_pf_dora.wav
