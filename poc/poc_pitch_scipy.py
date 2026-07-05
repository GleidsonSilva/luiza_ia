import os, wave, math
import numpy as np
from scipy import signal

BASE = "/Users/gleidson/Documents/Git/luiza_ia/poc/base_pf_dora.wav"
OUT = "/Users/gleidson/Documents/Git/luiza_ia/poc/abordagem_1_scipy_pitch.wav"
PITCH = 1.22

def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(w.getsampwidth(), np.int16)
        raw = w.readframes(n)
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if ch > 1:
            audio = audio.reshape(-1, ch).mean(axis=1)
        return audio, sr

def write_wav(path, audio, sr):
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())

audio, sr = read_wav(BASE)
ratio = PITCH
new_len = int(round(len(audio) * ratio))
audio_up = signal.resample(audio, new_len)
audio_back = signal.resample(audio_up, len(audio))
write_wav(OUT, audio_back, sr)
print("ABORDAGEM_1_OK", OUT, "sr=", sr, "pitch=", PITCH)
