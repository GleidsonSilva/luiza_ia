import os
import numpy as np
from scipy import signal

BASE = "/Users/gleidson/Documents/Git/luiza_ia/poc/base_pf_dora.wav"
OUT = "/Users/gleidson/Documents/Git/luiza_ia/poc/abordagem_2_phasevocoder_pitch.wav"
PITCH = 1.22

def read_wav(path):
    import wave
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if w.getnchannels() > 1:
            arr = arr.reshape(-1, w.getnchannels()).mean(axis=1)
        return arr, w.getframerate()

def write_wav(path, audio, sr):
    import wave
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())

y, sr = read_wav(BASE)
n_fft = 1024
hop = n_fft // 4
S = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)[2]
mag = np.abs(S)
phase = np.angle(S)
# phase vocoder pitch shift preserving duration
phase_shift = np.concatenate([phase[:, 0:1], phase[:, 1:] - phase[:, :-1]], axis=1)
freq_axis = np.linspace(0, sr/2, mag.shape[0])
new_mag = np.zeros_like(mag)
for f in range(mag.shape[0]):
    new_f = f / PITCH
    lo = int(np.floor(new_f))
    hi = min(lo + 1, mag.shape[0] - 1)
    alpha = new_f - lo
    new_mag[f] = (1 - alpha) * mag[lo] + alpha * mag[hi]
new_phase = np.zeros_like(phase)
for t in range(1, new_phase.shape[1]):
    new_phase[:, t] = new_phase[:, t-1] + phase_shift[:, t]
S_new = new_mag * np.exp(1j * new_phase)
_, y_back = signal.istft(S_new, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
y_back = y_back[: len(y)]
write_wav(OUT, y_back, sr)
print("ABORDAGEM_2_OK", OUT)
