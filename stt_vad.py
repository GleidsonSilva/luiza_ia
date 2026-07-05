import os, sys, tempfile
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper

SR = 16000
CHUNK = 0.5
SILENCE_LIMIT = 6
THRESHOLD = 0.008

print("🎤 Aguardando fala...", end="", flush=True)

buffer = []
silence = 0
started = False

while True:
    chunk = sd.rec(int(SR * CHUNK), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    rms = np.sqrt(np.mean(chunk**2))

    if rms > THRESHOLD:
        if not started:
            print(" gravando...")
            started = True
        silence = 0
        buffer.append(chunk.copy())
    elif started:
        silence += 1
        buffer.append(chunk.copy())
        if silence >= SILENCE_LIMIT:
            break

print("Transcrevendo...")

full = np.concatenate(buffer)
peak = np.max(np.abs(full))
if peak > 0:
    full = full / peak * 0.95

tmp = tempfile.mktemp(suffix=".wav")
sf.write(tmp, full, SR)

result = mlx_whisper.transcribe(
    tmp, language="pt",
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    initial_prompt="ola vamos brincar qual e o seu nome voce gosta de jogar que historia legal"
)
os.remove(tmp)

print(f"\n📝 {result['text'].strip()}")
