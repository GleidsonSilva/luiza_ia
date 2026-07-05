import os, sys, tempfile
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper

SR = 16000
CHUNK = 0.5
SILENCE_LIMIT = 6
THRESHOLD = 0.005

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

print("Pronto.")

full = np.concatenate(buffer)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
os.makedirs(out_dir, exist_ok=True)

# Salva o bruto (sem ganho) pra ouvir
sf.write(os.path.join(out_dir, "audio_bruto.wav"), full, SR)

# Aplica ganho SEM deixar saturar
peak = np.max(np.abs(full))
gain = min(10.0, 0.95 / peak) if peak > 0 else 1.0
full_norm = full * gain

sf.write(os.path.join(out_dir, "audio_normalizado.wav"), full_norm, SR)

print(f"\n   Pico original: {peak:.4f}")
print(f"   Ganho aplicado: {gain:.1f}x")
print(f"   Pico final: {np.max(np.abs(full_norm)):.4f}")
print(f"\n📁 debug/audio_bruto.wav")
print(f"📁 debug/audio_normalizado.wav")

print("\n📝 Transcrevendo...")
tmp = tempfile.mktemp(suffix=".wav")
sf.write(tmp, full_norm, SR)
r = mlx_whisper.transcribe(
    tmp, language="pt",
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    initial_prompt="ola vamos brincar qual e o seu nome voce gosta de jogar que historia legal"
)
os.remove(tmp)
print(f"   Resultado: {r['text'].strip()}")
