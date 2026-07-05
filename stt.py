import os, sys, tempfile
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import sounddevice as sd
import soundfile as sf
import mlx_whisper

SR = 16000
DURATION = 5

print(f"Fale agora (gravando {DURATION}s)...")
audio = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype="float32")
sd.wait()
print("Transcrevendo...")

tmp = tempfile.mktemp(suffix=".wav")
sf.write(tmp, audio, SR)

result = mlx_whisper.transcribe(tmp, language="pt", path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
os.remove(tmp)

print(f"\n📝 {result['text'].strip()}")
