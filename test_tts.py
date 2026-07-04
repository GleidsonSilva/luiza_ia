import os
import soundfile as sf
from kokoro_onnx import Kokoro

dir = os.path.dirname(os.path.abspath(__file__))
model = os.path.join(dir, "models", "kokoro-v1.0.onnx")
voices = os.path.join(dir, "models", "voices-v1.0.bin")

kokoro = Kokoro(model, voices)

texto = "Olá mestre Gleidson, estou a seu dispor"
samples, sr = kokoro.create(texto, voice="pf_dora", speed=1.0, lang="pt-br")

out = os.path.join(dir, "teste_tts.wav")
sf.write(out, samples, sr)
print(f"Audio salvo: {out} ({len(samples)/sr:.2f}s, {sr}Hz)")
