import sys
sys.path.insert(0, '/Users/gleidson/Documents/Git/luiza_ia')
from kokoro_onnx import Kokoro
import numpy as np
import soundfile as sf

MODEL = '/Users/gleidson/Documents/Git/luiza_ia/models/kokoro-v1.0.onnx'
VOICES = '/Users/gleidson/Documents/Git/luiza_ia/models/voices-v1.0.bin'
OUT = '/Users/gleidson/Documents/Git/luiza_ia/poc/abordagem_3_voice_blend_pf_dora_af_heart.wav'
TEXTO = "Olá! Eu sou a Luiza, sua amiga virtual. Que legal te ver aqui!"
VOZ_A = 'pf_dora'
VOZ_B = 'af_heart'
PESO_A = 0.55
PESO_B = 0.45
SPEED = 1.05
LANG = 'pt-br'

kokoro = Kokoro(MODEL, VOICES)
style_a = kokoro.get_voice_style(VOZ_A)
style_b = kokoro.get_voice_style(VOZ_B)
emb = PESO_A * style_a + PESO_B * style_b
emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-9)
samples, sr = kokoro.create(TEXTO, voice=emb, speed=SPEED, lang=LANG)
sf.write(OUT, samples, sr)
print('ABORDAGEM_3_OK', OUT, 'voices=', VOZ_A, VOZ_B)
