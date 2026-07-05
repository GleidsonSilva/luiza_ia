import os
import sys
import subprocess
from pathlib import Path

# Garante que estamos usando o venv do projeto
sys.path.insert(0, '/Users/gleidson/Documents/Git/luiza_ia')

TEXTO = "Olá! Eu sou a Luiza, sua amiga virtual. Que legal te ver aqui!"
VOZ = "pf_dora"
SPEED = 1.0
LANG = "pt-br"
MODEL = "/Users/gleidson/Documents/Git/luiza_ia/models/kokoro-v1.0.onnx"
VOICES = "/Users/gleidson/Documents/Git/luiza_ia/models/voices-v1.0.bin"
POC_DIR = Path("/Users/gleidson/Documents/Git/luiza_ia/poc")
BASE_WAV = POC_DIR / "base_pf_dora.wav"

VENV_PYTHON = "/Users/gleidson/Documents/Git/luiza_ia/.venv/bin/python"

def gerar_base():
    script = f"""
import sys
sys.path.insert(0, '/Users/gleidson/Documents/Git/luiza_ia')
from kokoro_onnx import Kokoro
import soundfile as sf

kokoro = Kokoro('{MODEL}', '{VOICES}')
samples, sr = kokoro.create('{TEXTO}', voice='{VOZ}', speed={SPEED}, lang='{LANG}')
sf.write('{BASE_WAV}', samples, sr)
print('BASE_OK', len(samples), sr)
"""
    result = subprocess.run([VENV_PYTHON, "-c", script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("Falha ao gerar áudio base")
    return BASE_WAV

if __name__ == "__main__":
    gerar_base()
