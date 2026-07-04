import sys, os, tempfile, select, time
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper
import requests
from kokoro_onnx import Kokoro

SR = 16000
MAX_DURATION = 15
GAIN = 8.0
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "glm-4.7-flash:q4_K_M"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

SYSTEM_PROMPT = "Você é um assistente infantil amigável, alegre e divertido. Responda em português brasileiro de forma curta, clara e simples, como se falasse com uma crianca. Nao use emotes, emojis ou simbolos."

def trim_silence(audio: np.ndarray, threshold=0.005) -> np.ndarray:
    mask = np.abs(audio) > threshold
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return audio
    return audio[idxs[0]:idxs[-1] + 1]

def transcrever(audio: np.ndarray) -> str:
    audio = trim_silence(audio)
    if len(audio) < SR * 0.3:
        return ""
    audio = audio * GAIN
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95

    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio, SR)

    r = mlx_whisper.transcribe(
        tmp, language="pt",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        initial_prompt="ola vamos brincar qual e o seu nome voce gosta de jogar que historia legal"
    )
    os.remove(tmp)
    return r["text"].strip()

def perguntar(texto: str) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texto}
        ],
        "stream": False
    })
    return r.json()["message"]["content"].strip()

def falar(texto: str):
    samples, sr = kokoro.create(texto, voice="pf_dora", speed=1.0, lang="pt-br")
    sd.play(samples, sr)
    sd.wait()

def key_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False

print("╔══════════════════════════════════╗")
print("║   Assistente de Voz Infantil     ║")
print("╚══════════════════════════════════╝")
print("ENTER para falar, ENTER de novo para parar (ou 15s max)")

buf = []
stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda i, f, t, s: buf.append(i.copy()))
stream.start()

while True:
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
    input("\n▶  ENTER para falar...")

    idx = len(buf)
    print("🎤 Gravando... (ENTER para parar)", end="", flush=True)
    inicio = time.time()

    while True:
        time.sleep(0.05)
        if key_pressed():
            print("")
            break
        if time.time() - inicio > MAX_DURATION:
            print(" (tempo máximo)")
            break

    full = np.concatenate(buf[idx:])

    print("📝 Transcrevendo...")
    texto = transcrever(full)

    if not texto:
        print("Nada reconhecido.")
        continue

    print(f"   Você: {texto}")

    print("🤖 Pensando...")
    resposta = perguntar(texto)
    print(f"   Assistente: {resposta}")

    print("🔊 Falando...")
    falar(resposta)
