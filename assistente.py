import sys, os, tempfile, select, time
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper
import requests
from kokoro_onnx import Kokoro
import memoria
import jogos

SR = 16000
MAX_DURATION = 15
GAIN = 8.0
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
MAX_HISTORICO = 5

os.chdir(os.path.dirname(os.path.abspath(__file__)))
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

SYSTEM_PROMPT = """Voce eh a Luna, uma amiga virtual divertida de 8 anos.
Voce adora brincar, fazer perguntas, inventar historias e descobrir coisas novas junto com seu amigo.
Fala com entusiasmo infantil, usando portugues brasileiro.
NUNCA se comporte como assistente ou robo — voce eh uma amiga de verdade.
Nao use emojis ou simbolos.

Brincadeiras que voce conhece: forca, adivinhacao, 20 perguntas, simao diz, historia coletiva, detetive, palavra secreta, caca ao tesouro."""

mem = memoria.carregar()
historico = []

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
    global historico
    memorias = memoria.formatar_para_prompt(mem)
    system = SYSTEM_PROMPT
    if memorias:
        system += f"\n\nCoisas que voce sabe sobre seu amigo:\n{memorias}"

    messages = [{"role": "system", "content": system}]
    messages.extend(historico)
    messages.append({"role": "user", "content": texto})

    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False
    })
    resposta = r.json()["message"]["content"].strip()

    historico.append({"role": "user", "content": texto})
    historico.append({"role": "assistant", "content": resposta})
    if len(historico) > MAX_HISTORICO * 2:
        historico = historico[-(MAX_HISTORICO * 2):]

    memoria.extrair_fatos(texto + " " + resposta, mem)
    memoria.extrair_temas(texto + " " + resposta, mem)
    memoria.extrair_jogos(texto + " " + resposta, mem)
    memoria.salvar(mem)

    return resposta

def falar(texto: str):
    samples, sr = kokoro.create(texto, voice="pf_dora", speed=1.0, lang="pt-br")
    sd.play(samples, sr)
    sd.wait()

def key_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False

nome_crianca = mem.get("crianca", {}).get("nome", "")
if nome_crianca:
    print(f"Ola {nome_crianca}! A Luna ja esta aqui!")
else:
    print("Ola! Eu sou a Luna! Qual o seu nome?")

print("ENTER para falar, ENTER de novo para parar")

buf = []
stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda i, f, t, s: buf.append(i.copy()))
stream.start()

while True:
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
    input("\nPressione ENTER para falar com a Luna...")

    idx = len(buf)
    print("Falando... (ENTER para parar)", end="", flush=True)
    inicio = time.time()

    while True:
        time.sleep(0.05)
        if key_pressed():
            print("")
            break
        if time.time() - inicio > MAX_DURATION:
            print(" (tempo maximo)")
            break

    full = np.concatenate(buf[idx:])

    print("Entendendo...")
    texto = transcrever(full)

    if not texto:
        print("Nao entendi. Pode falar de novo?")
        continue

    print(f"  Voce: {texto}")

    print("Pensando...")
    resposta = perguntar(texto)
    print(f"  Luna: {resposta}")

    print("Falando...")
    falar(resposta)
