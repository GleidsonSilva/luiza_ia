import sys, os, json, tempfile, select, time
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper
import requests
from kokoro_onnx import Kokoro
import memoria

SR = 16000
MAX_DURATION = 15
GAIN = 8.0
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
MAX_HISTORICO = 5

os.chdir(os.path.dirname(os.path.abspath(__file__)))
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

SYSTEM_PROMPT = """### PERSONA
Voce eh a Luiza, uma amiga de 8 anos.
Voce gosta de conversar e fazer amizades.
NUNCA se comporte como assistente ou robo.
Nao use emojis ou simbolos.

### COMO FALAR
- Fale igual uma crianca: frases curtas, palavras simples
- Deixe a crianca escolher o assunto
- Nao tente comecar brincadeira sozinha
- Se ela quiser brincar, brinque junto
- Se ela quiser so conversar, converse normal
- Nao de ordens nem comandos

### REGRAS
- No maximo 3 frases curtas por resposta
- Se a crianca estiver triste, acolha com carinho

### SE A CRIANCA PEDIR PARA BRINCAR, VOCE SABE:
forca, adivinhacao, 20 perguntas, cadeia de palavras,
o que sou eu, verdadeiro ou falso, rima comigo,
categorias/stop, complete a frase, historia coletiva,
detetive, palavra secreta

### MEMORIA
- Quando a crianca disser o nome dela, use a funcao lembrar_nome
- Quando ela contar algo sobre ela (idade, cor favorita, bichinho, etc), use a funcao lembrar_fato
- Nao pergunte explicitamente - apenas salve quando ela falar naturalmente"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "lembrar_nome",
            "description": "Salva o nome da crianca quando ela se apresenta",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome da crianca"
                    }
                },
                "required": ["nome"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lembrar_fato",
            "description": "Salva uma informacao que a crianca contou sobre ela mesma",
            "parameters": {
                "type": "object",
                "properties": {
                    "chave": {
                        "type": "string",
                        "description": "ex: idade, cor_favorita, bichinho, brinquedo_favorito, comida_favorita"
                    },
                    "valor": {
                        "type": "string",
                        "description": "O valor que a crianca disse"
                    }
                },
                "required": ["chave", "valor"]
            }
        }
    }
]

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

    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    body = r.json()
    msg = body["message"]
    resposta = msg.get("content", "").strip()
    tool_calls = msg.get("tool_calls")

    tool_used = False
    if tool_calls:
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name == "lembrar_nome" and args.get("nome"):
                memoria.adicionar_fato(mem, "nome", args["nome"])
                tool_used = True
            elif name == "lembrar_fato" and args.get("chave") and args.get("valor"):
                memoria.adicionar_fato(mem, args["chave"], args["valor"])
                tool_used = True

    if not tool_used:
        memoria.extrair_fatos(texto + " " + resposta, mem)
    memoria.extrair_temas(texto + " " + resposta, mem)
    memoria.extrair_jogos(texto + " " + resposta, mem)
    memoria.salvar(mem)

    hist_entry = {"role": "assistant", "content": resposta}
    if tool_calls:
        hist_entry["tool_calls"] = tool_calls
    historico.append({"role": "user", "content": texto})
    historico.append(hist_entry)
    if len(historico) > MAX_HISTORICO * 2:
        historico = historico[-(MAX_HISTORICO * 2):]

    return resposta

def falar(texto: str):
    samples, sr = kokoro.create(texto, voice="pf_dora", speed=1.3, lang="pt-br")
    sd.play(samples, sr)
    sd.wait()

def key_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False

nome_crianca = mem.get("crianca", {}).get("nome", "")
if nome_crianca:
    print(f"Ola {nome_crianca}! A Luiza ja esta aqui!")
else:
    print("Ola! Eu sou a Luiza! Qual o seu nome?")

print("ENTER para falar, ENTER de novo para parar")

buf = []
stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda i, f, t, s: buf.append(i.copy()))
stream.start()

while True:
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
    input("\nPressione ENTER para falar com a Luiza...")

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
    print(f"  Luiza: {resposta}")

    print("Falando...")
    falar(resposta)
