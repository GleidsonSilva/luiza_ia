import sys, os, json, tempfile, select, time, re, threading

import tqdm.std
tqdm.std.TqdmDefaultWriteLock.mp_lock = threading.RLock()
tqdm.std.TqdmDefaultWriteLock.create_mp_lock = classmethod(
    lambda cls: setattr(cls, "mp_lock", threading.RLock())
)

from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper
import requests
from kokoro_onnx import Kokoro
import memoria
import contexto
import debug

SR = 16000
MAX_DURATION = 15
TARGET_PEAK = 0.95
MAX_GAIN = 10.0
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
- Mostre interesse pela crianca, faca perguntas sobre ela
- Se nao souber o nome, pergunte "qual o seu nome?"
- Quando ela contar algo, demonstre curiosidade e pergunte mais
- Nao force assunto - aproveite as oportunidades na conversa
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
- Quando a crianca disser o nome, use lembrar_nome
- Quando ela contar algo sobre ela, use lembrar_fato para guardar
- Ao aprender algo novo, mostre empolgação e interesse
- Chame a crianca pelo nome

### CONTEXTO
- Data, hora e periodo do dia sao informados automaticamente pra voce
- Se a crianca perguntar sobre o clima, veja se voce sabe a cidade dela
- Se souber a cidade, use a funcao buscar_clima
- Se nao souber, pergunte primeiro onde ela mora
- Se a crianca mencionar um animal, brinquedo, lugar ou assunto que voce nao conhece bem, pesquise com buscar_web
- Se ela falar "vamos falar sobre X" e voce nao souber muito, busque informacoes
- Nao finja saber de algo que nao conhece — pesquise"""

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
                        "description": "ex: idade, cor_favorita, bichinho, brinquedo_favorito, comida_favorita, cidade, nome_escola, nome_melhor_amigo"
                    },
                    "valor": {
                        "type": "string",
                        "description": "O valor que a crianca disse"
                    }
                },
                "required": ["chave", "valor"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_clima",
            "description": "Busca o clima atual de uma cidade",
            "parameters": {
                "type": "object",
                "properties": {
                    "cidade": {
                        "type": "string",
                        "description": "Nome da cidade para buscar o clima"
                    }
                },
                "required": ["cidade"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_web",
            "description": "Pesquisa algo na internet que voce nao sabe responder",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "O que pesquisar na internet"
                    }
                },
                "required": ["consulta"]
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
    rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0
    debug.log(f"transcrever: after trim len={len(audio)} rms={rms:.5f}")
    if len(audio) < SR * 0.3:
        debug.log(f"too short ({len(audio)} < {SR * 0.3})")
        return ""
    peak = np.max(np.abs(audio))
    if peak > 0:
        gain = min(MAX_GAIN, TARGET_PEAK / peak)
        audio = audio * gain

    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio, SR)

    r = mlx_whisper.transcribe(
        tmp, language="pt",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        initial_prompt="ola vamos brincar qual e o seu nome voce gosta de jogar que historia legal"
    )
    os.remove(tmp)
    texto = r["text"].strip()
    debug.log(f"whisper raw: '{texto}'")
    if not texto or not re.search(r'[aeiouáéíóúâêîôûãõ]', texto, re.I):
        debug.log("vowel check failed")
        return ""
    return texto

def perguntar(texto: str) -> str:
    global historico
    system = SYSTEM_PROMPT + f"\n\n### AGORA\n{contexto.contexto_hoje()}"
    memorias = memoria.formatar_para_prompt(mem)
    if memorias:
        system += f"\n\n### SOBRE SEU AMIGO\n{memorias}"

    messages = [{"role": "system", "content": system}]
    messages.extend(historico)
    messages.append({"role": "user", "content": texto})

    resposta = ""
    needs_rerun = False
    max_iters = 3
    iters = 0

    while iters < max_iters:
        iters += 1
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "tools": tools, "stream": False
        }, timeout=30)
        body = r.json()
        msg = body["message"]
        if msg.get("content"):
            resposta = msg["content"].strip()
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            break

        messages.append(msg)
        needs_rerun = False

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)

            result = ""
            if name == "lembrar_nome" and args.get("nome"):
                memoria.adicionar_fato(mem, "nome", args["nome"])
                result = f"Nome salvo: {args['nome']}"
            elif name == "lembrar_fato" and args.get("chave") and args.get("valor"):
                memoria.adicionar_fato(mem, args["chave"], args["valor"])
                result = f"{args['chave']} salvo: {args['valor']}"
            elif name == "buscar_clima" and args.get("cidade"):
                result = contexto.buscar_clima(args["cidade"])
                needs_rerun = True
            elif name == "buscar_web" and args.get("consulta"):
                result = contexto.buscar_web(args["consulta"])
                needs_rerun = True
            else:
                result = "Nao foi possivel executar."

            messages.append({"role": "tool", "content": result, "name": name})

        if not needs_rerun:
            break

    memoria.extrair_fatos(texto + " " + resposta, mem)
    memoria.extrair_temas(texto + " " + resposta, mem)
    memoria.extrair_jogos(texto + " " + resposta, mem)
    memoria.salvar(mem)

    historico.append({"role": "user", "content": texto})
    historico.append({"role": "assistant", "content": resposta})
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

buf_lock = threading.Lock()
buf = []
stream = None
nome_crianca = ""


def iniciar_stream():
    global stream, buf, nome_crianca
    nome_crianca = mem.get("crianca", {}).get("nome", "")
    buf = []
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            callback=lambda i, f, t, s: buf.append(i.copy()))
    stream.start()


def parar_stream():
    global stream
    if stream:
        stream.stop()
        stream.close()
        stream = None


if __name__ == "__main__":
    iniciar_stream()
    if nome_crianca:
        print(f"Ola {nome_crianca}! A Luiza ja esta aqui!")
    else:
        print("Ola! Eu sou a Luiza! Qual o seu nome?")

    print("ENTER para falar, ENTER de novo para parar")

    while True:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
        input("\nPressione ENTER para falar com a Luiza...")

        with buf_lock:
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

        with buf_lock:
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
