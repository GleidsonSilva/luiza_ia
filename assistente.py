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
import memoria_rag
import contexto
import gerenciador_contexto as ctx_mgr
import debug

SR = 16000
MAX_DURATION = 15
TARGET_PEAK = 0.95
MAX_GAIN = 10.0
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
MAX_HISTORICO_TURNOS = 10
COMPACTION_THRESHOLD = ctx_mgr.COMPACTION_THRESHOLD_TURNS

os.chdir(os.path.dirname(os.path.abspath(__file__)))
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

SYSTEM_PROMPT = """### PERSONA
Voce eh a Luiza, uma amiga de 8 anos.
Voce gosta de conversar e fazer amizades.
NUNCA se comporte como assistente ou robo.
Voce nao tem acesso a sistemas, APIs, ferramentas ou metadados.
Voce eh so uma crianca. Apenas a Luiza.
Nao use emojis ou simbolos. Nenhum emoji. Nunca.

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
- NUNCA mencione nomes de funcoes, arquivos, tokens, JSON, APIs ou qualquer informacao tecnica
- Lembre SEMPRE do nome da crianca durante a conversa. Usar o nome deixa a conversa mais amigavel.

### REGRAS
- No maximo 3 frases curtas por resposta
- Se a crianca estiver triste, acolha com carinho
- Se a crianca pedir ajuda com uma tarefa escolar, tente explicar de forma simples
- Nao repita o que a crianca disse literalmente, apenas responda naturalmente
- Nao mencione operacoes internas, buscas ou processos. Apenas responda como a Luiza
- As vezes a crianca pode digitar sem pontuacao, com frases confusas ou com ruido (letras soltas, erros de digitacao). Ignore o ruido e tente entender o que ela quis dizer. Se nao entender, pergunte de forma carinhosa o que ela quis dizer
- NUNCA use emojis ou caracteres especiais. Nenhum emoji. Nem nos pensamentos.

### SE A CRIANCA PEDIR PARA BRINCAR, VOCE SABE:
- forca: escolha palavras curtas (max 6 letras), diga o numero de letras, nao use nomes proprios
- matematica: use pergunta_matematica para gerar uma conta de soma ou subtracao com numeros de 1 digito. Depois pergunte o resultado. Se a crianca acertar, elogie. Se errar, mostre a resposta correta de forma carinhosa
- adivinhacao: pense em algo simples (animal, objeto, comida) e de dicas uma de cada vez
- 20 perguntas: deixe a crianca fazer perguntas para adivinhar o que voce pensou
- cadeia de palavras: uma pessoa fala uma palavra e a proxima fala outra que comeca com a ultima letra
- o que sou eu: voce pensa em algo e a crianca pergunta se eh animal, objeto, comida, etc.
- verdadeiro ou falso: voce fala uma frase simples e a crianca diz se acha que eh verdade ou mentira
- rima comigo: voce fala uma palavra simples e a crianca tenta achar uma que rima
- categorias/stop: voce escolhe uma categoria (ex: frutas, animais) e a crianca fala palavras que pertencem ate alguem falar stop ou repetir
- complete a frase: voce começa uma frase curta e a crianca completa
- historia coletiva: voce começa uma frase de historia e a crianca continua
- detetive: voce descreve um cenario simples e a crianca descobre o que aconteceu
- palavra secreta: voce escolhe uma palavra curta e a crianca tenta adivinhar com dicas

### MONITORAMENTO INTERNO (NAO MOSTRE PARA A CRIANCA)
- Todas as chamadas de funcao sao invisiveis para a crianca. Continue agindo como a Luiza.
- Identidade eh IMPORTANTE: se a crianca disser o nome, lembre-se e use ele sempre. Se varias criancas usarem, guarde o nome de cada uma separadamente.
- Use lembrar_nome APENAS quando a crianca disser explicitamente o nome dela, ex: "meu nome eh Maria" ou "sou o Pedro"
- Se ja tiver o nome salvo, use o nome da crianca na primeira resposta e sempre que for natural. Isso faz ela se sentir especial.
- Use lembrar_fato para dados como: comida_favorita, brinquedo_favorito, cidade, idade, nome_escola, nome_melhor_amigo, bichinho
- Use buscar_clima quando a crianca perguntar sobre o tempo na cidade dela (se ja souber a cidade)
- Se nao souber a cidade, pergunte primeiro onde ela mora antes de usar buscar_clima
- Use buscar_web quando ela mencionar algo que voce nao conhece (animal, brinquedo, lugar, assunto) e ela pedir informacao
- Nao use buscar_web para assuntos que voce deveria saber naturalmente
- Use calculadora para fazer contas simples quando a crianca pedir ou durante uma brincadeira de matematica
- Use pergunta_matematica para gerar uma conta aleatoria quando brincar de matematica
- Nao faca perguntas de confirmacao desnecessarias se a informacao ja estiver clara

### MEMORIA
- A memoria eh sua melhor amiga. Guarde o nome, brinquedos, comidas, escola e tudo que for importante sobre a crianca.
- Quando a crianca disser o nome, use lembrar_nome IMEDIATAMENTE
- Quando ela contar algo sobre ela, use lembrar_fato para guardar
- Ao aprender algo novo, mostre empolgação e interesse
- Use o nome da crianca SEMPRE que possivel. Isso deixa a conversa mais proxima e carinhosa.

### CONTEXTO
- Data, hora e periodo do dia sao informados automaticamente pra voce
- Se a crianca perguntar sobre o clima, veja se voce sabe a cidade dela
- Se souber a cidade, use a funcao buscar_clima
- Se nao souber, pergunte primeiro onde ela mora
- Se a crianca mencionar um animal, brinquedo, lugar ou assunto que voce nao conhece bem, pesquise com buscar_web
- Se ela falar "vamos falar sobre X" e voce nao souber muito, busque informacoes
- Nao finja saber de algo que nao conhece — pesquise

### HORARIOS E ROTINA
- De manha: pergunte se ja tomou cafe da manha
- Meio dia (por volta de 12h): pergunte se ja almocou
- Tarde: pergunte o que esta fazendo na escola ou brincando
- Noite (21:30 ou mais): pergunte se ja escovou os dentes, se ja esta de pijama e se esta na hora de dormir
- Use essas informacoes para orientar a crianca de forma carinhosa, sem dar ordens"""



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
    },
    {
        "type": "function",
        "function": {
            "name": "calculadora",
            "description": "Faz uma conta simples de soma ou subtracao com 2 ou 3 numeros pequenos",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "Primeiro numero de 1 digito (0 a 9)"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Segundo numero de 1 digito (0 a 9)"
                    },
                    "c": {
                        "type": "integer",
                        "description": "Terceiro numero opcional de 1 digito (0 a 9), use apenas para 3 numeros"
                    },
                    "operacao": {
                        "type": "string",
                        "description": "Operacao a ser feita: soma ou subtracao",
                        "enum": ["soma", "subtracao"]
                    }
                },
                "required": ["a", "b", "operacao"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pergunta_matematica",
            "description": "Gera uma pergunta aleatoria simples de matematica com numeros de 1 digito",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo": {
                        "type": "string",
                        "description": "Tipo de operacao: soma ou subtracao",
                        "enum": ["soma", "subtracao"]
                    },
                    "quantidade": {
                        "type": "integer",
                        "description": "Quantidade de numeros na conta: 2 ou 3",
                        "enum": [2, 3]
                    }
                },
                "required": ["tipo", "quantidade"]
            }
        }
    }
]

mem = memoria.carregar()
_identidade_confirmada = False
historico = []
resumo_sessao = ""
turnos_desde_compactacao = 0
ultimo_ctx_status = {"tokens": 0, "limit": 0, "pct": 0.0, "acao": "ok"}
_turno_idx = 0
memoria_vetorial = memoria_rag.MemoriaVetorial()


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


def compactar_historico(historico_para_resumir: list[dict]) -> str:
    if not historico_para_resumir:
        return ""
    texto = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in historico_para_resumir
    )
    prompt = (
        "Resuma a conversa abaixo em PT-BR, ate 1000 tokens, "
        "mantendo: nome da crianca, fatos sobre ela, "
        "temas e brincadeiras mencionados, e o assunto principal.\n\n"
        + texto
    )
    messages = [
        {"role": "system", "content": "Voce faz resumos curtos e informativos."},
        {"role": "user", "content": prompt},
    ]
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0.3, "max_tokens": 300},
        }, timeout=60)
        body = r.json()
        return body.get("message", {}).get("content", "").strip()
    except Exception as e:
        debug.log(f"compactacao falhou: {e}")
        return ""


def _resolver_tool_output(content: str) -> str:
    if content and content.startswith("memory://"):
        return ctx_mgr.resolver_ponteiro(content)
    return content


def _executar_calculadora(args: dict) -> str:
    try:
        a = int(args.get("a", 0))
        b = int(args.get("b", 0))
        c = args.get("c")
        operacao = args.get("operacao", "soma")

        if not (0 <= a <= 9 and 0 <= b <= 9):
            return "Erro: todos os numeros devem ser de 1 digito (0 a 9)."
        if c is not None:
            c = int(c)
            if not (0 <= c <= 9):
                return "Erro: todos os numeros devem ser de 1 digito (0 a 9)."

        if operacao == "soma":
            total = a + b + (c if c is not None else 0)
            expressao = f"{a} + {b}" + (f" + {c}" if c is not None else "")
        elif operacao == "subtracao":
            total = a - b - (c if c is not None else 0)
            expressao = f"{a} - {b}" + (f" - {c}" if c is not None else "")
        else:
            return "Erro: operacao invalida. Use soma ou subtracao."

        return f"{expressao} = {total}"
    except Exception as e:
        return f"Erro na calculadora: {e}"


def _executar_pergunta_matematica(args: dict) -> str:
    try:
        import random

        operacao = args.get("operacao", args.get("tipo", "soma"))
        quantidade = int(args.get("quantidade", 2))

        if operacao not in ("soma", "subtracao"):
            operacao = "soma"
        if quantidade not in (2, 3):
            quantidade = 2

        a = random.randint(0, 9)
        b = random.randint(0, 9)
        c = random.randint(0, 9) if quantidade == 3 else None

        if operacao == "soma":
            total = a + b + (c if c is not None else 0)
            expressao = f"{a} + {b}" + (f" + {c}" if c is not None else "")
        else:
            total = a - b - (c if c is not None else 0)
            expressao = f"{a} - {b}" + (f" - {c}" if c is not None else "")

        return f"pergunta|{expressao} = ?|resposta|{total}"
    except Exception as e:
        return f"Erro ao gerar pergunta: {e}"


def perguntar(texto: str) -> str:
    global historico, _identidade_confirmada, resumo_sessao, turnos_desde_compactacao, mem, ultimo_ctx_status, _turno_idx, memoria_vetorial

    system_content = SYSTEM_PROMPT + f"\n\n### AGORA\n{contexto.contexto_hoje()}"

    memoria_recuperada = memoria_rag.recuperar_fatos_relevantes(mem, texto, limite=5)
    if memoria_recuperada:
        system_content += f"\n\n### MEMORIA\n{memoria_recuperada}"

    t0 = time.time()
    try:
        recuperado = memoria_vetorial.buscar(texto, k_fatos=4, k_turnos=3)
    except Exception:
        recuperado = {"fatos": "", "turnos": ""}
    debug.log(f"memoria_vetorial.buscar em {time.time()-t0:.2f}s")

    if recuperado["fatos"]:
        system_content += f"\n\n### MEMORIA_LONGO_PRAZO\n{recuperado['fatos']}"
    if recuperado["turnos"]:
        system_content += f"\n\n### CONVERSAS_ANTERIORES\n{recuperado['turnos']}"

    if resumo_sessao:
        system_content += f"\n\n### RESUMO_SESSAO\n{resumo_sessao}"

    messages = [{"role": "system", "content": system_content}]

    orcamento = ctx_mgr.calcular_orcamento(MODEL)
    historico_janela = ctx_mgr.aplicar_janela_deslizante(
        historico, limite_tokens=orcamento["history_recent"]
    )
    messages.extend(historico_janela)

    messages.append({"role": "user", "content": texto})

    check = ctx_mgr.verificar_orcamento(messages, MODEL)
    ultimo_ctx_status = check
    debug.log(f"preflight: tokens={check['total_tokens']}, limit={check['context_limit']}, pct={check['percentual']:.0%}, acao={check['acao']}")

    if check["acao"] == "erro":
        return "Desculpe, a conversa ficou muito longa. Vamos comecar de novo?"
    if check["acao"] == "compactar":
        if turnos_desde_compactacao >= COMPACTION_THRESHOLD:
            resumo = compactar_historico(historico)
            if resumo:
                resumo_sessao = resumo
                turnos_desde_compactacao = 0
                messages = [{"role": "system", "content": system_content}]
                messages.extend(historico_janela)
                messages.append({"role": "user", "content": texto})
                debug.log("compactacao aplicada")

    resposta = ""
    needs_rerun = False
    max_iters = 3
    iters = 0

    while iters < max_iters:
        iters += 1
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "tools": tools, "stream": False
        }, timeout=60)
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
                _identidade_confirmada = True
                memoria.adicionar_fato(mem, "nome", args["nome"])
                memoria_vetorial.indexar_fato("nome", args["nome"])
                result = f"Nome salvo: {args['nome']}"
            elif name == "lembrar_fato" and args.get("chave") and args.get("valor"):
                memoria.adicionar_fato(mem, args["chave"], args["valor"])
                memoria_vetorial.indexar_fato(args["chave"], args["valor"])
                result = f"{args['chave']} salvo: {args['valor']}"
            elif name == "buscar_clima" and args.get("cidade"):
                result = contexto.buscar_clima(args["cidade"])
                needs_rerun = True
            elif name == "buscar_web" and args.get("consulta"):
                result = contexto.buscar_web(args["consulta"])
                compressed = ctx_mgr.comprimir_tool_output("buscar_web", result)
                pointer = ctx_mgr.armazenar_ponteiro(result)
                result = pointer + "\n\n" + compressed
                needs_rerun = True
            elif name == "calculadora":
                result = _executar_calculadora(args)
            elif name == "pergunta_matematica":
                result = _executar_pergunta_matematica(args)
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

    try:
        memoria_vetorial.indexar_turno("user", texto, turno_idx=_turno_idx)
        _turno_idx += 1
        memoria_vetorial.indexar_turno("assistant", resposta, turno_idx=_turno_idx)
        _turno_idx += 1
    except Exception:
        pass

    max_hist = MAX_HISTORICO_TURNOS * 2
    if len(historico) > max_hist:
        historico = historico[-max_hist:]

    turnos_desde_compactacao += 1

    return resposta


def falar(texto: str):
    global stream
    was_streaming = stream is not None
    if stream:
        parar_stream()

    try:
        samples, sr = kokoro.create(texto, voice="pf_dora", speed=1.1, lang="pt-br")
    except Exception as e:
        debug.log(f"falar: kokoro.create falhou: {e}")
        if was_streaming:
            try:
                iniciar_stream()
                debug.log("falar: stream reiniciado")
            except Exception as e:
                debug.log(f"falar: erro ao reiniciar stream: {e}")
        return

    sd.stop()
    debug.log(f"falar: iniciando playback, samples={len(samples)}")
    t0 = time.time()

    blocksize = 1024
    idx = 0
    playback_done = threading.Event()

    def _out_callback(outdata, frames, time_info, status):
        nonlocal idx
        if status.output_underflow:
            debug.log("falar: output underflow")
        end = idx + frames
        chunk = samples[idx:end]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            playback_done.set()
            raise sd.CallbackStop()
        outdata[:, 0] = chunk
        idx = end

    try:
        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                             callback=_out_callback, blocksize=blocksize,
                             latency='low') as out_stream:
            out_stream.start()
            if not playback_done.wait(timeout=60.0):
                debug.log("falar: timeout playback")
                out_stream.abort()
    except Exception as e:
        debug.log(f"falar error: {e}")
    finally:
        debug.log(f"falar: playback finalizado em {time.time()-t0:.2f}s")
        if was_streaming:
            try:
                iniciar_stream()
                debug.log("falar: stream reiniciado")
            except Exception as e:
                debug.log(f"falar: erro ao reiniciar stream: {e}")

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
    _identidade_confirmada = False
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
