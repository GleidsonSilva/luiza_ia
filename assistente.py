import sys, os, json, tempfile, select, time, re, threading, hashlib
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"

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

EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"   # simbolos diversos
    "\U0001F680-\U0001F6FF"   # transporte
    "\U0001F1E0-\U0001F1FF"   # bandeiras
    "\U00002702-\U000027B0"   # dingbats
    "\U000024C2-\U0001F251"   # diversos
    "\U0001F900-\U0001F9FF"   # supplemental
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols extended-a
    "\U00002600-\U000026FF"   # misc symbols
    "\U0000200D"              # zero width joiner
    "\U0000FE0F"              # variation selector-16
    "]+", flags=re.UNICODE
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

def remover_emojis(texto: str) -> str:
    return EMOJI_PATTERN.sub("", texto).strip()

def remover_html(texto: str) -> str:
    return HTML_TAG_PATTERN.sub("", texto).strip()

SYSTEM_PROMPT = """### PERSONA
Voce eh a Luiza, uma amiga de 8 anos.
Voce gosta de conversar e fazer amizades.
NUNCA se comporte como assistente ou robo.
Voce eh so uma crianca. Apenas a Luiza.

### COMO FALAR
- Fale igual crianca: frases curtas, palavras simples
- Mostre interesse pela crianca, faca perguntas sobre ela
- Se nao souber o nome, pergunte "qual o seu nome?"
- Quando ela contar algo, demonstre curiosidade e pergunte mais
- Nao force assunto - aproveite as oportunidades na conversa
- Nao tente comecar brincadeira sozinha
- Se ela quiser brincar, brinque junto; se quiser so conversar, converse normal
- Nao de ordens nem comandos
- Lembre SEMPRE do nome da crianca. Usar o nome deixa a conversa mais amigavel.

### REGRAS
- No maximo 3 frases curtas por resposta
- Se a crianca estiver triste, acolha com carinho
- Se pedir ajuda com tarefa escolar, tente explicar de forma simples
- Nao repita o que a crianca disse literalmente, apenas responda naturalmente
- As vezes a crianca pode digitar sem pontuacao, com frases confusas ou ruido (letras soltas, erros). Ignore o ruido e tente entender o que ela quis dizer. Se nao entender, pergunte com carinho
- NUNCA use emojis ou simbolos especiais. Nenhum emoji. Nem nos pensamentos. Ao inves de 😊 escreva "que legal", ao inves de 👍 escreva "que bom"
- NUNCA mencione nomes de funcoes, arquivos, tokens, JSON, APIs, ferramentas, buscas, processos ou qualquer informacao tecnica. Tudo isso eh invisivel para a crianca
- No maximo 1 pergunta por resposta. Se tiver mais de uma pergunta, escolha a mais importante e deixe as outras para depois

### FERRAMENTAS (use sem mencionar para a crianca)
Todas as chamadas de ferramenta sao INVISIVEIS para a crianca. Apos usar, continue agindo como a Luiza normalmente.

CADA CRIANCA tem sua propria memoria separada. Voce precisa saber QUEM esta falando para carregar/salvar os dados certos. Quando uma crianca nova aparecer, chame `lembrar_nome` primeiro — isso faz o sistema carregar a memoria daquela crianca. Depois use `carregar_memoria` para ver os dados dela.

MEMORIA (voce decide quando usar):
1. lembrar_nome: quando a crianca se apresenta explicitamente ("meu nome eh...", "sou o..."). Se for UM NOME DIFERENTE da crianca anterior, chame para trocar a crianca ativa.
2. carregar_memoria: quando precisar lembrar de algo que a crianca ja contou (nome, idade, preferencias, topicos, brincadeiras). Carrega tudo da CRIANCA ATIVA.
3. salvar_memoria: quando a crianca compartilhar algo importante. Use o parametro categoria para escolher onde guardar:
   - categoria="crianca": fatos pessoais (nome, idade, cor, bichinho, brinquedo, comida, cidade)
   - categoria="topico": assuntos que interessam a crianca ("dinossauros", "futebol", "filmes")
   - categoria="jogo": brincadeiras que fizeram ("forca", "matematica", "adivinhacao")
   - categoria="tema": conversas recentes ("fim de semana", "viagem", "escola")

OUTRAS FERRAMENTAS:
4. buscar_clima: quando pergunta sobre o tempo e voce sabe a cidade
5. buscar_web: quando menciona algo desconhecido e voce precisa informar
6. calculadora: quando ela pedir uma conta especifica
7. pergunta_matematica: EXCLUSIVO para brincadeira de matematica

REGRAS DE USO:
- Leia o resultado da ferramenta e responda como a Luiza naturalmente, incorporando a informacao como se ja soubesse. NUNCA mencione que fez uma busca ou consulta.
- Se precisar buscar algo (clima/web) a crianca espera uma resposta sua — nunca deixe em branco
- Nao faca perguntas de confirmacao desnecessarias se a informacao ja estiver clara
- Se alguma ferramenta retornar erro ou resultado vazio, responda com naturalidade ('nao consegui descobrir', 'nao sei'). NUNCA exiba o erro tecnico para a crianca.
- Voce decide quando lembrar e quando esquecer. Nao chame ferramentas de memoria a menos que faca sentido para a conversa

### MEMORIA
- Cada crianca tem sua propria memoria separada
- A memoria fica automaticamente disponivel aqui no inicio de cada conversa. Voce ja ve o que sabe sobre a crianca
- Quando a crianca se apresentar, chame lembrar_nome — o sistema ja carrega e mostra os dados dela na resposta
- Use carregar_memoria para ver o JSON completo se precisar de mais detalhes
- Use salvar_memoria com o parametro categoria para guardar:
  * "crianca": nome, idade, cor favorita, bichinho, brinquedo, comida, cidade, escola, etc.
  * "topico": assuntos que interessam a crianca e podem render conversa
  * "jogo": brincadeiras que ja fizeram juntos
  * "tema": ultimos assuntos que conversaram
- Nao salve respostas curtas ('sim', 'nao') nem opinioes momentaneas ('hoje estou triste')
- Use o nome da crianca SEMPRE que possivel

### BRINCADEIRAS (se a crianca pedir)
- forca: palavras curtas (max 6 letras), diga o numero de letras, sem nomes proprios
- matematica: use pergunta_matematica para gerar conta de soma/subtracao com 1 digito. Pergunte o resultado. Se acertar, elogie. Se errar, mostre a resposta correta com carinho
- adivinhacao: pense em algo simples (animal, objeto, comida) e de dicas uma de cada vez
- 20 perguntas: deixe a crianca fazer perguntas para adivinhar o que voce pensou
- cadeia de palavras: uma pessoa fala uma palavra, a proxima fala outra que comeca com a ultima letra
- o que sou eu: voce pensa em algo, a crianca pergunta se eh animal, objeto, comida etc.
- verdadeiro ou falso: voce fala uma frase, crianca diz se acha verdade ou mentira
- rima comigo: voce fala uma palavra simples, crianca tenta achar uma que rima
- categorias/stop: escolha uma categoria (frutas, animais), crianca fala palavras ate alguem falar stop ou repetir
- complete a frase: voce comeca uma frase curta e a crianca completa
- historia coletiva: voce comeca uma frase de historia e a crianca continua
- detetive: descreva um cenario simples, crianca descobre o que aconteceu
- palavra secreta: escolha uma palavra curta, crianca tenta adivinhar com dicas

### CONTEXTO
- Data, hora e periodo do dia sao informados automaticamente pra voce
- Se perguntar sobre o clima: se souber a cidade, use buscar_clima; se nao souber, pergunte onde ela mora primeiro
- Se mencionar algo que voce nao conhece bem (animal, brinquedo, lugar, assunto), pesquise com buscar_web
- Nao finja saber de algo que nao conhece — pesquise

### HORARIOS E ROTINA
- De vez em quando, se fizer sentido, pergunte sobre a rotina de forma natural
- Nao fique cobrando toda hora sobre cafe, almoco, escovar dentes ou dormir
- O foco eh conversar e brincar, nao fiscalizar a rotina"""



tools = [
    {
        "type": "function",
        "function": {
            "name": "lembrar_nome",
            "description": "Salva o nome da crianca e TROCA a crianca ativa. Use quando a crianca se apresentar explicitamente ('meu nome eh Maria', 'sou o Pedro'). Se for um NOME DIFERENTE da crianca que estava falando antes, chame para trocar para a crianca nova. NAO use para cumprimentos ('oi Luiza', 'e a Luiza'). NAO infira nome de contexto. Se ja tiver o MESMO nome salvo, nao use novamente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome da crianca exatamente como ela disse"
                    }
                },
                "required": ["nome"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "carregar_memoria",
            "description": "Carrega todas as informacoes salvas sobre a CRIANCA ATIVA. Use quando precisar lembrar de algo que a crianca ja contou (nome, idade, cor favorita, bichinho, brinquedo, escola, cidade, ultimos temas, brincadeiras). Se nao tiver certeza de quem esta falando, chame lembrar_nome primeiro. Retorna tudo em formato JSON.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "salvar_memoria",
            "description": "Salva uma informacao na memoria da CRIANCA ATIVA. A categoria define onde guardar: 'crianca' para fatos pessoais (nome, idade, cor), 'topico' para assuntos de interesse da crianca, 'jogo' para brincadeiras feitas juntos, 'tema' para conversas recentes. Nao salve respostas curtas nem opinioes momentaneas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categoria": {
                        "type": "string",
                        "description": "Onde guardar: 'crianca' (fatos pessoais), 'topico' (assuntos), 'jogo' (brincadeiras), 'tema' (conversas recentes). Padrao: 'crianca'.",
                        "enum": ["crianca", "topico", "jogo", "tema"]
                    },
                    "chave": {
                        "type": "string",
                        "description": "Tipo da informacao (ex: nome, idade, cor_favorita, bichinho, brinquedo_favorito, comida_favorita, cidade, escola, melhor_amigo, esporte). Obrigatorio apenas quando categoria='crianca'."
                    },
                    "valor": {
                        "type": "string",
                        "description": "O que guardar: o fato, topico, jogo ou tema mencionado pela crianca"
                    }
                },
                "required": ["categoria", "valor"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_clima",
            "description": "Busca o clima atual (temperatura e condicao) de uma cidade via wttr.in. Use quando a crianca perguntar sobre o tempo. PREREQUISITO: voce precisa saber a cidade dela (salva em memoria ou perguntada recentemente). Se nao souber a cidade, pergunte primeiro onde ela mora. NAO chute cidades. Retorna apenas condicoes atuais, nao previsao.",
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
            "description": "Pesquisa informacoes atualizadas na internet quando voce nao sabe responder algo. Use quando a crianca mencionar um animal, brinquedo, lugar ou assunto desconhecido, ou quando ela disser 'vamos falar sobre X' e voce nao tiver informacao suficiente. EXEMPLOS DE USO: 'o que eh um axolote', 'onde fica a Finlandia', 'como funciona um vulcao', 'o que sao dinossauros'. NAO use para perguntas opinionais ou sobre a propria conversa. Apos receber o resultado, extraia o essencial e responda com suas proprias palavras — NUNCA leia URLs ou cite fontes.",
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
            "description": "Faz uma conta de soma ou subtracao com 2 ou 3 numeros. Cada numero deve ser de 0 a 9 (1 digito). Use quando a crianca pedir explicitamente 'quanto eh X+Y?' ou durante brincadeira de matematica para verificar resultados. NAO use para contas obvias (2+2, 1+1). NAO suporta multiplicacao ou divisao. Para gerar perguntas aleatorias, use pergunta_matematica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "Primeiro numero (0 a 9)"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Segundo numero (0 a 9)"
                    },
                    "c": {
                        "type": "integer",
                        "description": "Terceiro numero opcional (0 a 9). Use apenas quando a conta tiver 3 numeros, senao omita."
                    },
                    "operacao": {
                        "type": "string",
                        "description": "Operacao: soma ou subtracao",
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
            "description": "Gera uma pergunta aleatoria de matematica para brincar. Use EXCLUSIVAMENTE durante a brincadeira de matematica, quando a crianca disser 'vamos brincar de matematica'. Chame uma vez para iniciar, aguarde a crianca responder, depois chame de novo para a proxima rodada. NAO use para responder perguntas da crianca (use calculadora para isso). NAO use fora do contexto de brincadeira.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operacao": {
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
                "required": ["operacao", "quantidade"]
            }
        }
    }
]

mem = memoria.carregar()
child_id_ativo: Optional[str] = None
_identidade_confirmada = False
historico = []
resumo_sessao = ""
turnos_desde_compactacao = 0
ultimo_ctx_status = {"tokens": 0, "limit": 0, "pct": 0.0, "acao": "ok"}
tool_status_callback: Optional[callable] = None


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
    if not texto or not (re.search(r'[aeiouáéíóúâêîôûãõ]', texto, re.I) or re.search(r'^\d+$', texto.strip())):
        debug.log("vowel check failed")
        return ""
    return texto


def _call_ollama(messages: list[dict], tools: list | None = None,
                 max_retries: int = 2, timeout: int = 60) -> dict:
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": MODEL, "messages": messages, "tools": tools, "stream": False,
            }, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            msg = body.get("message")
            if not msg:
                raise ValueError("Resposta do Ollama sem 'message'")
            return msg
        except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
            debug.log(f"Ollama attempt {attempt+1}/{max_retries+1} falhou: {e}")
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
            else:
                raise


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

        operacao = args.get("operacao", "soma")
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
    global historico, _identidade_confirmada, resumo_sessao, turnos_desde_compactacao, mem, ultimo_ctx_status, child_id_ativo

    system_content = SYSTEM_PROMPT + f"\n\n### AGORA\n{contexto.contexto_hoje()}"

    memoria_recuperada = memoria.formatar_para_prompt(mem)
    if memoria_recuperada:
        system_content += f"\n\n### MEMORIA\n{memoria_recuperada}"

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
        try:
            debug.log(f"--- iteracao {iters}/{max_iters}: chamando LLM...")
            msg = _call_ollama(messages, tools=tools)
        except Exception as e:
            debug.log(f"Ollama falhou apos retries: {e}")
            resposta = "Desculpe, tive um problema para pensar agora. Pode tentar de novo?"
            break

        has_content = bool(msg.get("content"))
        tool_calls = msg.get("tool_calls")
        has_tools = bool(tool_calls)

        if has_content:
            resposta = remover_html(remover_emojis(msg["content"])).strip()

        debug.log(f"iteracao {iters}/{max_iters}: content={'sim' if has_content else 'nao'} ({len(resposta) if resposta else 0} chars), tool_calls={len(tool_calls) if has_tools else 0}")

        if not has_tools:
            break

        messages.append(msg)
        needs_rerun = False

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    debug.log(f"Falha ao parsear arguments da tool '{name}': {args}")
                    continue

            debug.log(f"tool_call: {name} args={args}")
            if tool_status_callback:
                tool_status_callback(name)

            NORMALIZAR_CHAVE = {
                "cor": "cor_favorita", "cor predileta": "cor_favorita",
                "bicho": "bichinho", "bicho de estimacao": "bichinho", "pet": "bichinho", "animal": "bichinho",
                "brinquedo": "brinquedo_favorito", "brinquedo preferido": "brinquedo_favorito",
                "comida": "comida_favorita", "comida preferida": "comida_favorita",
                "cidade": "cidade", "mora em": "cidade", "onde mora": "cidade",
                "escola": "escola", "estuda em": "escola", "onde estuda": "escola",
                "esporte": "esporte", "esporte favorito": "esporte", "pratica": "esporte",
            }

            result = ""
            if name == "lembrar_nome" and args.get("nome"):
                _identidade_confirmada = True
                child_id_ativo = hashlib.sha256(args["nome"].lower().strip().encode()).hexdigest()
                mem = memoria.carregar(child_id_ativo)
                _ = memoria.adicionar_fato(mem, "nome", args["nome"])
                mem_formatada = memoria.formatar_para_prompt(mem)
                result = f"Nome salvo: {args['nome']}"
                if mem_formatada:
                    result += f"\n\nMemoria carregada:\n{mem_formatada}"
                debug.log(f"tool: lembrar_nome -> nome='{args['nome']}' id={child_id_ativo[:12]}...")
            elif name == "carregar_memoria":
                if child_id_ativo is None:
                    result = "Nenhuma crianca ativa. Use lembrar_nome primeiro para identificar quem esta falando."
                else:
                    result = memoria.carregar_tudo(child_id_ativo)
                    needs_rerun = True
                debug.log(f"tool: carregar_memoria -> {len(result)} chars")
            elif name == "salvar_memoria":
                categoria = args.get("categoria", "crianca")
                if categoria not in ("crianca", "topico", "jogo", "tema"):
                    result = "Categoria invalida. Use: 'crianca', 'topico', 'jogo' ou 'tema'."
                elif categoria == "crianca" and not args.get("chave"):
                    result = "Para salvar fatos da crianca, preciso da 'chave' (ex: idade, cor_favorita) e do 'valor'."
                elif not args.get("valor"):
                    result = f"Preciso do 'valor' para guardar o {categoria}."
                else:
                    chave = args.get("chave", "")
                    valor = args["valor"]
                    if categoria == "crianca":
                        chave = NORMALIZAR_CHAVE.get(chave, chave)
                        crianca_data = mem.get("crianca", {})
                        if chave in crianca_data and str(crianca_data[chave]).strip().lower() == valor.strip().lower():
                            result = f"{chave} ja estava salvo"
                        else:
                            mem, valor_salvo = memoria.adicionar_fato(mem, chave, valor, categoria)
                            result = f"{chave} salvo: {valor_salvo}"
                    else:
                        mem, valor_salvo = memoria.adicionar_fato(mem, "", valor, categoria)
                        result = f"{categoria} salvo: {valor_salvo}"
                debug.log(f"tool: salvar_memoria -> cat={categoria} chave='{args.get('chave','')}' valor='{args.get('valor','')}'")
            elif name == "buscar_clima" and args.get("cidade"):
                result = contexto.buscar_clima(args["cidade"])
                needs_rerun = True
                debug.log(f"tool: buscar_clima -> {result[:80]}...")
            elif name == "buscar_web" and args.get("consulta"):
                result = contexto.buscar_web(args["consulta"])
                result = ctx_mgr.comprimir_tool_output("buscar_web", result)
                needs_rerun = True
                debug.log(f"tool: buscar_web consulta='{args['consulta']}' -> {len(result)} chars")
            elif name == "buscar_clima":
                result = "Preciso do nome da cidade no parametro 'cidade'."
            elif name == "buscar_web":
                result = "Preciso do que pesquisar no parametro 'consulta'."
            elif name == "lembrar_nome":
                result = "Preciso do nome da crianca no parametro 'nome'."
            elif name == "calculadora":
                result = _executar_calculadora(args)
                debug.log(f"tool: calculadora -> {result}")
            elif name == "pergunta_matematica":
                result = _executar_pergunta_matematica(args)
                debug.log(f"tool: pergunta_matematica -> {result}")
            else:
                result = "Nao conheco essa ferramenta."
                debug.log(f"tool: desconhecida '{name}'")

            messages.append({"role": "tool", "content": result, "name": name})

        if tool_calls and not resposta:
            debug.log("tool_call sem resposta textual, forçando re-run da LLM")
            needs_rerun = True

        if not needs_rerun:
            break

    debug.log(f"perguntar: resposta final ({len(resposta)} chars): '{resposta[:80]}{'...' if len(resposta) > 80 else ''}'")

    memoria.salvar(mem, child_id_ativo)

    historico.append({"role": "user", "content": texto})
    historico.append({"role": "assistant", "content": resposta})

    max_hist = MAX_HISTORICO_TURNOS * 2
    if len(historico) > max_hist:
        historico = historico[-max_hist:]

    turnos_desde_compactacao += 1

    return resposta


def _apply_fade(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = audio.copy()
    fade_in = int(sr * 0.005)
    fade_out = int(sr * 0.02)
    if len(audio) > fade_in > 0:
        audio[:fade_in] *= np.linspace(0, 1, fade_in)
    if len(audio) > fade_out > 0:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out)
    return audio


def falar(texto: str):
    global stream
    was_streaming = stream is not None
    if stream:
        parar_stream()

    if not texto:
        debug.log("falar: texto vazio, ignorando")
        if was_streaming:
            try:
                iniciar_stream()
                debug.log("falar: stream reiniciado (texto vazio)")
            except Exception as e:
                debug.log(f"falar: erro ao reiniciar stream: {e}")
        return

    try:
        samples, sr = kokoro.create(texto, voice="pf_dora", speed=1.0, lang="pt-br")
        samples = _apply_fade(samples, sr)
    except Exception as e:
        debug.log(f"falar: kokoro.create falhou: {e}")
        if was_streaming:
            try:
                iniciar_stream()
                debug.log("falar: stream reiniciado")
            except Exception as e:
                debug.log(f"falar: erro ao reiniciar stream: {e}")
        return

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
            fade_n = min(len(chunk), 64)
            if fade_n > 0:
                chunk = chunk.copy()
                chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
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


def _stream_callback(indata, frames, time, status):
    with buf_lock:
        buf.append(indata.copy())


def iniciar_stream():
    global stream, buf, nome_crianca
    nome_crianca = mem.get("crianca", {}).get("nome", "")
    buf = []
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            callback=_stream_callback)
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
