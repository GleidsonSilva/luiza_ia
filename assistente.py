import sys, os, json, select, time, re, threading, hashlib, random
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
import mlx_whisper
import requests
from kokoro_onnx import Kokoro
import memoria
import contexto
import gerenciador_contexto as ctx_mgr
import historico
import debug
import reconhecimento_voz

SR = 16000
MAX_DURATION = 15
TARGET_PEAK = 0.95
MAX_GAIN = 10.0
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = os.environ.get("LUZIA_MODEL", "gemma4:26b")
KEEP_ALIVE = os.environ.get("LUZIA_KEEP_ALIVE", "30m")

# Roteador de intencao (decision model Clef) que decide se o turno precisa de
# raciocinio (think=true) ou e conversa simples (think=false, bem mais rapido).
ROUTER_URL = "http://localhost:11434/v1/systemone"
ROUTER_MODEL = os.environ.get("LUZIA_ROUTER_MODEL", "clef-flash")
ROUTER_ENABLED = os.environ.get("LUZIA_ROUTER", "0").lower() not in ("0", "false", "no")
THINK_PADRAO = os.environ.get("LUZIA_THINK", "false").lower() not in ("0", "false", "no")

# Categorias que realmente precisam de raciocinio (decidir/criar). As demais
# (conversa, memoria, conta) respondem direto, sem thinking -> bem mais rapido.
_CATEGORIAS_COM_THINK = {"brincadeira", "clima_web"}

_CATEGORIAS_QUESTAO = {
    "acao": {
        "type": "choice",
        "instructions": (
            "CONTEXTO: A Luiza e uma amiga de 8 anos que conversa com criancas em portugues do Brasil. "
            "Ela tem estes recursos: (1) MEMORIA - salvar, lembrar ou esquecer fatos da crianca "
            "(nome, idade, cor, bichinho, brinquedo, comida, cidade, escola, amigo); (2) BRINCADEIRAS - "
            "sortear e conduzir jogos (forca, adivinhacao, 20 perguntas, rimas, etc.); (3) CALCULADORA - "
            "contas de somar/subtrair; (4) CLIMA - tempo de uma cidade; (5) WEB - pesquisar fatos que "
            "ela nao sabe. Quando houver 'ultima_fala_luiza', ela situa a conversa (a crianca pode estar "
            "respondendo a ela). Classifique a fala da crianca em UMA categoria, pensando em qual recurso "
            "da Luiza deve entrar em acao."
        ),
        "criteria": {
            "conversa": "Bate-papo simples, saudacao, desabafo, opiniao ou pergunta casual que a Luiza responde apenas conversando, sem usar nenhum recurso acima.",
            "brincadeira": "A crianca quer brincar ou jogar, pede uma brincadeira ou aceita uma proposta de brincar.",
            "memoria": "A crianca diz, pede para lembrar ou pede para esquecer dados sobre si mesma (nome, idade, gostos, bichinho, amigo, escola).",
            "conta": "A crianca pede um calculo ou envia uma conta de somar/subtrair.",
            "clima_web": "A crianca pergunta o clima ou um fato/assunto que exige buscar informacao na internet.",
        },
    }
}
MAX_HISTORICO_TURNOS = 10

# mlx_whisper/Metal nao suporta inferencias concorrentes no mesmo modelo.
# Serializa o uso entre a transcricao principal e o detector de wake word.
whisper_lock = threading.Lock()
MIN_RMS_ENROLLMENT = 0.01
COMPACTION_THRESHOLD = ctx_mgr.COMPACTION_THRESHOLD_TURNS

# Buffer de audio: limitado a ~60s para evitar consumo infinito de memoria
BUF_MAX_CHUNKS = 2880  # ~180s de audio; reduz poda agressiva e log spam
ULTIMO_CHILD_PATH = os.path.join("memorias", "ultimo_child.json")

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
1. lembrar_nome: quando a crianca se apresenta explicitamente ("meu nome eh...", "sou o..."). Se for UM NOME DIFERENTE da crianca anterior, chame para trocar a crianca ativa. Se a crianca CORRIGIR o proprio nome (ex: "nao eh Ilona, eh Eloa"), use lembrar_nome com o nome correto — o sistema atualiza a identidade existente automaticamente.
2. carregar_memoria: quando precisar lembrar de algo que a crianca ja contou (nome, idade, preferencias, topicos, brincadeiras). Carrega tudo da CRIANCA ATIVA (use para ver o JSON completo).
3. salvar_memoria: quando a crianca compartilhar algo importante. Use o parametro categoria para escolher onde guardar:
   - categoria="crianca": fatos pessoais (nome, idade, cor, bichinho, brinquedo, comida, cidade)
   - categoria="topico": assuntos que interessam a crianca ("dinossauros", "futebol", "filmes")
   - categoria="jogo": brincadeiras que fizeram ("forca", "matematica", "adivinhacao")
   - categoria="tema": conversas recentes ("fim de semana", "viagem", "escola")
   Nao salve respostas curtas ('sim', 'nao') nem opinioes momentaneas ('hoje estou triste').

OUTRAS FERRAMENTAS:
4. buscar_clima: quando pergunta sobre o tempo e voce sabe a cidade
5. buscar_web: quando menciona algo desconhecido e voce precisa informar
6. calculadora: quando ela pedir uma conta especifica
7. pergunta_matematica: EXCLUSIVO para brincadeira de matematica
8. buscar_historico: quando voce quiser lembrar de conversas antigas com a crianca. Parametros: query (palavra-chave), child_id (hex da crianca, opcional), ultimas_n_sessoes (quantas sessoes recentes ver, default 5), max_results (default 50). Retorna mensagens do historico local. Use para lembrar de detalhes, temas ou brincadeiras antigas.
9. esquecer_memoria: EXCLUSIVO quando a crianca PEDIR para esquecer algo ('esquece isso', 'apaga'). Nao use por conta propria. Parametros: chave (o que apagar, ex: idade, cor_favorita) e categoria (onde buscar).

REGRAS DE USO:
- Leia o resultado da ferramenta e responda como a Luiza naturalmente, incorporando a informacao como se ja soubesse. NUNCA mencione que fez uma busca ou consulta.
- Se precisar buscar algo (clima/web) a crianca espera uma resposta sua — nunca deixe em branco
- Nao faca perguntas de confirmacao desnecessarias se a informacao ja estiver clara
- Se alguma ferramenta retornar erro ou resultado vazio, responda com naturalidade ('nao consegui descobrir', 'nao sei'). NUNCA exiba o erro tecnico para a crianca.
- Voce decide quando lembrar e quando esquecer. Nao chame ferramentas de memoria a menos que faca sentido para a conversa

### IDENTIFICACAO AUTOMATICA POR VOZ
- O sistema pode identificar automaticamente qual crianca esta falando pela voz.
- Quando o sistema detectar uma troca de crianca, a memoria da crianca anterior ja foi substituida pela da nova.
- Se voce nao tiver certeza de quem esta falando, confirme de forma natural: " voce eh o [nome da memoria]?"
- Se a crianca confirmar, continue a conversa normalmente.
- Se a crianca negar ou corrigir, chame `lembrar_nome` com o nome correto.
- Se a identificacao por voz estiver falhando muito, voce pode chamar `registrar_voz` para atualizar a assinatura.

### BRINCADEIRAS (se a crianca pedir)
- Quando a crianca quiser brincar, chame `sortear_brincadeira` para sortear uma brincadeira variada (evita repetir as ultimas). Nao escolha voce mesma.
- A ferramenta devolve o nome e uma dica de como conduzir; siga a dica.
- Conduza leve: uma dica/rodada por vez, elogie quando acertar e ajude com carinho quando errar.
- matematica: use `pergunta_matematica`; se acertar elogie, se errar mostre a resposta com carinho.

### CONTEXTO
- Data, hora e periodo do dia sao informados automaticamente pra voce
- Se perguntar sobre o clima: se souber a cidade, use buscar_clima; se nao souber, pergunte onde ela mora primeiro
- Se mencionar algo que voce nao conhece bem (animal, brinquedo, lugar, assunto), pesquise com buscar_web
- Nao finja saber de algo que nao conhece — pesquise

### REGRAS OBRIGATORIAS DE FERRAMENTAS (use SEMPRE, sem excecao)
QUANDO A CRIANCA DIZ -> CHAME ESTA FERRAMENTA IMEDIATAMENTE:
- "vamos brincar", "quero brincar", "brincadeira" -> sortear_brincadeira
- "previsao", "tempo", "clima", "ta frio", "ta quente", "chuva" -> buscar_clima (se nao souber cidade, pergunte primeiro)
- "meu nome eh", "sou o", "sou a", "me chamo" -> lembrar_nome
- conta matematica (ex: "quanto eh", "soma", "multiplica", "divide") -> calculadora
- pergunta de fato/assunto que voce nao sabe -> buscar_web

NAO RESPONDA DIRETAMENTE PARA ESSES CASOS — CHAME A FERRAMENTA.
A resposta so vem DEPOIS do resultado da ferramenta.

### HORARIOS E ROTINA
- De vez em quando, se fizer sentido, pergunte sobre a rotina de forma natural
- Nao fique cobrando toda hora sobre cafe, almoco, escovar dentes ou dormir
- O foco eh conversar e brincar, nao fiscalizar a rotina"""



tools = [
    {
        "type": "function",
        "function": {
            "name": "lembrar_nome",
            "description": "Salva o nome da crianca e TROCA a crianca ativa. Use quando a crianca se apresentar explicitamente ('meu nome eh Maria', 'sou o Pedro'). Se for um NOME DIFERENTE da crianca que estava falando antes, chame para trocar a crianca nova. NAO use para cumprimentos ('oi Luiza', 'e a Luiza'). NAO infira nome de contexto. Se ja tiver o MESMO nome salvo, nao use novamente. Se a crianca CORRIGIR o proprio nome (ex: 'nao eh Ilona, eh Eloa'), use lembrar_nome com o nome correto — o sistema atualiza a identidade existente automaticamente.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_historico",
            "description": "Busca no historico local de conversas por palavra, crianca ou periodo. Retorna ate max_results resultados recentes, filtrando por ultimas_n_sessoes. Use para lembrar do que ja conversou com a crianca. Retorna itens com session_id, timestamp, role e content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Palavra-chave para buscar no texto. Vazio retorna historico recente sem filtro de texto. Case-insensitive."
                    },
                    "child_id": {
                        "type": "string",
                        "description": "Hex da crianca para filtrar. Obrigatorio quando quiser buscar apenas conversas de uma crianca especifica."
                    },
                    "ultimas_n_sessoes": {
                        "type": "integer",
                        "description": "Quantidade de sessoes mais recentes a consultar. Padrao: 5.",
                        "default": 5
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Limite maximo de mensagens retornadas. Padrao: 50.",
                        "default": 50
                    }
                },
                "required": ["ultimas_n_sessoes", "max_results"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "registrar_voz",
            "description": "Registra ou atualiza a assinatura de voz da crianca ativa. Use quando: (1) a crianca pedir explicitamente ('quero registrar minha voz', 'salvar minha voz'), (2) a identificacao por voz falhou ou esta imprecisa, (3) voce suspeitar que a crianca errada esta associada a este nome. Gravamos 5s de audio atual para criar uma nova assinatura.",
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
            "name": "esquecer_memoria",
            "description": "Apaga uma informacao especifica da memoria da crianca ativa. Use quando a crianca pedir explicitamente para esquecer algo que falou ('esquece isso', 'apaga', 'nao quero que lembre'). Nao use para assuntos momentaneos - apenas quando ela pedir para esquecer. Parametros: chave (ex: idade, cor_favorita, nome - opcional, se omitido limpa a categoria inteira), categoria ('crianca', 'topico', 'jogo', 'tema' - opcional, se omitido busca em todas).",
            "parameters": {
                "type": "object",
                "properties": {
                    "chave": {
                        "type": "string",
                        "description": "O que apagar: nome do campo em 'crianca' (ex: idade, cor_favorita, nome). Opcional - se omitido, limpa toda a categoria."
                    },
                    "categoria": {
                        "type": "string",
                        "description": "De onde apagar: 'crianca' (fatos pessoais), 'topico' (assuntos), 'jogo' (brincadeiras), 'tema' (conversas recentes). Opcional - se omitido, busca em todas.",
                        "enum": ["crianca", "topico", "jogo", "tema"]
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sortear_brincadeira",
            "description": "Sorteia uma brincadeira para jogar com a crianca, evitando repetir as ultimas que ja jogaram. Use SEMPRE que a crianca pedir para brincar ou quando for comecar uma brincadeira nova. Retorna o nome da brincadeira e uma dica de como conduzir. O jogo sorteado ja fica registrado automaticamente — nao precisa salvar depois.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def _carregar_ultimo_child() -> tuple[Optional[str], Optional[str]]:
    """Carrega o ultimo child_id e nome salvos do disco."""
    try:
        if os.path.exists(ULTIMO_CHILD_PATH):
            with open(ULTIMO_CHILD_PATH, "r") as f:
                data = json.load(f)
            return data.get("child_id"), data.get("nome")
    except Exception as e:
        debug.log(f"_carregar_ultimo_child: erro: {e}")
    return None, None


def _salvar_ultimo_child(child_id: Optional[str], nome: str = "") -> None:
    """Salva o child_id e nome ativos no disco para recuperacao futura."""
    try:
        os.makedirs(os.path.dirname(ULTIMO_CHILD_PATH), exist_ok=True)
        with open(ULTIMO_CHILD_PATH, "w") as f:
            json.dump({
                "child_id": child_id,
                "nome": nome,
                "atualizado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        debug.log(f"_salvar_ultimo_child: erro: {e}")


def obter_nome_crianca() -> str:
    """Retorna o nome da crianca ativa, ou string vazia."""
    return mem.get("crianca", {}).get("nome", "")




mem = memoria.carregar()
_ultimo_child_id, _ultimo_nome = _carregar_ultimo_child()
if _ultimo_child_id:
    child_id_ativo: Optional[str] = _ultimo_child_id
    mem = memoria.carregar(_ultimo_child_id)
    debug.log(f"assistente: child_id recuperado do cache: {_ultimo_child_id[:12]}... ({_ultimo_nome})")
else:
    child_id_ativo: Optional[str] = None
_identidade_confirmada = False
session_id, historico_sessao = historico.carregar_ativa()
# limita o historico carregado do disco (senao a 1a chamada envia centenas de
# mensagens e o modelo fica lento)
_max_hist_inicial = MAX_HISTORICO_TURNOS * 2
if len(historico_sessao) > _max_hist_inicial:
    historico_sessao = historico_sessao[-_max_hist_inicial:]
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
    if len(audio) < SR * 0.5:
        debug.log(f"too short ({len(audio)} < {SR * 0.5})")
        return ""
    if rms < 0.010:
        debug.log(f"rms muito baixo ({rms:.5f} < 0.010)")
        return ""
    peak = np.max(np.abs(audio))
    if peak > 0:
        gain = min(MAX_GAIN, TARGET_PEAK / peak)
        audio = audio * gain

    with whisper_lock:
        r = mlx_whisper.transcribe(
            audio, language="pt",
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            initial_prompt="ola vamos brincar qual e o seu nome voce gosta de jogar que historia legal"
        )
    texto = r["text"].strip()
    debug.log(f"whisper raw: '{texto}'")
    if not texto or not (re.search(r'[aeiouáéíóúâêîôûãõ]', texto, re.I) or re.search(r'^\d+$', texto.strip())):
        debug.log("vowel check failed")
        return ""
    return texto


def _classificar_acao(texto: str, contexto: str = "") -> str:
    """Usa o Clef (decision model) para rotular a intencao do turno."""
    if not ROUTER_ENABLED:
        return "?"
    estado = {"fala_crianca": texto}
    if contexto:
        estado["ultima_fala_luiza"] = contexto
    try:
        r = requests.post(ROUTER_URL, json={
            "model": ROUTER_MODEL,
            "state": estado,
            "questions": _CATEGORIAS_QUESTAO,
        }, timeout=10)
        r.raise_for_status()
        return (r.json().get("answers", {}).get("acao", {}) or {}).get("choice", "?")
    except Exception as e:
        debug.log(f"roteador: falha na classificacao: {e}")
        return "?"


def _call_ollama(messages: list[dict], tools: list | None = None,
                 max_retries: int = 2, timeout: int = 60, think: bool | None = None) -> dict:
    for attempt in range(max_retries + 1):
        try:
            payload = {
                "model": MODEL, "messages": messages, "tools": tools, "stream": False,
                "keep_alive": KEEP_ALIVE,
            }
            if think is not None:
                payload["think"] = think
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
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


def aquecer_modelo():
    """Carrega o modelo e compila os grafos (think on/off) em background."""
    def _warm_llm(messages, think):
        try:
            requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "messages": messages,
                "tools": tools, "stream": False, "keep_alive": KEEP_ALIVE,
                "think": think,
            }, timeout=300)
            debug.log(f"aquecer_modelo: llm pronto (think={think}, msgs={len(messages)})")
        except Exception as e:
            debug.log(f"aquecer_modelo: llm think={think} falhou: {e}")

    def _warm():
        sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 1) saudação curta (resposta rápida)
        _warm_llm(sys_msg + [{"role": "user", "content": "oi"}], THINK_PADRAO)
        # 2) chamada de tool (buscar_clima) - exercita tool calling
        _warm_llm(sys_msg + [{"role": "user", "content": "qual a previsao em Sao Paulo?"}], THINK_PADRAO)
        # 3) resposta mais longa (conversa)
        _warm_llm(sys_msg + [{"role": "user", "content": "conte uma historia curta sobre um gato"}], THINK_PADRAO)
        if ROUTER_ENABLED and THINK_PADRAO:
            _warm_llm(sys_msg + [{"role": "user", "content": "oi"}], False)
            _warm_llm(sys_msg + [{"role": "user", "content": "oi"}], False)
        if ROUTER_ENABLED:
            try:
                requests.post(ROUTER_URL, json={
                    "model": ROUTER_MODEL, "state": {"fala_crianca": "oi"},
                    "questions": _CATEGORIAS_QUESTAO,
                }, timeout=300)
                debug.log("aquecer_modelo: roteador carregado")
            except Exception as e:
                debug.log(f"aquecer_modelo: roteador falhou: {e}")
    threading.Thread(target=_warm, daemon=True).start()


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
            "keep_alive": KEEP_ALIVE,
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


BRINCADEIRAS = [
    {"nome": "forca", "dica": "uma palavra curta (max 6 letras, sem nomes proprios); diga o numero de letras"},
    {"nome": "adivinhacao", "dica": "pense em algo simples (animal, objeto ou comida) e de uma dica por vez"},
    {"nome": "20 perguntas", "dica": "a crianca faz perguntas para adivinhar o que voce pensou"},
    {"nome": "cadeia de palavras", "dica": "cada um fala uma palavra que comeca com a ultima letra da anterior"},
    {"nome": "rima comigo", "dica": "voce fala uma palavra simples e a crianca tenta achar uma que rima"},
    {"nome": "complete a frase", "dica": "comece uma frase curta e a crianca continua"},
    {"nome": "historia coletiva", "dica": "comece uma frase de historia e a crianca continua"},
    {"nome": "detetive", "dica": "descreva um cenario simples e a crianca descobre o que aconteceu"},
    {"nome": "palavra secreta", "dica": "escolha uma palavra curta e de dicas ate a crianca acertar"},
    {"nome": "categorias (stop)", "dica": "escolha uma categoria e digam palavras ate alguem errar ou repetir"},
    {"nome": "verdadeiro ou falso", "dica": "diga uma frase e a crianca diz se acha verdade ou mentira"},
    {"nome": "numeros em ingles", "dica": "brinque com os numeros de 1 a 10 em ingles e ajude com carinho"},
    {"nome": "o que sou eu", "dica": "voce pensa em algo e a crianca pergunta se e animal, objeto ou comida"},
    {"nome": "eu vejo algo", "dica": "diga 'eu vejo algo que e...' e de dicas de cor/formato ate acertar"},
    {"nome": "pedra papel tesoura", "dica": "joguem rodadas de pedra, papel e tesoura"},
    {"nome": "quiz de animais", "dica": "faca perguntas simples sobre animais (onde vive, o que come)"},
    {"nome": "historias malucas", "dica": "cada um acrescenta uma frase absurda na historia"},
    {"nome": "matematica", "dica": "use pergunta_matematica para gerar a conta e conduza a brincadeira"},
]

JANELA_NAO_REPETIR = 4


def _sortear_brincadeira(mem: dict) -> str:
    recentes = [str(x).strip().lower() for x in mem.get("historico_jogos", [])]
    ultimas = recentes[-JANELA_NAO_REPETIR:]
    disponiveis = [b for b in BRINCADEIRAS if b["nome"].lower() not in ultimas]
    if not disponiveis:
        ultima = recentes[-1] if recentes else ""
        disponiveis = [b for b in BRINCADEIRAS if b["nome"].lower() != ultima] or BRINCADEIRAS
    escolhida = random.choice(disponiveis)
    memoria.adicionar_fato(mem, "", escolhida["nome"], "jogo")
    return (
        f"Brincadeira sorteada: {escolhida['nome']}. "
        f"Como conduzir: {escolhida['dica']}. "
        f"Comece a brincadeira agora, naturalmente, sem mencionar o sorteio."
    )


def perguntar(texto: str, audio: Optional[np.ndarray] = None, voice_match_id: Optional[str] = None, voice_match_name: str = "") -> str:
    global historico_sessao, _identidade_confirmada, resumo_sessao, turnos_desde_compactacao, mem, ultimo_ctx_status, child_id_ativo

    # O system fica ESTAVEL entre turnos (para o cache de prefixo do Ollama).
    # O contexto volatil (data/hora) vai na mensagem do usuario.
    system_content = SYSTEM_PROMPT
    texto_llm = f"{texto}\n\n### CONTEXTO_ATUAL\n{contexto.contexto_hoje()}"

    # Se a voz parece ser de outra crianca, carrega a memoria dela para contexto
    # mas NAO troca a crianca ativa — a LLM precisa confirmar primeiro
    if voice_match_id and voice_match_name:
        mem_match = memoria.carregar(voice_match_id)
        memoria_match_str = memoria.formatar_para_prompt(mem_match)
        if memoria_match_str:
            system_content += (
                f"\n\n### ATENCAO - CONFIRMACAO DE IDENTIDADE\n"
                f"A voz capturada PARECE ser de {voice_match_name}. "
                f"Isso pode ser a mesma crianca de antes ou uma diferente.\n"
                f"Memoria suspeita:\n{memoria_match_str}\n\n"
                f"REGRAS:\n"
                f"1. ANTES de responder qualquer pergunta da crianca, CONFIRME a identidade perguntando: "
                f"'Você é o/a {voice_match_name}?'\n"
                f"2. Se a crianca CONFIRMAR ('sim', 'sou eu'): use lembrar_nome com {voice_match_name} "
                f"para carregar os dados certos\n"
                f"3. Se a crianca NEGAR ou der outro nome: pergunte qual o nome dela e use lembrar_nome "
                f"com o nome correto\n"
                f"4. NUNCA trate a crianca como {voice_match_name} sem confirmacao explicita\n"
                f"\nEsta regra de confirmacao eh obrigatoria. "
                f"Considere tambem chamar registrar_voz se a identificacao por voz parecer util."
            )

    memoria_recuperada = memoria.formatar_para_prompt(mem)
    if memoria_recuperada:
        system_content += f"\n\n### MEMORIA\n{memoria_recuperada}"

    if resumo_sessao:
        system_content += f"\n\n### RESUMO_SESSAO\n{resumo_sessao}"

    # regras de ferramenta como system message separado (perto do fim do contexto,
    # pra nao ser "esquecido" pelo modelo conforme o historico cresce)
    tool_rules = """### REGRAS OBRIGATORIAS DE FERRAMENTAS (use SEMPRE)
QUANDO A CRIANCA DIZ -> CHAME ESTA FERRAMENTA IMEDIATAMENTE:
- "vamos brincar", "quero brincar", "brincadeira", "vamos jogar", "jogar", "forca", "jogo", "brincar de" -> sortear_brincadeira
- "previsao", "tempo", "clima", "ta frio", "ta quente", "chuva", "frio", "quente", "graus" -> buscar_clima (se nao souber cidade, pergunte primeiro)
- "meu nome eh", "sou o", "sou a", "me chamo", "meu nome" -> lembrar_nome
- conta matematica (ex: "quanto eh", "soma", "mais", "menos", "vezes", "multiplica", "divide") -> calculadora
- pergunta de fato/assunto que voce nao sabe -> buscar_web
NAO RESPONDA DIRETAMENTE PARA ESSES CASOS — CHAME A FERRAMENTA."""

    messages = [{"role": "system", "content": system_content}]

    orcamento = ctx_mgr.calcular_orcamento(MODEL)
    historico_janela = ctx_mgr.aplicar_janela_deslizante(
        historico_sessao, limite_tokens=orcamento["history_recent"]
    )
    messages.extend(historico_janela)

    # injeta regras de ferramenta antes da mensagem do usuario
    messages.append({"role": "system", "content": tool_rules})

    messages.append({"role": "user", "content": texto_llm})

    check = ctx_mgr.verificar_orcamento(messages, MODEL)
    ultimo_ctx_status = check
    debug.log(f"preflight: tokens={check['total_tokens']}, limit={check['context_limit']}, pct={check['percentual']:.0%}, acao={check['acao']}")

    if check["acao"] == "erro":
        return "Desculpe, a conversa ficou muito longa. Vamos comecar de novo?"
    if check["acao"] == "compactar":
        if turnos_desde_compactacao >= COMPACTION_THRESHOLD:
            resumo = compactar_historico(historico_sessao)
            if resumo:
                resumo_sessao = resumo
                turnos_desde_compactacao = 0
                messages = [{"role": "system", "content": system_content}]
                messages.extend(historico_janela)
                messages.append({"role": "system", "content": tool_rules})
                messages.append({"role": "user", "content": texto_llm})
                debug.log("compactacao aplicada")

    resposta = ""
    needs_rerun = False
    max_iters = 3
    iters = 0

    ultima_luiza = ""
    for _m in reversed(historico_sessao):
        if _m.get("role") == "assistant":
            ultima_luiza = _m.get("content", "")
            break
    categoria = _classificar_acao(texto, ultima_luiza)
    usar_think = THINK_PADRAO and (categoria in _CATEGORIAS_COM_THINK or categoria == "?")
    debug.log(f"roteador: '{texto[:40]}' -> categoria={categoria} think={usar_think}")

    while iters < max_iters:
        iters += 1
        try:
            debug.log(f"--- iteracao {iters}/{max_iters}: chamando LLM...")
            msg = _call_ollama(messages, tools=tools, think=usar_think)
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
                novo_nome = args["nome"].strip()
                nome_salvo = mem.get("crianca", {}).get("nome", "")

                if child_id_ativo and nome_salvo and nome_salvo.lower().strip() != novo_nome.lower().strip():
                    mem = memoria.carregar(child_id_ativo)
                    mem["crianca"]["nome"] = novo_nome
                    result = f"Nome atualizado para: {novo_nome}"
                else:
                    child_id_ativo = hashlib.sha256(novo_nome.lower().strip().encode()).hexdigest()
                    mem = memoria.carregar(child_id_ativo)
                    memoria.adicionar_fato(mem, "nome", novo_nome)
                    result = f"Nome salvo: {novo_nome}"

                mem_formatada = memoria.formatar_para_prompt(mem)
                if mem_formatada:
                    result += f"\n\nMemoria carregada:\n{mem_formatada}"
                debug.log(f"tool: lembrar_nome -> nome='{novo_nome}' id={child_id_ativo[:12]}...")

                try:
                    audio_para_assinatura = None
                    if audio is not None and len(audio) > 0.5 * SR:
                        audio_para_assinatura = audio
                    else:
                        import assistente as _assistente_mod
                        with _assistente_mod.buf_lock:
                            fim = len(_assistente_mod.buf)
                            inicio = max(0, fim - int(5 * SR))
                            chunks = _assistente_mod.buf[inicio:fim]
                        if chunks:
                            audio_para_assinatura = np.concatenate(chunks)
                    if audio_para_assinatura is not None:
                        rms = float(np.sqrt(np.mean(audio_para_assinatura**2)))
                        baixa_confianca = len(audio_para_assinatura) < int(3 * SR)
                        if rms > MIN_RMS_ENROLLMENT:
                            reconhecimento_voz.salvar_assinatura(
                                child_id_ativo, novo_nome, audio_para_assinatura,
                                baixa_confianca=baixa_confianca
                            )
                            if baixa_confianca:
                                result += "\n\n(Assinatura de voz capturada, mas com pouca qualidade. Se a identificacao falhar, peça para registrar novamente.)"
                        else:
                            debug.log(f"voice_id: RMS muito baixo ({rms:.4f}), assinatura nao salva")
                    else:
                        debug.log("voice_id: sem audio disponivel para enrollment")
                except Exception as e:
                    debug.log(f"voice_id: erro no enrollment automatico: {e}")
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
            elif name == "sortear_brincadeira":
                result = _sortear_brincadeira(mem)
                needs_rerun = True
                debug.log(f"tool: sortear_brincadeira -> {result}")
            elif name == "buscar_historico":
                arg_q = args.get("query", "")
                arg_child = args.get("child_id") or None
                try:
                    arg_ult = int(args.get("ultimas_n_sessoes", 5))
                except Exception:
                    arg_ult = 5
                try:
                    arg_max = int(args.get("max_results", 50))
                except Exception:
                    arg_max = 50
                itens = historico.buscar(
                    query=arg_q,
                    child_id=arg_child,
                    max_results=arg_max,
                    ultimas_n_sessoes=arg_ult,
                )
                try:
                    import json as _json
                    result = _json.dumps(itens, ensure_ascii=False)
                except Exception as e:
                    result = f"[]"
                debug.log(f"tool: buscar_historico -> {len(itens)} itens")
            elif name == "registrar_voz":
                if child_id_ativo is None:
                    result = "Nenhuma crianca ativa. Use lembrar_nome primeiro para registrar uma voz."
                else:
                    try:
                        import assistente as _assistente_mod
                        with _assistente_mod.buf_lock:
                            fim = len(_assistente_mod.buf)
                            inicio = max(0, fim - int(5 * SR))
                            chunks = _assistente_mod.buf[inicio:fim]
                        if chunks:
                            audio_novo = np.concatenate(chunks)
                            rms = float(np.sqrt(np.mean(audio_novo**2)))
                            if rms > MIN_RMS_ENROLLMENT:
                                nome_salvo = mem.get("crianca", {}).get("nome", "")
                                reconhecimento_voz.salvar_assinatura(
                                    child_id_ativo, nome_salvo, audio_novo, baixa_confianca=False
                                )
                                reconhecimento_voz.marcar_assinatura_confirmada(child_id_ativo)
                                result = f"Assinatura de voz atualizada para {nome_salvo}!"
                            else:
                                result = f"Audio muito silencioso (RMS={rms:.4f}). Fale mais alto e tente de novo."
                        else:
                            result = "Sem audio disponivel. Fale algo primeiro para registrar sua voz."
                    except Exception as e:
                        result = f"Erro ao registrar voz: {e}"
                        debug.log(f"tool: registrar_voz erro: {e}")
                debug.log(f"tool: registrar_voz -> child_id={child_id_ativo[:12] if child_id_ativo else 'none'}...")
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

    if not resposta:
        resposta = "Hmm, me perdi no pensamento. Pode me contar de novo?"
        debug.log("perguntar: resposta vazia da LLM, usando fallback")

    if child_id_ativo and reconhecimento_voz.verificar_re_enrollment(child_id_ativo, mem.get("crianca", {}).get("nome", "")):
        debug.log(f"voice_id: re-enrollment necessario para {child_id_ativo[:12]}...")
        try:
            audio_novo = None
            if audio is not None and len(audio) > 0.5 * SR:
                audio_novo = audio
            else:
                import assistente as _assistente_mod
                with _assistente_mod.buf_lock:
                    fim = len(_assistente_mod.buf)
                    inicio = max(0, fim - int(5 * SR))
                    chunks = _assistente_mod.buf[inicio:fim]
                if chunks:
                    audio_novo = np.concatenate(chunks)
            if audio_novo is not None:
                rms = float(np.sqrt(np.mean(audio_novo**2)))
                if rms > MIN_RMS_ENROLLMENT:
                    nome = mem.get("crianca", {}).get("nome", "")
                    reconhecimento_voz.atualizar_assinatura(child_id_ativo, nome, audio_novo)
                    reconhecimento_voz.marcar_assinatura_confirmada(child_id_ativo)
                    debug.log(f"voice_id: re-enrollment concluido para {child_id_ativo[:12]}...")
        except Exception as e:
            debug.log(f"voice_id: erro no re-enrollment: {e}")

    memoria.salvar(mem, child_id_ativo)
    if child_id_ativo:
        _salvar_ultimo_child(child_id_ativo, mem.get("crianca", {}).get("nome", ""))

    historico_sessao.append({"role": "user", "content": texto})
    historico_sessao.append({"role": "assistant", "content": resposta})
    try:
        historico.salvar_mensagem(session_id, "user", texto, child_id=child_id_ativo)
        historico.salvar_mensagem(session_id, "assistant", resposta, child_id=child_id_ativo)
    except Exception as e:
        debug.log(f"erro_historico_salvar: {e}")

    max_hist = MAX_HISTORICO_TURNOS * 2
    if len(historico_sessao) > max_hist:
        historico_sessao = historico_sessao[-max_hist:]

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


ultima_metricas_tts = {"gen": 0.0, "primeiro_audio": 0.0, "playback": 0.0}


def falar(texto: str):
    global stream
    t_chamada = time.time()
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

    gen = time.time() - t_chamada
    debug.log(f"falar: iniciando playback, samples={len(samples)} gen={gen:.2f}s")
    t0 = time.time()
    t_play = time.time()

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
        playback = time.time() - t0
        ultima_metricas_tts["gen"] = gen
        ultima_metricas_tts["primeiro_audio"] = t_play - t_chamada
        ultima_metricas_tts["playback"] = playback
        debug.log(f"falar: gen={gen:.2f}s primeiro_audio={t_play - t_chamada:.2f}s playback={playback:.2f}s")
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
buf_base = 0
stream = None
nome_crianca = ""
_accum = np.array([], dtype=np.float32)

# Acumula chunks pequenos (ex: 15 samples) em blocos de 1024 amostras
# para evitar que o buffer encha e pode demais
_ACCUM_TARGET = 1024


def _stream_callback(indata, frames, time, status):
    global _accum, buf_base
    _accum = np.concatenate([_accum, indata.ravel()])
    if len(_accum) < _ACCUM_TARGET:
        return
    chunk = _accum[:_ACCUM_TARGET].copy()
    _accum = _accum[_ACCUM_TARGET:]
    with buf_lock:
        buf.append(chunk)
        if len(buf) > BUF_MAX_CHUNKS:
            excesso = len(buf) - BUF_MAX_CHUNKS
            del buf[:excesso]
            buf_base += excesso


def marca_atual() -> int:
    """Indice global do proximo chunk a ser gravado (estavel mesmo com poda)."""
    with buf_lock:
        return buf_base + len(buf)


def ler_desde(marca: int) -> list:
    """Retorna os chunks gravados a partir de uma marca global."""
    with buf_lock:
        local = max(0, marca - buf_base)
        return list(buf[local:])


def descartar_ate(marca: int) -> None:
    """Descarta do buffer tudo antes da marca global informada."""
    global buf_base
    with buf_lock:
        local = max(0, min(marca - buf_base, len(buf)))
        if local > 0:
            del buf[:local]
            buf_base += local


def iniciar_stream():
    global stream, buf, nome_crianca, _accum, buf_base
    nome_crianca = mem.get("crianca", {}).get("nome", "")
    with buf_lock:
        _accum = np.array([], dtype=np.float32)
        buf = []
        buf_base = 0
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
