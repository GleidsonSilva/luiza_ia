#!/usr/bin/env python3
"""
POC: validacao de transcricao STT (Whisper) com o decision model Clef da Cloudflare
rodando localmente via Ollama (/v1/systemone).

Ideia: depois que o mlx-whisper transcreve a fala da crianca, o Clef avalia se a
transcricao faz sentido antes de mandar para a LLM. Se nao fizer sentido, o
assistente pede para repetir em vez de responder bobeira.

Uso:
  python poc_clef_stt.py --demo                 # casos + pares + degradacao
  python poc_clef_stt.py --pares                # frase correta x erro fonetico
  python poc_clef_stt.py --degradacao           # quantas trocas ate reprovar
  python poc_clef_stt.py --audio a.wav b.wav    # whisper real + pipeline inteiro
  python poc_clef_stt.py --audio               # so cenarios sinteticos (ruido)
  python poc_clef_stt.py --demo --sem-contexto  # A/B: mede o custo do contexto
  python poc_clef_stt.py --texto "meu nome e elo"
  python poc_clef_stt.py --texto "ah e" --contexto "Luiza: voce quer jogar?"
  python poc_clef_stt.py --texto "a a a a" --modelo clef
  echo "vamos jogar" | python poc_clef_stt.py --stdin
  python poc_clef_stt.py --demo --json

Requisitos:
  ollama serve                     # servidor local (>= 0.35.1)
  ollama pull clef-flash           # 9B, ~10 GB (ou: ollama pull clef)

Decisao: valida = p(true) >= LIMIAR_VALIDO e tipo_texto fora de TIPOS_INVALIDOS.

Achados do POC (37 casos rotulados + 21 pares + 4 curvas de degradacao):
- COM contexto (padrao): clef 35/37 (95%). SEM contexto: 34/37. O filtro atual
  de vogais: 22/37. Os 2 erros restantes sao ambiguidades documentadas
  ('eu quero' cortado; 'com licenca' alucinado pelo whisper em silencio).
- ERROS FONETICOS NAO DERRUBAM O GATE: 21/21 pares corrompidos aceitos,
  delta medio -0.063. R>L: -0.05 a -0.06; R omitida: -0.13; omissao/sotaque
  e troca que vira palavra real: ~0. So 1 troca individual passa de -0.2
  ('farinha'->'falinha').
- DEGRADACAO: 5 trocas acumuladas nao quebram nenhuma curva (p continua
  0.85-0.94). So quebra quando a palavra vira irreconhecivel (cluster de
  consoantes, 'xqzt' -> 0.42). O gate mede FALA CAPTADA, nao ortografia.
- LIMITE (sem cura pelo texto): troca semantica 'carro'->'calo'->'bolo'
  mantem p=0.70 e passa; alucinacao plausivel do whisper tambem passa.
- CONTEXTO: ajuda frases curtas ('qual e o seu nome' 0.39 -> 0.89), mas so
  se a instrucao disser que a transcricao NAO precisa responder o contexto —
  sem isso, resposta fora de assunto caia para 0.53. Contexto nao infla
  ruido (vocalizes ficam em 0.01-0.08), mas punia fala formal fora do
  padrao ('ola mestre, estou a seu dispor' 0.36); a regra 'nao penalize
  formalidade/nome proprio/idade aparente' subiu para 0.69 sem regredir
  nenhum outro caso.
- REGRA: valida = p>=0.55 E NAO (tipo invalido com conf>=0.50). A confianca
  do choice e o que separa ruido real (conf 0.88) de texto incomum (0.19-0.31).
- AUDIO REAL (--audio, whisper mlx-large-v3 + initial_prompt do producao):
  gate de rms 0.010 ja mata silencio e ruido fraco ANTES do whisper; ruido
  forte passa o rms e o whisper alucina ('que isso'), clef reprova (p=0.33)
  onde a heuristica de vogais deixaria passar; loop 'oi oi oi...' do whisper
  tambem cai para p=0.04. Nao houve falso negativo apos a regra de estilo.
- WHISPER NAO E DETERMINISTICO: o mesmo wav deu 'Tudo bem?', 'tudo E aí' e
  'ae tudo vindo E aí' em execucoes iguais (temperature fallback).
- Latencia ~2.2-2.4s por chamada (clef-flash 9B no M4 Pro, modelo carregado).
"""

import argparse
import json
import os
import re
import sys
import time

import requests

OLLAMA_SYSTEMONE = "http://localhost:11434/v1/systemone"
MODELO_PADRAO = "clef-flash"
LIMIAR_VALIDO = 0.55
# O tipo so bloqueia quando o modelo tem certeza; em textos ambigos a confianca
# do choice cai para ~0.2 e o julgamento fica com o p(true).
LIMIAR_TIPO_FORTE = 0.50
TIPOS_INVALIDOS = {"ruido_sem_sentido", "outro_idioma"}
# No pipeline real a LLM sempre tem a ultima fala dela; sem isso frases
# simples como 'qual e o seu nome' caem para ~0.4 e sao reprovadas.
CONTEXTO_PADRAO = "Luiza: ola! eu sou a Luiza, voce quer conversar?"

QUESTOES = {
    "transcricao_valida": {
        "type": "noul",
        "instructions": (
            "O estado abaixo contem a transcricao automatica de uma fala de uma crianca "
            "em portugues do Brasil, captada por microfone e transcrita por um "
            "reconhecimento de voz. Quando houver contexto_anterior, ele e a ultima "
            "fala da conversa. A transcricao corresponde a uma fala real dita pela "
            "crianca e faz sentido?\n"
            "REGRAS:\n"
            "- contexto_anterior e so a ultima fala da Luiza, para situar a "
            "conversa. A transcricao NAO precisa ser uma resposta direta dele; "
            "crianca fala o que quer. Julgue apenas se o texto e uma fala real "
            "e coerente por si so.\n"
            "- Crianca fala com gramatica errada, erra contas e mistura palavras. "
            "Isso NAO invalida a transcricao se for uma fala plausivel, ex: "
            "'sete mais cinco doze', 'minha escola e chata'.\n"
            "- Respostas curtas ('oi', 'sim', 'nao sei', 'ah e') sao validas quando "
            "plausiveis na conversa.\n"
            "- Frase curta mas fechada ('vamos brincar?', 'eu vi um rato preto') "
            "e COMPLETA; nao e incompleta so por ser curta.\n"
            "- Se quase todo o texto estiver em portugues, e valido, mesmo que "
            "cite outro idioma ('voce sabe falar ingles').\n"
            "- INVALIDO: repeticoes, vocalizes, trechos cortados antes de fechar a "
            "ideia ('eu quero' sem dizer o quê, 'eles pre'), saladas de palavras "
            "sem relacao, ou texto em outro idioma.\n"
            "- VALIDO mesmo incompleto se a ideia principal ja fecha: 'vamos jogar'.\n"
            "- NAO penalize formalidade, tom, vocabulario incomum, nome proprio ou "
            "idade aparente do falante ('ola mestre, estou a seu dispor'). O "
            "julgamento e se houve fala real captada, nao quem falou nem como falou.\n"
            "- NAO penalize por a transcricao nao responder o contexto; tambem nao "
            "penalize por ser fora de assunto."
        ),
        "criteria": {
            "true": "A transcricao e coerente: uma fala real da crianca em portugues.",
            "false": "A transcricao esta errada, cortada, repetida, em outro idioma, ou e apenas ruido/vocalizes.",
        },
    },
    "tipo_texto": {
        "type": "choice",
        "instructions": (
            "Classifique a transcricao recebida do reconhecimento de voz. Use o "
            "IDIOMA em que o texto esta escrito, nao o assunto dele: um texto em "
            "portugues que fala sobre ingles ou espanhol NAO e outro idioma.\n"
            "'ruido_sem_sentido' SIGNIFICA que nenhuma fala humana foi "
            "reconhecida (vocalizes, repeticoes mechanicas, letras soltas). "
            "Palavras incomuns, erros de pronuncia, neologismos ou gírias de "
            "crianca ('chimco', 'minino', 'blinca') sao FALA REAL, nao ruido."
        ),
        "criteria": {
            "frase_completa": "Frase, pergunta ou resposta curta em portugues, pronta para a Luiza responder",
            "frase_incompleta": "Fala real em portugues, mas cortada antes do fim ou com trechos faltando",
            "ruido_sem_sentido": "Nenhuma fala foi reconhecida: vocalizes, repeticoes mecanicas ou letras soltas",
            "outro_idioma": "Texto escrito em outro idioma (ingles, espanhol etc.), praticamente nada em portugues",
        },
    },
    "qualidade": {
        "type": "score",
        "instructions": "Qual a qualidade da transcricao da fala da crianca?",
        "criteria": [
            "Incompreensivel: nao da para entender nada",
            "Parcial: da para entender o assunto, mas com erros ou cortes",
            "Clara: fala bem transcrita e facil de entender",
        ],
    },
}

DEMO = [
    # (rotulo, transcricao, contexto, esperado)
    # contexto = ultima fala da Luiza, como no pipeline real (None = padrao)

    # --- falas validas ---
    ("fala limpa", "meu nome é eloá e eu gosto de dinossauros",
     "Luiza: ola! como e o seu nome?", True),
    ("fala limpa", "vamos jogar forca",
     "Luiza: voce quer jogar alguma coisa?", True),
    ("fala limpa", "quantos é sete mais cinco",
     "Luiza: quanto e sete mais cinco?", True),
    ("fala limpa", "oi luiza tudo bem",
     "Luiza: ola! eu sou a Luiza", True),
    ("fala limpa", "qual e o seu nome",
     "Luiza: como voce se chama?", True),
    ("fala limpa", "voce quer ser minha amiga",
     "Luiza: adoro conversar com voce", True),
    ("fala limpa", "eu tenho oito anos e moro em sao paulo",
     "Luiza: quantos anos voce tem?", True),
    ("fala limpa", "QUER JOGAR COMIGO",
     "Luiza: voce quer brincar?", True),
    ("fala limpa", "eu fui na escola ontem de bicicleta",
     "Luiza: o que voce fez ontem?", True),
    ("fala limpa", "voce sabe falar ingles",
     "Luiza: voce fala outro idioma?", True),
    ("fala com erro de grafa", "eu gosto de dinosauro",
     "Luiza: voce gosta de dinossauros?", True),
    ("fala com erro de grafa", "minha escola e chata mas eu gosto de la",
     "Luiza: voce gosta da escola?", True),
    ("fala com erro de grafa", "sete mais cinco doze",
     "Luiza: quanto e sete mais cinco?", True),
    ("nome proprio ood", "meu melhor amigo se chama gleverson",
     "Luiza: voce tem amigos na escola?", True),
    ("resposta curta", "ah é", "Luiza: voce quer jogar forca comigo?", True),
    ("resposta curta", "sim", "Luiza: voce quer jogar forca?", True),
    ("resposta curta", "nao sei", "Luiza: qual e a sua cor favorita?", True),
    ("resposta curta", "oi", "Luiza: como voce se chama?", True),

    # --- transcricoes falhas (contexto presente, como no pipeline real) ---
    ("ruido vocalize", "a a a a a", None, False),
    ("ruido vocalize", "ah ah ah ah ah", None, False),
    ("ruido vocalize", "nhe nhe nhe nhe", None, False),
    ("ruido vocalize", "ta ta ta ta ta", None, False),
    ("repeticao whisper", "eu eu eu eu eu eu", None, False),
    # loop de whisper com conteudo legivel: a intencao e clara, a Luiza pode
    # responder normalmente -> nao bloquear
    ("repeticao legivel", "vamos jogar vamos jogar vamos jogar",
     "Luiza: voce quer jogar forca?", True),
    ("repeticao whisper", "voce voce voce voce voce", None, False),
    ("babble", "banana banana banana banana", None, False),
    ("babble", "casa azul comigo amanhã quartoba", None, False),
    ("frase cortada", "eles pre", "Luiza: o que voce fez na escola?", False),
    ("frase cortada", "eu quero", "Luiza: voce quer mais um bolo?", False),
    ("idioma errado", "i don't know what you are saying", None, False),
    ("idioma errado", "the weather is nice today", None, False),
    ("texto sem sentido", "xyzzy plugh frobnicate", None, False),
    ("texto sem sentido", "...", None, False),
    ("texto sem sentido", "1234567890", None, False),
    ("sem vogal", "tch pff", None, False),
    ("sem vogal", "pff ts pf", None, False),

    # --- limites do julgamento so por texto ---
    ("limite", "com licenca", "silencio no microfone (whisper alucinou)", False),
]

# Pares: frase correta x mesma frase com erro fonetico (controle pareado).
# Ambos os lados sao fala real -> os dois devem passar. O que se mede e o
# delta de p(true) causado pela troca.
# (rotulo_da_troca, frase_correta, frase_corrompida, contexto)
PARES = [
    # --- troca de R por L (processo fonologico comum em crianca) ---
    ("R>L inicio", "eu vi um rato preto", "eu vi um lato preto",
     "Luiza: voce gosta de animais?"),
    ("R>L inicio", "a Rita chegou agora", "a Lita chegou agora",
     "Luiza: voce tem amigas?"),
    ("R>L inicio", "meu irmao se chama Rafael", "meu irmao se chama Lafael",
     "Luiza: como se chama seu irmao?"),
    ("R>L medial", "meu carro é vermelho", "meu calo é vermelho",
     "Luiza: qual a cor do carro da sua familia?"),
    ("R>L medial", "fui no restaurante ontem", "fui no lestaurante ontem",
     "Luiza: o que voce fez ontem?"),
    ("R>L medial", "ta muito barulho ai", "ta muito balulho ai",
     "Luiza: o que e esse barulho?"),
    ("R>L medial", "eu so queria farinha", "eu so queria falinha",
     "Luiza: voce ajuda na cozinha?"),
    ("R>L parada", "vamos brincar?", "vamos blincar?",
     "Luiza: voce quer brincar?"),
    ("R>L parada", "vou pra praça", "vou pra plaça",
     "Luiza: voce sai de casa?"),
    ("R>L parada", "parabéns pra voce", "palabéns pra voce",
     "Luiza: hoje e aniversario de alguem?"),
    ("R omitida", "ele trabalha na escola", "ele tabalha na escola",
     "Luiza: o que seus pais fazem?"),
    ("R omitida", "meu tio tem um trator", "meu tio tem um tlator",
     "Luiza: seu tio trabalha com o que?"),

    # --- outras trocas de fala infantil / ASR ---
    ("vogal", "o menino brinca na escola", "o minino blinca na escola",
     "Luiza: quantos anos voce tem?"),
    ("troca real", "eu tenho uma bola azul", "eu tenho um bolo azul",
     "Luiza: voce tem brinquedos?"),
    ("consoante", "cinco mais cinco", "chimco mais chimco",
     "Luiza: quanto e cinco mais cinco?"),
    ("omissao", "muito obrigado", "mui brigado",
     "Luiza: voce agradece quando ganha algo?"),
    ("sotaque", "ela está na escola", "ela tá na escola",
     "Luiza: onde esta sua mae?"),
    ("nome distorcido", "minha amiga se chama Eloá", "minha amiga se chama Elo",
     "Luiza: como se chama sua melhor amiga?"),

    # --- multiplas trocas (frase ainda legivel) ---
    ("multi 2", "meu tio trabalha no restaurante",
     "meu tio tabalha no lestaurante", "Luiza: o que seu tio faz?"),
    ("multi 3", "o menino brinca na escola",
     "o minino blinca na lescola", "Luiza: voce vai pra escola?"),
    ("multi 4", "a menina comeu o lanche na praça",
     "a minina cumeu o lanchi na plaça", "Luiza: o que voce comeu hoje?"),
]

# Curva de degradacao: mesma frase com N trocas foneticas acumuladas.
# Mostra em quantas trocas o gate comeca a reprovar fala real.
# (rotulo, contexto, [(n_trocas, frase), ...])
DEGRADACAO = [
    ("R>L acumulada", "Luiza: o que seu tio faz?", [
        (0, "meu tio trabalha no restaurante perto da praça"),
        (1, "meu tio tabalha no restaurante perto da praça"),
        (2, "meu tio tabalha no lestaurante perto da praça"),
        (3, "meu tio tabalha no lestaurante pelto da praça"),
        (4, "meu tio tabalha no lestaurante pelto da plaça"),
        (5, "meu tio tabalha na lestaulante pelto da plaça"),
    ]),
    ("erros infantis acumulados", "Luiza: o que voce comeu hoje?", [
        (0, "a menina comeu o lanche na praça da escola"),
        (1, "a minina comeu o lanche na praça da escola"),
        (2, "a minina cumeu o lanche na praça da escola"),
        (3, "a minina cumeu o lanchi na plaça da escola"),
        (4, "a minina cumeu o lanchi na plaça da lecola"),
        (5, "a minina cumiu o lanchi na plaça da lecola"),
    ]),
    ("controle: trocas progressivas em palavra unica", "Luiza: o que voce comeu?", [
        (0, "fui no restaurante ontem"),
        (1, "fui no lestaurante ontem"),
        (2, "fui no lestaulante ontem"),
        (3, "fui no lestaulanti ontem"),
        (4, "fui no lestunti ontem"),
        (5, "fui no liatanti ontem"),
        (6, "fui no xqzt ontem"),
        (7, "fui no blmpt ontem"),
    ]),
    # Limite: o gate julga FALA CAPTADA, nao significado. Troca que vira
    # palavra real de outro sentido passa — e nao tem como o texto sozinho
    # acusar.
    ("troca semantica (limite)", "Luiza: seu familia tem carro?", [
        (0, "meu pai levou o carro pra lavar"),
        (1, "meu pai levou o calo pra lavar"),
        (2, "meu pai levou o bolo pra lavar"),
        (3, "meu pai colou o bolo pra lavar"),
    ]),
]


def heuristica_atual(texto: str) -> bool:
    """Réplica do filtro de vogais em assistente.py:transcrever()."""
    return bool(
        re.search(r"[aeiouáéíóúâêîôûãõ]", texto, re.I)
        or re.search(r"^\d+$", texto.strip())
    )


def montar_estado(texto: str, contexto: str | None = None) -> str | dict:
    if contexto:
        return {"contexto_anterior": contexto, "transcricao_recebida": texto}
    return texto


def verificar_transcricao(texto: str, modelo: str = MODELO_PADRAO,
                          contexto: str | None = None, timeout: int = 60) -> dict:
    payload = {
        "model": modelo,
        "state": montar_estado(texto, contexto),
        "questions": QUESTOES,
    }
    t0 = time.time()
    r = requests.post(OLLAMA_SYSTEMONE, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    body["latencia_ms"] = (time.time() - t0) * 1000
    return body


def extrair(body: dict) -> dict:
    ans = body.get("answers", {})
    noul = ans.get("transcricao_valida", {})
    escolha = ans.get("tipo_texto", {})
    score = ans.get("qualidade", {})
    p_valido = noul.get("noul", 0.0)
    tipo = escolha.get("choice", "?")
    conf_tipo = escolha.get("confidence", 0.0)
    bloqueado = tipo in TIPOS_INVALIDOS and conf_tipo >= LIMIAR_TIPO_FORTE
    valido = p_valido >= LIMIAR_VALIDO and not bloqueado
    return {
        "valido": valido,
        "p_valido": p_valido,
        "tipo": tipo,
        "p_tipo": conf_tipo,
        "bloqueado": bloqueado,
        "score": score.get("score", 0.0),
        "latencia_ms": body.get("latencia_ms", 0.0),
    }


def imprimir(label: str, texto: str, heuristica: bool, res: dict,
             contexto: str | None = None) -> None:
    v = "VALIDA" if res["valido"] else "REPROVA"
    print(f"\n  [{label}] {texto!r}" if label else f"\n  transcricao : {texto!r}")
    if contexto:
        print(f"  contexto    : {contexto!r}")
    print(f"  heuristica  : {'aceita' if heuristica else 'rejeita'} (filtro atual de vogais)")
    print(f"  clef        : {v}  p(true)={res['p_valido']:.3f}  "
          f"limiar={LIMIAR_VALIDO}")
    bloqueio = "  [tipo bloqueia]" if res.get("bloqueado") else ""
    print(f"  tipo        : {res['tipo']} (conf={res['p_tipo']:.3f}){bloqueio}   "
          f"qualidade={res['score']:.2f}/2   {res['latencia_ms']:.0f} ms")


def rodar(casos: list[tuple], modelo: str) -> int:
    ctxs = {ctx for _, _, ctx, _ in casos}
    if ctxs == {None}:
        hdr_ctx = "NENHUM (--sem-contexto)"
    elif len(ctxs) == 1:
        hdr_ctx = repr(next(iter(ctxs)))
    else:
        hdr_ctx = "por caso (impresso em cada teste)"
    print(f"modelo: {modelo}   endpoint: {OLLAMA_SYSTEMONE}   limiar: {LIMIAR_VALIDO}")
    print(f"contexto: {hdr_ctx}")
    stats = {"heuristica_reprova": 0, "clef_reprova": 0, "divergencias": 0,
             "latencias": [], "clef_acertos": 0, "heuristica_acertos": 0, "rotulos": 0}
    por_categoria: dict[str, dict[str, int]] = {}
    erros: list[str] = []
    for label, texto, ctx, esperado in casos:
        h = heuristica_atual(texto)
        try:
            body = verificar_transcricao(texto, modelo=modelo, contexto=ctx)
            res = extrair(body)
        except requests.RequestException as e:
            print(f"\n  ERRO ao consultar Clef: {e}")
            print("  Rode: ollama serve && ollama pull clef-flash")
            return 2
        imprimir(label, texto, h, res, ctx)
        stats["latencias"].append(res["latencia_ms"])
        if not h:
            stats["heuristica_reprova"] += 1
        if not res["valido"]:
            stats["clef_reprova"] += 1
        if h != res["valido"]:
            stats["divergencias"] += 1
        if esperado is not None:
            stats["rotulos"] += 1
            clef_ok = int(res["valido"] == esperado)
            heur_ok = int(h == esperado)
            stats["clef_acertos"] += clef_ok
            stats["heuristica_acertos"] += heur_ok
            cat = por_categoria.setdefault(
                label, {"total": 0, "clef": 0, "heur": 0})
            cat["total"] += 1
            cat["clef"] += clef_ok
            cat["heur"] += heur_ok
            if not clef_ok:
                tipo = "FALSO POSITIVO" if res["valido"] else "FALSO NEGATIVO"
                erros.append(f"  {tipo:15s} [{label}] {texto!r} "
                             f"p={res['p_valido']:.3f} tipo={res['tipo']}")
    lat = stats["latencias"]
    print("\n" + "=" * 70)
    print(f"casos: {len(casos)}   heuristica rejeitou: {stats['heuristica_reprova']}   "
          f"clef reprovou: {stats['clef_reprova']}   divergencias: {stats['divergencias']}")
    if stats["rotulos"]:
        n = stats["rotulos"]
        print(f"acuracia total: clef {stats['clef_acertos']}/{n}   "
              f"heuristica atual {stats['heuristica_acertos']}/{n}")
        print("\npor categoria (clef | heuristica):")
        for cat, c in por_categoria.items():
            marca = "" if c["clef"] == c["total"] else "  <-"
            print(f"  {cat:<26s} {c['clef']:>2}/{c['total']:<2}   "
                  f"{c['heur']:>2}/{c['total']:<2}{marca}")
    if erros:
        print("\nerros do clef:")
        print("\n".join(erros))
    if lat:
        print(f"\nlatencia clef: min={min(lat):.0f}ms  media={sum(lat)/len(lat):.0f}ms  "
              f"max={max(lat):.0f}ms")
    return 0


def rodar_pares(modelo: str, pares: list[tuple] | None = None) -> int:
    pares = PARES if pares is None else pares
    print("\n" + "=" * 70)
    print("pares: frase correta x mesma frase com erro fonetico (mesmo contexto)")
    print("os dois lados sao fala real -> os dois devem ser aceitos")
    print(f"modelo: {modelo}   limiar: {LIMIAR_VALIDO}")
    aceitos = 0
    controles = 0
    deltas: list[float] = []
    por_rotulo: dict[str, list[float]] = {}
    fracassos: list[str] = []
    for rotulo, correta, corrompida, ctx in pares:
        try:
            ok = extrair(verificar_transcricao(correta, modelo=modelo, contexto=ctx))
            erro = extrair(verificar_transcricao(corrompida, modelo=modelo, contexto=ctx))
        except requests.RequestException as e:
            print(f"\n  ERRO ao consultar Clef: {e}")
            print("  Rode: ollama serve && ollama pull clef-flash")
            return 2
        d = erro["p_valido"] - ok["p_valido"]
        deltas.append(d)
        por_rotulo.setdefault(rotulo, []).append(d)
        controles += int(ok["valido"])
        aceitos += int(erro["valido"])
        marca_ctrl = "" if ok["valido"] else "   !! controle reprova"
        status = "aceita" if erro["valido"] else "REPROVA"
        print(f"\n  [{rotulo}]")
        print(f"    correta    : {correta!r}")
        print(f"                 p={ok['p_valido']:.3f}  {ok['tipo']}{marca_ctrl}")
        print(f"    corrompida : {corrompida!r}")
        print(f"                 p={erro['p_valido']:.3f}  {erro['tipo']}   "
              f"delta={d:+.3f}  -> {status}")
        if not erro["valido"]:
            fracassos.append(f"  REPROVA [{rotulo}] {corrompida!r} "
                             f"p={erro['p_valido']:.3f} tipo={erro['tipo']}")

    n = len(pares)
    print("\n" + "-" * 70)
    print(f"pares: {n}   controles aceitos: {controles}/{n}   "
          f"corrompidas aceitas: {aceitos}/{n}")
    if deltas:
        print(f"delta medio: {sum(deltas)/len(deltas):+.3f}   "
              f"pior: {min(deltas):+.3f}   melhor: {max(deltas):+.3f}")
    print("\ndelta medio por tipo de troca:")
    for rotulo, ds in por_rotulo.items():
        print(f"  {rotulo:<16s} {sum(ds)/len(ds):+.3f}   (n={len(ds)})")
    if fracassos:
        print("\nfracassos (corrompidas reprovadas):")
        print("\n".join(fracassos))
    return 0


def medir_degradacao(modelo: str, degradacao: list | None = None) -> list[dict]:
    """Roda a curva de degradacao e devolve os dados (para relatorio ou JSON)."""
    degradacao = DEGRADACAO if degradacao is None else degradacao
    saida = []
    for rotulo, ctx, niveis in degradacao:
        linhas = []
        base_p = None
        quebra = None
        for n, frase in niveis:
            body = verificar_transcricao(frase, modelo=modelo, contexto=ctx)
            r = extrair(body)
            if base_p is None:
                base_p = r["p_valido"]
            if not r["valido"] and quebra is None:
                quebra = n
            linhas.append({
                "n_trocas": n,
                "frase": frase,
                "p": r["p_valido"],
                "delta": r["p_valido"] - base_p,
                "tipo": r["tipo"],
                "valido": r["valido"],
            })
        saida.append({"rotulo": rotulo, "contexto": ctx, "niveis": linhas,
                      "quebra_em": quebra})
    return saida


def imprimir_degradacao(dados: list[dict]) -> int:
    print("\n" + "=" * 70)
    print("curva de degradacao: quantas trocas foneticas ate o gate reprovar")
    print(f"limiar: {LIMIAR_VALIDO}   (as frases sao fala real em todos os niveis)")
    quebras = []
    for bloco in dados:
        print(f"\n  [{bloco['rotulo']}]  contexto={bloco['contexto']!r}")
        for l in bloco["niveis"]:
            status = "VALIDA  " if l["valido"] else "REPROVA"
            print(f"    {l['n_trocas']} trocas  p={l['p']:.3f}  "
                  f"delta={l['delta']:+.3f}  {status}  {l['frase']!r}")
        if bloco["quebra_em"] is None:
            ultimo = bloco["niveis"][-1]["n_trocas"]
            print(f"    -> nao quebra em {ultimo} trocas")
        else:
            print(f"    -> QUEBRA em {bloco['quebra_em']} troca(s)")
            quebras.append(bloco["quebra_em"])
    if quebras:
        print("\nresumo: gate reprova fala real a partir de "
              f"{min(quebras)} troca(s) (media {sum(quebras)/len(quebras):.1f})")
    else:
        print("\nresumo: nenhuma curva quebrou")
    return 0


# --- pipeline real de audio (replica assistente.transcrever(), linhas 439-464) ---
SR_LOCAL = 16000
TARGET_PEAK = 0.95
MAX_GAIN = 10.0
RMS_MINIMO = 0.010
TRIM_THRESHOLD = 0.005
REPO_WHISPER = "mlx-community/whisper-large-v3-mlx"
PROMPT_INICIAL = (
    "ola vamos brincar qual e o seu nome voce gosta de jogar que historia legal"
)


def _carregar_audio(caminho: str):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(caminho, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sr != SR_LOCAL:
        n = int(round(len(audio) * SR_LOCAL / sr))
        velho = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        novo = np.linspace(0.0, 1.0, num=max(n, 1), endpoint=False)
        audio = np.interp(novo, velho, audio).astype("float32")
    return audio


def transcrever_audio(caminho: str, modelo: str = MODELO_PADRAO,
                      contexto: str | None = CONTEXTO_PADRAO) -> dict:
    """Roda o pipeline de assistente.transcrever() num arquivo e devolve cada etapa."""
    import numpy as np

    audio = _carregar_audio(caminho)
    duracao = len(audio) / SR_LOCAL
    if len(audio):
        mask = np.abs(audio) > TRIM_THRESHOLD
        idxs = np.where(mask)[0]
        if len(idxs):
            audio = audio[idxs[0]:idxs[-1] + 1]
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0

    res: dict = {"arquivo": caminho, "duracao": duracao, "rms": rms,
                 "trim_s": len(audio) / SR_LOCAL, "etapa": "ok",
                 "whisper": "", "heuristica": False, "clef": None}
    if len(audio) < SR_LOCAL * 0.5:
        res["etapa"] = "REJEITADO antes do whisper: trecho < 0.5s"
        return res
    if rms < RMS_MINIMO:
        res["etapa"] = f"REJEITADO antes do whisper: rms {rms:.5f} < {RMS_MINIMO}"
        return res

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * min(MAX_GAIN, TARGET_PEAK / peak)

    import mlx_whisper
    r = mlx_whisper.transcribe(audio, language="pt", path_or_hf_repo=REPO_WHISPER,
                               initial_prompt=PROMPT_INICIAL)
    texto = r["text"].strip()
    res["whisper"] = texto
    res["heuristica"] = heuristica_atual(texto)
    if texto:
        res["clef"] = extrair(verificar_transcricao(texto, modelo=modelo,
                                                    contexto=contexto))
    return res


def _gerar_sinteticos() -> list[tuple]:
    """Silencio e ruido: os cenarios que fazem o whisper alucinar fala."""
    import os
    import tempfile

    import numpy as np
    import soundfile as sf

    base = tempfile.mkdtemp(prefix="poc_audio_")
    dur = 7.5
    n = int(SR_LOCAL * dur)
    rng = np.random.default_rng(7)
    casos = [
        ("silencio digital", np.zeros(n, dtype="float32")),
        ("ruido fraco (rms ~0.003)", rng.normal(0, 0.003, n).astype("float32")),
        ("ruido forte (rms ~0.02, passa o gate)", rng.normal(0, 0.02, n).astype("float32")),
    ]
    saida = []
    for i, (rotulo, buf) in enumerate(casos):
        caminho = os.path.join(base, f"sintetico_{i}.wav")
        sf.write(caminho, buf, SR_LOCAL)
        saida.append((rotulo, caminho))
    return saida


def medir_audio(arquivos: list[str], modelo: str = MODELO_PADRAO,
                contexto: str | None = CONTEXTO_PADRAO) -> list[dict]:
    itens = [dict(rotulo=os.path.basename(a), **transcrever_audio(a, modelo, contexto))
             for a in arquivos]
    itens += [dict(rotulo=r, **transcrever_audio(c, modelo, contexto))
              for r, c in _gerar_sinteticos()]
    return itens


def imprimir_audio(itens: list[dict], contexto: str | None = CONTEXTO_PADRAO) -> int:
    print("\n" + "=" * 70)
    print("pipeline real: trim -> rms gate -> whisper -> heuristica -> clef")
    print(f"limiar clef: {LIMIAR_VALIDO}   rms gate: {RMS_MINIMO}   "
          f"contexto: {contexto!r}")
    for r in itens:
        print(f"\n  [{r['rotulo']}]  {os.path.basename(r['arquivo'])}  "
              f"dur={r['duracao']:.2f}s trim={r['trim_s']:.2f}s rms={r['rms']:.5f}")
        if r["etapa"] != "ok":
            print(f"    {r['etapa']}   -> descartado, whisper nem roda")
            continue
        c = r["clef"]
        if c is None:
            print("    whisper: (vazio)   -> descartado")
            continue
        h = "PASSA" if r["heuristica"] else "REPROVA"
        print(f"    whisper   : {r['whisper']!r}")
        print(f"    heuristica: {h}")
        print(f"    clef      : p={c['p_valido']:.3f} tipo={c['tipo']} "
              f"-> {'VALIDA' if c['valido'] else 'INVALIDA'}")
        print(f"    decisao   : {'PROSSEGUE' if c['valido'] else 'REPETIR PEDIDO'}")
    passou_gate = [r for r in itens if r["etapa"] == "ok" and r["whisper"]]
    clef_ok = [r for r in passou_gate if r["clef"] and r["clef"]["valido"]]
    print("\nresumo: "
          f"{len(itens)} audios -> {len(passou_gate)} chegaram no clef -> "
          f"{len(clef_ok)} aceitos")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="POC Clef via Ollama para validar transcricao STT")
    ap.add_argument("--texto", action="append", default=[],
                    help="texto transcrito (repetivel)")
    ap.add_argument("--stdin", action="store_true", help="le textos do stdin (um por linha)")
    ap.add_argument("--demo", action="store_true",
                    help="roda os casos de demonstracao + pares foneticos")
    ap.add_argument("--pares", action="store_true",
                    help="roda so os pares (frase correta x erro fonetico)")
    ap.add_argument("--degradacao", action="store_true",
                    help="roda so a curva de degradacao (N trocas ate reprovar)")
    ap.add_argument("--audio", nargs="*", default=None, metavar="WAV",
                    help="transcreve wav(s) com o whisper do projeto e aplica o "
                         "pipeline completo; sem arquivos roda so os cenarios "
                         "sinteticos (silencio e ruido)")
    ap.add_argument("--contexto", default=None,
                    help=f"ultima fala da Luiza (padrao: {CONTEXTO_PADRAO!r})")
    ap.add_argument("--sem-contexto", action="store_true",
                    help="nao envia contexto (mostra quanto a acuracia cai)")
    ap.add_argument("--modelo", default=MODELO_PADRAO,
                    help="clef-flash (padrao) ou clef (27B, mais preciso)")
    ap.add_argument("--json", action="store_true", help="saida crua em JSON")
    args = ap.parse_args()

    if args.sem_contexto:
        base_ctx = None
    else:
        base_ctx = args.contexto if args.contexto is not None else CONTEXTO_PADRAO

    def resolver_ctx(ctx: str | None) -> str | None:
        if args.sem_contexto:
            return None
        return ctx or base_ctx

    casos: list[tuple] = [("", t, base_ctx, None) for t in args.texto]
    if args.stdin:
        casos += [("", l.strip(), base_ctx, None) for l in sys.stdin if l.strip()]
    quer_audio = args.audio is not None
    audio_arquivos = args.audio or []
    so_extras = args.pares or args.degradacao or quer_audio
    if args.demo or (not casos and not so_extras):
        print("== casos de demonstracao ==")
        casos += [(lab, txt, resolver_ctx(ctx), esp) for lab, txt, ctx, esp in DEMO]

    sem_extras = not (args.texto or args.stdin) and not quer_audio
    quer_pares = args.pares or args.demo or (sem_extras and not args.degradacao)
    quer_degrada = args.degradacao or args.demo or (sem_extras and not args.pares)
    pares_ativos = [(rot, cor, rom, resolver_ctx(ctx)) for rot, cor, rom, ctx in PARES]

    if args.json:
        if not (quer_pares or quer_degrada or quer_audio):
            out: list = []
            for _, texto, ctx, _esperado in casos:
                body = verificar_transcricao(texto, modelo=args.modelo, contexto=ctx)
                body["heuristica_atual"] = heuristica_atual(texto)
                out.append(body)
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0
        payload: dict = {"casos": [], "pares": [], "degradacao": [], "audio": []}
        for _, texto, ctx, _esperado in casos:
            body = verificar_transcricao(texto, modelo=args.modelo, contexto=ctx)
            body["heuristica_atual"] = heuristica_atual(texto)
            payload["casos"].append(body)
        if quer_pares:
            for rotulo, correta, corrompida, ctx in pares_ativos:
                payload["pares"].append({
                    "rotulo": rotulo,
                    "correta_texto": correta,
                    "correta": extrair(verificar_transcricao(
                        correta, modelo=args.modelo, contexto=ctx)),
                    "corrompida_texto": corrompida,
                    "corrompida": extrair(verificar_transcricao(
                        corrompida, modelo=args.modelo, contexto=ctx)),
                })
        if quer_degrada:
            payload["degradacao"] = medir_degradacao(args.modelo)
        if quer_audio:
            payload["audio"] = medir_audio(audio_arquivos, args.modelo, base_ctx)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    rc = 0
    if casos:
        rc = rodar(casos, args.modelo)
        if rc:
            return rc
    if quer_pares:
        rc = rodar_pares(args.modelo, pares_ativos)
        if rc:
            return rc
    if quer_degrada:
        try:
            dados = medir_degradacao(args.modelo)
        except requests.RequestException as e:
            print(f"\n  ERRO ao consultar Clef: {e}")
            print("  Rode: ollama serve && ollama pull clef-flash")
            return 2
        rc = imprimir_degradacao(dados)
    if quer_audio:
        try:
            itens = medir_audio(audio_arquivos, args.modelo, base_ctx)
        except (requests.RequestException, OSError, ValueError) as e:
            print(f"\n  ERRO no pipeline de audio: {e}")
            return 2
        rc = imprimir_audio(itens, base_ctx)
    return rc


if __name__ == "__main__":
    sys.exit(main())
