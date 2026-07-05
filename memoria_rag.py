
import re
from collections import Counter

STOP_PT = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
    "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos",
    "como", "mas", "foi", "ao", "ele", "das", "tem", "seu", "sua", "ou",
    "ser", "quando", "muito", "nos", "já", "está", "eu", "também", "só",
    "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem",
    "mesmo", "aos", "ter", "seus", "suas", "num", "numa", "pelos", "pelas",
    "vou", "tá", "tô", "quer", "fazer", "vou", "sei", "sou", "é", "tem",
    "ah", "eh", "oh", "hmm", "bem", "la", "lá", "cá", "aqui", "ali",
}


def tokenizar(texto: str) -> list[str]:
    texto = texto.lower()
    palavras = re.findall(r"[a-záéíóúâêîôûãõç]+", texto)
    return [p for p in palavras if p not in STOP_PT and len(p) > 2]


def similaridade(texto_a: str, texto_b: str) -> float:
    ta = tokenizar(texto_a)
    tb = tokenizar(texto_b)
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    intersecao = sum(min(ca[w], cb[w]) for w in ca if w in cb)
    return intersecao / max(len(ta), len(tb))


def recuperar_fatos_relevantes(mem: dict, texto_usuario: str, limite: int = 6) -> str:
    c = mem.get("crianca", {})
    fatos = []
    campos_chave = [
        "nome", "idade", "cor_favorita", "bichinho", "brinquedo_favorito",
    ]
    for chave in campos_chave:
        if c.get(chave):
            fatos.append((chave, c[chave]))

    campos_extras = [
        (k, v) for k, v in c.items()
        if k not in campos_chave and v
    ]
    fatos.extend(campos_extras)

    pontuados = []
    for chave, valor in fatos:
        score = similaridade(texto_usuario, f"{chave} {valor}")
        pontuados.append((score, chave, valor))

    pontuados.sort(key=lambda x: x[0], reverse=True)
    selecionados = pontuados[:limite]

    partes = []
    for _, chave, valor in selecionados:
        partes.append(f"{chave}: {valor}")

    if mem.get("ultimos_temas"):
        temas = ", ".join(mem["ultimos_temas"][-3:])
        partes.append(f"ultimos_temas: {temas}")

    if mem.get("historico_jogos"):
        jogos = ", ".join(mem["historico_jogos"][-3:])
        partes.append(f"jogos_recentes: {jogos}")

    return "; ".join(partes) if partes else ""
