import json, os, re
from typing import Optional

MEMORIAS_DIR = "memorias"
LIMITE_TOPICOS = 30
LIMITE_JOGOS = 20
LIMITE_TEMAS = 5


def _path(child_id: Optional[str] = None) -> str:
    os.makedirs(MEMORIAS_DIR, exist_ok=True)
    nome = f"{child_id}.json" if child_id else "_anon.json"
    return os.path.join(MEMORIAS_DIR, nome)


def carregar(child_id: Optional[str] = None) -> dict:
    path = _path(child_id)
    if os.path.exists(path):
        try:
            with open(path) as f:
                conteudo = f.read().strip()
                if not conteudo:
                    raise ValueError("arquivo vazio")
                return json.loads(conteudo)
        except Exception:
            pass
    if child_id is None and os.path.exists("memoria.json"):
        try:
            with open("memoria.json") as f:
                return json.load(f)
        except Exception:
            pass
    return {"crianca": {}, "topicos": [], "ultimos_temas": [], "historico_jogos": []}


def salvar(mem: dict, child_id: Optional[str] = None):
    mem["topicos"] = mem.get("topicos", [])[-LIMITE_TOPICOS:]
    mem["historico_jogos"] = mem.get("historico_jogos", [])[-LIMITE_JOGOS:]
    mem["ultimos_temas"] = mem.get("ultimos_temas", [])[-LIMITE_TEMAS:]
    path = _path(child_id)
    with open(path, "w") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)


def formatar_para_prompt(mem: dict) -> str:
    partes = []
    c = mem.get("crianca", {})
    if c.get("nome"):
        partes.append(f"O nome da crianca eh {c['nome']}.")
    if c.get("idade"):
        partes.append(f"Ela tem {c['idade']} anos.")
    if c.get("cor_favorita"):
        partes.append(f"A cor favorita dela eh {c['cor_favorita']}.")
    if c.get("bichinho"):
        partes.append(f"Ela tem um bichinho: {c['bichinho']}.")
    if c.get("brinquedo_favorito"):
        partes.append(f"O brinquedo favorito dela eh {c['brinquedo_favorito']}.")
    extras = [f"{k}: {v}" for k, v in c.items() if k not in ("nome", "idade", "cor_favorita", "bichinho", "brinquedo_favorito")]
    if extras:
        partes.extend(extras)
    if mem.get("ultimos_temas"):
        temas = ", ".join(mem["ultimos_temas"][-3:])
        partes.append(f"Ultimamente voces conversaram sobre: {temas}.")
    if mem.get("historico_jogos"):
        jogos = ", ".join(mem["historico_jogos"][-5:])
        partes.append(f"Brincadeiras que ja fizeram juntos: {jogos}.")
    return " ".join(partes) if partes else ""


def extrair_fatos(texto: str, mem: dict) -> dict:
    c = mem["crianca"]
    padroes = [
        (r"(?:meu nome eh|me chamo|eu sou o?)\s*([^\s.!?,]+(?:\s[^\s.!?,]+)?)", "nome"),
        (r"(\d+)\s*anos", "idade"),
        (r"(?:gosto|amo|adoro)\s+(?:da?\s+)?cor\s+(\w+)", "cor_favorita"),
        (r"(?:tenho um?|meu bichinho|meu pet|meu cachorro|meu gato)\s*(?:chamado\s+)?(\w+)", "bichinho"),
        (r"(?:meu brinquedo favorito|gosto de brincar com)\s*(.+?)(?:\.|,|$)", "brinquedo_favorito"),
    ]
    for padrao, chave in padroes:
        if chave not in c or not c[chave]:
            m = re.search(padrao, texto, re.IGNORECASE)
            if m:
                c[chave] = m.group(1).strip().capitalize()
    return mem


def extrair_temas(texto: str, mem: dict) -> dict:
    temas_palavras = ["brincar", "jogar", "escola", "amigo", "familia", "filme", "musica",
                      "desenho", "comida", "animal", "futebol", "boneca", "carrinho", "pintar",
                      "ler", "viajar", "natureza", "super-heroi", "princesa", "dinossauro"]
    encontrados = [p for p in temas_palavras if p in texto.lower()]
    for t in encontrados:
        if t not in mem["topicos"]:
            mem["topicos"].append(t)
    mem["ultimos_temas"] = (mem["ultimos_temas"] + encontrados)[-5:]
    return mem


def extrair_jogos(texto: str, mem: dict) -> dict:
    jogos_conhecidos = ["forca", "adivinhacao", "20 perguntas",
                        "historia coletiva", "detetive", "palavra secreta",
                        "cadeia de palavras", "o que sou eu", "verdadeiro ou falso",
                        "rima comigo", "categorias", "stop", "complete a frase"]
    for j in jogos_conhecidos:
        if j in texto.lower() and j not in mem["historico_jogos"]:
            mem["historico_jogos"].append(j)
    return mem


def adicionar_fato(mem: dict, chave: str, valor: str) -> dict:
    mem["crianca"][chave] = valor.strip().capitalize()
    return mem
