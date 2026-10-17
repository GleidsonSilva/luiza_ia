import json, os
from typing import Optional

MEMORIAS_DIR = "memorias"
LIMITE_TOPICOS = 30
LIMITE_JOGOS = 20
LIMITE_TEMAS = 5

_dir_criado = False


def _garantir_dir():
    global _dir_criado
    if not _dir_criado:
        os.makedirs(MEMORIAS_DIR, exist_ok=True)
        _dir_criado = True


def _path(child_id: Optional[str] = None) -> str:
    _garantir_dir()
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


def _dedup_ordenedado(lista: list) -> list:
    vistos = set()
    return [x for x in lista if not (x in vistos or vistos.add(x))]


def salvar(mem: dict, child_id: Optional[str] = None):
    mem["topicos"] = _dedup_ordenedado(mem.get("topicos", []))[-LIMITE_TOPICOS:]
    mem["historico_jogos"] = _dedup_ordenedado(mem.get("historico_jogos", []))[-LIMITE_JOGOS:]
    mem["ultimos_temas"] = _dedup_ordenedado(mem.get("ultimos_temas", []))[-LIMITE_TEMAS:]
    path = _path(child_id)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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
    if mem.get("topicos"):
        topicos = ", ".join(mem["topicos"][-5:])
        partes.append(f"Assuntos que interessam a crianca: {topicos}.")
    if mem.get("ultimos_temas"):
        temas = ", ".join(mem["ultimos_temas"][-3:])
        partes.append(f"Ultimamente voces conversaram sobre: {temas}.")
    if mem.get("historico_jogos"):
        jogos = ", ".join(mem["historico_jogos"][-5:])
        partes.append(f"Brincadeiras que ja fizeram juntos: {jogos}.")
    return " ".join(partes) if partes else ""


def adicionar_fato(mem: dict, chave: str, valor: str, categoria: str = "crianca") -> tuple[dict, str]:
    v = valor.strip()
    valor_final = (v[:1].upper() + v[1:]) if v else v
    if categoria == "crianca":
        mem["crianca"][chave] = valor_final
    elif categoria == "topico":
        mem.setdefault("topicos", []).append(valor_final)
    elif categoria == "jogo":
        mem.setdefault("historico_jogos", []).append(valor_final)
    elif categoria == "tema":
        mem.setdefault("ultimos_temas", []).append(valor_final)
    return mem, valor_final


def remover_fato(mem: dict, chave: Optional[str] = None, categoria: Optional[str] = None) -> dict:
    """Remove um fato da memoria.
    
    Args:
        chave: Nome do campo em 'crianca' (ex: idade, cor_favorita).
               Se None e categoria='crianca', limpa todos os dados pessoais.
        categoria: 'crianca', 'topico', 'jogo' ou 'tema'.
                   Se None, busca em todas as categorias pelo valor.
    """
    if categoria and categoria != "crianca":
        lista_key = {"topico": "topicos", "jogo": "historico_jogos", "tema": "ultimos_temas"}.get(categoria)
        if lista_key and lista_key in mem:
            if chave:
                # Remove itens que contem o texto (case-insensitive)
                mem[lista_key] = [x for x in mem[lista_key] if chave.lower() not in x.lower()]
            else:
                mem[lista_key] = []
    elif categoria == "crianca":
        if chave:
            mem.setdefault("crianca", {}).pop(chave, None)
        else:
            mem["crianca"] = {}
    else:
        # Sem categoria definida: tenta remover de todas
        mem.setdefault("crianca", {}).pop(chave, None)
        for lista_key in ("topicos", "historico_jogos", "ultimos_temas"):
            if chave and lista_key in mem:
                mem[lista_key] = [x for x in mem[lista_key] if chave.lower() not in x.lower()]
    return mem


def carregar_tudo(child_id: Optional[str] = None) -> str:
    mem = carregar(child_id)
    return json.dumps(mem, ensure_ascii=False, indent=2)
