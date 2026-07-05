import json, os, time
from collections import Counter
from typing import Optional

import numpy as np

MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
_IDX_PATH = "memoria_idx.json"
_TOP_K_FATOS = 5
_TOP_K_TURNOS = 4
_EMB_CACHE = None


def _get_model():
    global _EMB_CACHE
    if _EMB_CACHE is None:
        from sentence_transformers import SentenceTransformer
        _EMB_CACHE = SentenceTransformer(MODELO_EMBEDDING, local_files_only=True)
    return _EMB_CACHE


def embed(texto: str) -> np.ndarray:
    modelo = _get_model()
    vetor = modelo.encode(texto, normalize_embeddings=True)
    return np.asarray(vetor, dtype=np.float32)


def similaridade(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def tokenizar(texto: str) -> list[str]:
    texto = texto.lower()
    palavras = __import__("re").findall(r"[a-záéíóúâêîôûãõç]+", texto)
    stop = {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
        "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos",
        "como", "mas", "foi", "ao", "ele", "das", "tem", "seu", "sua", "ou",
        "ser", "quando", "muito", "nos", "já", "está", "eu", "também", "só",
        "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem",
        "mesmo", "aos", "ter", "seus", "suas", "num", "numa", "pelos", "pelas",
        "vou", "tá", "tô", "quer", "fazer", "vou", "sei", "sou", "é", "tem",
        "ah", "eh", "oh", "hmm", "bem", "la", "lá", "cá", "aqui", "ali",
    }
    return [p for p in palavras if p not in stop and len(p) > 2]


def similaridade_jaccard(a: str, b: str) -> float:
    ta, tb = tokenizar(a), tokenizar(b)
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    inter = sum(min(ca[w], cb[w]) for w in ca if w in cb)
    return inter / max(len(ta), len(tb))


class MemoriaVetorial:
    def __init__(self, path: str = _IDX_PATH):
        self.path = path
        self.fatos: list[dict] = []
        self.turnos: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.fatos = data.get("fatos", [])
                self.turnos = data.get("turnos", [])
            except Exception:
                self.fatos = []
                self.turnos = []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({
                    "fatos": self.fatos,
                    "turnos": self.turnos,
                    "updated_at": time.time(),
                }, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    def _serializar_embedding(self, v: np.ndarray) -> list[float]:
        return v.tolist()

    def _desserializar_embedding(self, v) -> np.ndarray:
        return np.asarray(v, dtype=np.float32)

    def indexar_fato(self, chave: str, valor: str):
        texto = f"{chave}: {valor}"
        try:
            vetor = embed(texto)
        except Exception:
            return
        self.fatos.append({
            "texto": texto,
            "embedding": self._serializar_embedding(vetor),
            "tipo": "fato",
            "chave": chave,
            "valor": valor,
            "ts": time.time(),
        })
        self._save()

    def indexar_turno(self, role: str, content: str, turno_idx: int = -1):
        texto = f"{role}: {content}"
        try:
            vetor = embed(texto)
        except Exception:
            return
        self.turnos.append({
            "texto": texto,
            "embedding": self._serializar_embedding(vetor),
            "tipo": "turno",
            "role": role,
            "content": content,
            "turno_idx": turno_idx,
            "ts": time.time(),
        })
        self._save()

    def buscar(self, query: str, k_fatos: int = _TOP_K_FATOS, k_turnos: int = _TOP_K_TURNOS) -> dict:
        try:
            qv = embed(query)
        except Exception:
            return {"fatos": "", "turnos": ""}

        fatos_rankeados = []
        for item in self.fatos:
            v = self._desserializar_embedding(item["embedding"])
            fatos_rankeados.append((similaridade(qv, v), item["texto"]))
        fatos_rankeados.sort(reverse=True)
        fatos_top = [t for _, t in fatos_rankeados[:k_fatos]]

        turnos_rankeados = []
        for item in self.turnos:
            v = self._desserializar_embedding(item["embedding"])
            turnos_rankeados.append((similaridade(qv, v), item["texto"]))
        turnos_rankeados.sort(reverse=True)
        turnos_top = [t for _, t in turnos_rankeados[:k_turnos]]

        return {
            "fatos": "; ".join(fatos_top) if fatos_top else "",
            "turnos": "\n".join(turnos_top) if turnos_top else "",
        }


def recuperar_fatos_relevantes(mem: dict, texto_usuario: str, limite: int = 6) -> str:
    c = mem.get("crianca", {})
    fatos = []
    campos_chave = [
        "nome", "idade", "cor_favorita", "bichinho", "brinquedo_favorito",
    ]
    for chave in campos_chave:
        if c.get(chave):
            fatos.append((chave, c[chave]))
    campos_extras = [(k, v) for k, v in c.items() if k not in campos_chave and v]
    fatos.extend(campos_extras)

    try:
        qv = embed(texto_usuario)
    except Exception:
        texto = "; ".join(f"{k}: {v}" for k, v in fatos[:limite])
        if mem.get("ultimos_temas"):
            temas = ", ".join(mem["ultimos_temas"][-3:])
            texto += f"; ultimos_temas: {temas}"
        if mem.get("historico_jogos"):
            jogos = ", ".join(mem["historico_jogos"][-3:])
            texto += f"; jogos_recentes: {jogos}"
        return texto

    pontuados = []
    for chave, valor in fatos:
        v = embed(f"{chave} {valor}")
        pontuados.append((similaridade(qv, v), chave, valor))
    pontuados.sort(reverse=True)
    selecionados = pontuados[:limite]

    partes = [f"{chave}: {valor}" for _, chave, valor in selecionados]
    if mem.get("ultimos_temas"):
        temas = ", ".join(mem["ultimos_temas"][-3:])
        partes.append(f"ultimos_temas: {temas}")
    if mem.get("historico_jogos"):
        jogos = ", ".join(mem["historico_jogos"][-3:])
        partes.append(f"jogos_recentes: {jogos}")
    return "; ".join(partes) if partes else ""
