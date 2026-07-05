import json
import os
from datetime import datetime
from typing import Optional

HISTORICO_DIR = "historico"
SESSION_ATIVA_PATH = os.path.join(HISTORICO_DIR, "session_ativa.json")


def _garantir_dirs():
    os.makedirs(HISTORICO_DIR, exist_ok=True)


def _caminho(session_id: str) -> str:
    return os.path.join(HISTORICO_DIR, f"{session_id}.jsonl")


def _ler_linhas(session_id: str) -> list[dict]:
    path = _caminho(session_id)
    if not os.path.exists(path):
        return []
    linhas = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for num, linha in enumerate(f, 1):
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    obj = json.loads(linha)
                    if isinstance(obj, dict) and "timestamp" in obj:
                        linhas.append(obj)
                except json.JSONDecodeError as e:
                    import debug as _debug
                    try:
                        _debug.log(f"historico_jsonl_invalido session={session_id} linha={num} erro={e}")
                    except Exception:
                        pass
    except Exception:
        pass
    return linhas


def _escrever_linha(session_id: str, obj: dict):
    _garantir_dirs()
    path = _caminho(session_id)
    linha = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def _gerar_session_id() -> str:
    import uuid
    return uuid.uuid4().hex


def nova_sessao(child_id: Optional[str] = None) -> str:
    _garantir_dirs()
    session_id = _gerar_session_id()
    _escrever_linha(session_id, {
        "timestamp": datetime.now().isoformat(),
        "role": "system",
        "content": "inicio_sessao",
        "child_id": child_id,
        "session_id": session_id,
    })
    marcar_ativa(session_id)
    return session_id


def carregar(session_id: str) -> list[dict]:
    return _ler_linhas(session_id)


def salvar_mensagem(session_id: str, role: str, content: str, child_id: Optional[str] = None):
    obj = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content,
        "child_id": child_id,
        "session_id": session_id,
    }
    _escrever_linha(session_id, obj)


def salvar_lote(session_id: str, mensagens: list[dict]):
    _garantir_dirs()
    path = _caminho(session_id)
    linhas = []
    for msg in mensagens:
        msg.setdefault("timestamp", datetime.now().isoformat())
        msg.setdefault("session_id", session_id)
        linhas.append(json.dumps(msg, ensure_ascii=False))
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


def listar_sessoes(child_id: Optional[str] = None) -> list[str]:
    _garantir_dirs()
    try:
        nomes = [f for f in os.listdir(HISTORICO_DIR) if f.endswith(".jsonl") and f != "session_ativa.json"]
        ids = [n.replace(".jsonl", "") for n in nomes]
        if not child_id:
            return ids
        ids2 = []
        for sid in ids:
            for obj in _ler_linhas(sid):
                cid = obj.get("child_id")
                if isinstance(cid, str) and cid and (sid not in ids2):
                    ids2.append(sid)
        return ids2
    except Exception:
        return []


def carregar_ativa() -> tuple[str, list[dict]]:
    if os.path.exists(SESSION_ATIVA_PATH):
        try:
            with open(SESSION_ATIVA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                session_id = data.get("session_id")
                if session_id and os.path.exists(_caminho(session_id)):
                    return session_id, carregar(session_id)
        except Exception:
            pass
    session_id = nova_sessao()
    return session_id, []


def marcar_ativa(session_id: str):
    _garantir_dirs()
    with open(SESSION_ATIVA_PATH, "w", encoding="utf-8") as f:
        json.dump({"session_id": session_id}, f)


def buscar(query: str = "", child_id: Optional[str] = None, max_results: int = 50, ultimas_n_sessoes: int = 5) -> list[dict]:
    candidatos = listar_sessoes()
    if child_id:
        candidatos = listar_sessoes(child_id=child_id)
    try:
        candidatos.sort(key=lambda sid: os.path.getmtime(_caminho(sid)), reverse=True)
    except Exception:
        pass
    candidatos = candidatos[:ultimas_n_sessoes]
    q = query.strip().lower()
    resultados = []
    for sid in candidatos:
        linhas = _ler_linhas(sid)
        for obj in linhas:
            if obj.get("role") not in ("user", "assistant"):
                continue
            if child_id:
                if not obj.get("child_id") or obj["child_id"] != child_id:
                    continue
            if q and q not in obj.get("content", "").lower():
                continue
            resultados.append(obj)
    if len(resultados) > max_results:
        resultados = resultados[:max_results]
    resultados.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return resultados
