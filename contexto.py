from datetime import datetime
import threading
import requests
from ddgs import DDGS

DIAS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
MESES = ["janeiro", "fevereiro", "marco", "abril", "maio", "junho",
         "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]

def contexto_hoje() -> str:
    agora = datetime.now()
    periodo = "manha" if agora.hour < 12 else "tarde" if agora.hour < 18 else "noite"
    return (
        f"Hoje eh {DIAS[agora.weekday()]}, {agora.day} de "
        f"{MESES[agora.month - 1]} de {agora.year}. "
        f"Sao {agora.hour:02d}:{agora.minute:02d}. Periodo: {periodo}."
    )

def buscar_clima(cidade: str) -> str:
    try:
        r = requests.get(f"https://wttr.in/{cidade}?format=j1", timeout=10)
        r.raise_for_status()
        data = r.json()
        c = data["current_condition"][0]
        temp = c["temp_C"]
        desc = c["weatherDesc"][0]["value"]
        umid = c["humidity"]
        vento = c["windspeedKmph"]
        return (
            f"Clima em {cidade}: {temp}°C, {desc}, "
            f"umidade {umid}%, vento {vento} km/h."
        )
    except Exception as e:
        return f"Nao consegui buscar o clima de {cidade}. Erro: {e}"

def _buscar_web_worker(consulta: str, resultado: dict, pronto: threading.Event):
    try:
        with DDGS(timeout=8) as ddgs:
            resultado["data"] = list(ddgs.text(consulta, max_results=3, safesearch="moderate"))
    except Exception as e:
        resultado["erro"] = e
    finally:
        pronto.set()


def buscar_web(consulta: str) -> str:
    resultado = {}
    pronto = threading.Event()
    t = threading.Thread(
        target=_buscar_web_worker, args=(consulta, resultado, pronto), daemon=True
    )
    t.start()
    if not pronto.wait(timeout=12):
        return f"A busca por '{consulta}' excedeu o tempo limite."
    if "erro" in resultado:
        return f"Erro ao buscar '{consulta}': {resultado['erro']}"
    results = resultado.get("data", [])
    if not results:
        return f"Nao encontrei nada sobre '{consulta}'."
    linhas = [f"- {r['title']}: {r['href']}" for r in results]
    return f"Resultados da busca por '{consulta}':\n" + "\n".join(linhas)
