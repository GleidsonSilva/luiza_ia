from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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

def buscar_web(consulta: str) -> str:
    try:
        with DDGS(timeout=5) as ddgs:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(ddgs.text, consulta, max_results=3, safesearch="moderate")
                results = future.result(timeout=10)
        if not results:
            return f"Nao encontrei nada sobre '{consulta}'."
        linhas = [f"- {r['title']}: {r['href']}" for r in results]
        return f"Resultados da busca por '{consulta}':\n" + "\n".join(linhas)
    except TimeoutError:
        return f"A busca por '{consulta}' excedeu o tempo limite."
    except Exception as e:
        return f"Erro ao buscar '{consulta}': {e}"
