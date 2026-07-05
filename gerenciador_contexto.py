
MODEL_CONTEXT_LIMITS = {
    "gemma4:e4b": 131_072,
    "gemma4:t4b": 262_144,
    "gemma3:e1b": 32_768,
    "llama3:8b": 8_192,
    "mistral": 32_768,
}

DEFAULT_BUDGET_RATIOS = {
    "system_prompt": 0.04,
    "findings": 0.06,
    "tool_summary": 0.04,
    "history_recent": 0.25,
    "history_summary": 0.06,
    "current_user": 0.10,
}

COMPACTION_THRESHOLD_TURNS = 10
CONTEXT_WARNING_THRESHOLD = 0.85
CONTEXT_ERROR_THRESHOLD = 0.95
MAX_TOOL_OUTPUT_TOKENS = 500


def estimar_tokens(texto: str) -> int:
    return max(1, len(texto) // 4)


def get_context_limit(model: str) -> int:
    base = model.split(":")[0]
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if model == key or base == key.split(":")[0]:
            return limit
    return 8_192


def calcular_orcamento(model: str) -> dict:
    limite = get_context_limit(model)
    margem_resposta = min(2_000, limite * 0.10)
    disponivel = limite - margem_resposta
    return {
        nome: int(disponivel * ratio)
        for nome, ratio in DEFAULT_BUDGET_RATIOS.items()
    }


def verificar_orcamento(messages: list[dict], model: str, max_tokens_resposta: int = 512) -> dict:
    limite = get_context_limit(model)
    total = sum(estimar_tokens(m.get("content", "")) for m in messages) + max_tokens_resposta
    percentual = total / limite

    status = {
        "total_tokens": total,
        "context_limit": limite,
        "percentual": percentual,
        "acao": "enviar",
    }

    if percentual >= CONTEXT_ERROR_THRESHOLD:
        status["acao"] = "erro"
        status["motivo"] = f"contexto em {percentual:.0%} do limite"
    elif percentual >= CONTEXT_WARNING_THRESHOLD:
        status["acao"] = "compactar"
        status["motivo"] = f"contexto em {percentual:.0%} do limite"
    return status


def aplicar_janela_deslizante(historico: list[dict], limite_tokens: int) -> list[dict]:
    aa = []
    total = 0
    for msg in reversed(historico):
        t = estimar_tokens(msg.get("content", ""))
        if total + t > limite_tokens:
            break
        aa.append(msg)
        total += t
    return list(reversed(aa))


def comprimir_tool_output(nome: str, conteudo: str) -> str:
    if nome == "buscar_web":
        linhas = [l for l in conteudo.splitlines() if l.startswith("- ")]
        return "Resultados da pesquisa:\n" + "\n".join(linhas[:3])
    tokens = conteudo.split()
    if len(tokens) > MAX_TOOL_OUTPUT_TOKENS:
        return " ".join(tokens[:MAX_TOOL_OUTPUT_TOKENS]) + "..."
    return conteudo
