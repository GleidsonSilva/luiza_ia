import json
import sys
import time

import requests

import assistente as A

OLLAMA = "http://127.0.0.1:11434/api/chat"
GEN = "http://127.0.0.1:11434/api/generate"

# (fala_da_crianca, tools_aceitaveis)  None = nao deve chamar nenhuma tool
CASOS = [
    ("oi bom dia, voce lembra de mim?", {"carregar_memoria"}),
    ("o que a gente pode fazer em floripa amanha?", {"buscar_web"}),
    ("ta frio hoje em sao paulo?", {"buscar_clima"}),
    ("meu nome eh Nicolau", {"lembrar_nome"}),
    ("vamos brincar?", {"sortear_brincadeira"}),
    ("quanto eh 4 mais 4 mais 4?", {"calculadora", None}),
    ("eu gosto de pizza", {None}),
    ("voce gosta de cachorro?", {None}),
]

MODELOS = sys.argv[1:] or ["gemma4:e4b", "qwen3.5:9b", "glm-4.7-flash:q4_K_M", "gemma4:26b"]


def unload(m):
    try:
        requests.post(GEN, json={"model": m, "keep_alive": 0, "prompt": ""}, timeout=15)
    except Exception:
        pass


def chamar(modelo, texto, think):
    msgs = [{"role": "system", "content": A.SYSTEM_PROMPT}, {"role": "user", "content": texto}]
    body = {"model": modelo, "messages": msgs, "tools": A.tools, "stream": False,
            "keep_alive": "30m"}
    if think is not None:
        body["think"] = think
    t0 = time.time()
    r = requests.post(OLLAMA, json=body, timeout=300)
    r.raise_for_status()
    j = r.json()
    tcs = [tc["function"]["name"] for tc in (j.get("message", {}).get("tool_calls") or [])]
    return tcs, time.time() - t0


def avaliar(tcs, aceitaveis):
    if not tcs:
        return None in aceitaveis
    return any(t in aceitaveis for t in tcs)


def main():
    print(f"{'modelo':20} {'think':5} {'caso':42} {'tools':28} {'t':>5}  ok")
    for modelo in MODELOS:
        for think in (False, True):
            # warm-up deste modo
            try:
                chamar(modelo, "oi", think)
            except Exception as e:
                print(f"{modelo:20} think={think} ERRO warmup: {e}")
                continue
            acertos = 0
            for texto, aceit in CASOS:
                try:
                    tcs, dt = chamar(modelo, texto, think)
                except Exception as e:
                    print(f"{modelo:20} {str(think):5} {texto[:42]:42} ERRO {e}")
                    continue
                ok = avaliar(tcs, aceit)
                acertos += ok
                esperado = "/".join(str(x) for x in aceit)
                print(f"{modelo:20} {str(think):5} {texto[:42]:42} {','.join(tcs) or '-':28} {dt:5.1f}  {'OK' if ok else 'X ('+esperado+')'}")
            print(f"{'':20} {str(think):5} -> acertos {acertos}/{len(CASOS)}\n")
        unload(modelo)


if __name__ == "__main__":
    main()
