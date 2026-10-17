import json
import sys
import time

import requests

import assistente
from assistente import SYSTEM_PROMPT, tools

OLLAMA = "http://127.0.0.1:11434/api/chat"
GEN = "http://127.0.0.1:11434/api/generate"

MODELOS_PADRAO = [
    "gemma4:e4b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "lfm2.5:8b",
    "gpt-oss:20b",
    "glm-4.7-flash:q4_K_M",
    "gemma4:26b",
]

CASOS = [
    ("meu nome eh Ana", "lembrar_nome"),
    ("quanto eh 3 mais 4?", "calculadora"),
    ("oi Luiza tudo bem?", None),
    ("vamos brincar de adivinhacao", None),
]


def unload(modelo):
    try:
        requests.post(GEN, json={"model": modelo, "keep_alive": 0, "prompt": ""}, timeout=15)
    except Exception:
        pass


def warmup(modelo):
    t0 = time.time()
    r = requests.post(OLLAMA, json={
        "model": modelo,
        "messages": [{"role": "user", "content": "oi"}],
        "stream": False,
        "options": {"num_predict": 1},
    }, timeout=600)
    r.raise_for_status()
    return time.time() - t0


def rodar(modelo, texto):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": texto},
    ]
    body = {"model": modelo, "messages": msgs, "tools": tools, "stream": True}
    t0 = time.time()
    first = None
    content = []
    nomes_tools = []
    with requests.post(OLLAMA, json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message", {}) or {}
            c = msg.get("content")
            tcs = msg.get("tool_calls")
            if first is None and (c or tcs):
                first = time.time() - t0
            if c:
                content.append(c)
            if tcs:
                for tc in tcs:
                    nome = (tc.get("function") or {}).get("name")
                    if nome:
                        nomes_tools.append(nome)
            if obj.get("done"):
                break
    total = time.time() - t0
    texto_resp = "".join(content).strip().replace("\n", " ")
    return first, total, nomes_tools, texto_resp


def main():
    modelos = sys.argv[1:] or MODELOS_PADRAO
    print(f"{'modelo':22} {'load(s)':>7} {'caso':28} {'ttft':>6} {'total':>6}  tool resp")
    for m in modelos:
        try:
            load = warmup(m)
        except Exception as e:
            print(f"{m:22} ERRO warmup: {e}")
            continue
        print(f"{m:22} {load:7.1f}  (modelo carregado)")
        for texto, esperado in CASOS:
            try:
                first, total, nomes, resp = rodar(m, texto)
            except Exception as e:
                print(f"{'':22} {'':7} {texto[:28]:28} ERRO: {e}")
                continue
            tool_str = ",".join(nomes) if nomes else "-"
            ok = ""
            if esperado and esperado in nomes:
                ok = "OK"
            elif esperado and esperado not in nomes:
                ok = "FALTOU"
            ttft = f"{first:.2f}" if first is not None else "  -  "
            print(f"{'':22} {'':7} {texto[:28]:28} {ttft:>6} {total:6.2f}  [{tool_str}] {ok} | {resp[:60]}")
        unload(m)
    print("\nObs: ttft = tempo ate o primeiro token (percepcao de resposta).")


if __name__ == "__main__":
    main()
