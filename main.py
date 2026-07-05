import os, sys, importlib

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)


def _check_ollama():
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        modelos = [m["name"] for m in r.json().get("models", [])]
        if "gemma4:e4b" not in modelos:
            print("Erro: modelo 'gemma4:e4b' nao encontrado no Ollama.")
            print("  Baixe com: ollama pull gemma4:e4b")
            return False
        return True
    except requests.ConnectionError:
        print("Erro: Ollama nao esta rodando em http://localhost:11434")
        print("  Inicie com: ollama serve")
        return False
    except Exception as e:
        print(f"Erro ao conectar no Ollama: {e}")
        return False


def _check_microfone():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if not any(d["max_input_channels"] > 0 for d in devices):
            print("Erro: nenhum microfone encontrado.")
            return False
        return True
    except Exception as e:
        print(f"Erro ao verificar microfone: {e}")
        return False


def _check_tts():
    for f in ("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin"):
        if not os.path.exists(f):
            print(f"Erro: arquivo TTS nao encontrado: {f}")
            return False
    return True


def _check_embedding():
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True
        )
        return True
    except Exception:
        print("Modelo de embedding nao encontrado em cache. Tentando baixar...")
        try:
            old = os.environ.pop("HF_HUB_OFFLINE", None)
            from sentence_transformers import SentenceTransformer
            SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            if old is not None:
                os.environ["HF_HUB_OFFLINE"] = old
            print("  Modelo de embedding baixado com sucesso.")
            return True
        except Exception as e:
            print(f"  Falha ao baixar modelo de embedding: {e}")
            return False


def verificar_dependencias():
    erros = []
    checks = [
        ("Modelos de TTS (Kokoro)", _check_tts),
        ("Ollama + modelo gemma4:e4b", _check_ollama),
        ("Microfone", _check_microfone),
        ("Modelo de embedding (sentence-transformers)", _check_embedding),
    ]
    for nome, fn in checks:
        if not fn():
            erros.append(nome)

    if erros:
        print("\nDependencias com problema:")
        for e in erros:
            print(f"  - {e}")
        print("\nCorrija os erros acima e tente novamente.")
        sys.exit(1)


if __name__ == "__main__":
    verificar_dependencias()
    from interface import LuizaApp
    LuizaApp().run()
