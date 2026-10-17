import sys
import threading
import time
import traceback

import debug

HEARTBEAT_THRESHOLD = 5.0
STATE_INTERVAL = 15.0

_lock = threading.Lock()
_heartbeat = time.monotonic()
_state_provider = None
_started = False


def beat() -> None:
    """Marca que a thread principal (event loop) esta viva."""
    global _heartbeat
    with _lock:
        _heartbeat = time.monotonic()


def start(state_provider=None) -> None:
    """Inicia o watchdog. state_provider deve retornar um resumo do estado."""
    global _started, _state_provider, _heartbeat
    if _started:
        return
    _started = True
    _state_provider = state_provider
    with _lock:
        _heartbeat = time.monotonic()
    threading.Thread(target=_loop, daemon=True, name="watchdog").start()
    debug.log("watchdog: iniciado")


def _estado() -> str:
    if _state_provider is None:
        return ""
    try:
        return str(_state_provider())
    except Exception as e:
        return f"<erro estado: {e}>"


def _dump_threads() -> None:
    frames = sys._current_frames()
    nomes = {t.ident: t.name for t in threading.enumerate()}
    for tid, frame in frames.items():
        nome = nomes.get(tid, "?")
        pilha = "".join(traceback.format_stack(frame))
        debug.log(f"watchdog: thread {nome} (id={tid}):\n{pilha}")


def _loop() -> None:
    alertado = False
    ultimo_estado = time.monotonic()
    while True:
        time.sleep(1.0)
        with _lock:
            decorrido = time.monotonic() - _heartbeat
        if decorrido >= HEARTBEAT_THRESHOLD:
            if not alertado:
                debug.log(
                    f"watchdog: UI sem heartbeat ha {decorrido:.1f}s - "
                    f"dump de threads. estado={_estado()}"
                )
                _dump_threads()
                alertado = True
        else:
            alertado = False
        if time.monotonic() - ultimo_estado >= STATE_INTERVAL:
            ultimo_estado = time.monotonic()
            debug.log(f"watchdog: estado={_estado()}")
