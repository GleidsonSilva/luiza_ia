import os
import threading
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_file = open(os.path.join(_script_dir, "debug.log"), "w", buffering=1)
_lock = threading.Lock()

def log(msg):
    linha = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}\n"
    with _lock:
        _file.write(linha)
        _file.flush()
