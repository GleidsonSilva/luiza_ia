import sys, os
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
_file = open(os.path.join(_script_dir, "debug.log"), "w", buffering=1)

def log(msg):
    _file.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}\n")
    _file.flush()
