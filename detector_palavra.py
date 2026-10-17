import os
import re
import threading
import time
import tempfile
import numpy as np
import soundfile as sf
import mlx_whisper

import assistente
import debug

os.environ["HF_HUB_OFFLINE"] = "1"

SR = 16000

WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(SR * WINDOW_SECONDS)
RMS_THRESHOLD = 0.035
TRIM_THRESHOLD = 0.006
_COOLDOWN_SECS = 3.5

_WAKE_RE = re.compile(
    r'\bluiza\b',
    re.IGNORECASE
)

_running = False
_callback = None
_last_trigger = 0.0
_thread = None


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    mask = np.abs(audio) > TRIM_THRESHOLD
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return audio[:0]  # empty
    return audio[idxs[0]:idxs[-1] + 1]


def _transcrever_checar(audio: np.ndarray) -> bool:
    clip = np.clip(audio, -1.0, 1.0)
    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak > 0:
        clip = clip / peak * 0.95

    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, clip, SR)
    try:
        with assistente.whisper_lock:
            result = mlx_whisper.transcribe(
                tmp,
                language="pt",
                path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                initial_prompt="luiza ola luiza ei luiza"
            )
        texto = (result.get("text") or "").strip().lower()
        debug.log(f"detector_palavra: whisper='{texto}'")
        match = bool(_WAKE_RE.search(texto))
        debug.log(f"detector_palavra: match={match}")
        return match
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def _is_idle() -> bool:
    if _callback is None:
        return False
    return bool(_callback(trigger=False))


def _run():
    global _running, _last_trigger

    debug.log("detector_palavra: thread iniciada")

    while _running:
        try:
            idle = _is_idle()
            debug.log(f"detector_palavra: ciclo idle={idle}")

            if not idle:
                time.sleep(0.5)
                continue

            if not assistente.stream or not assistente.stream.active:
                debug.log("detector_palavra: stream inativo")
                time.sleep(1.0)
                continue

            with assistente.buf_lock:
                buf_len = len(assistente.buf)
                if buf_len == 0:
                    debug.log("detector_palavra: buf vazio")
                    time.sleep(0.5)
                    continue
                chunk_size = len(assistente.buf[0])
                chunk_count_take = WINDOW_SAMPLES // chunk_size + 1
                recent = assistente.buf[-chunk_count_take:]
                recent_snapshot_start = buf_len - len(recent)

            if len(recent) < 16:
                debug.log(f"detector_palavra: poucos chunks {len(recent)}")
                time.sleep(0.3)
                continue

            # Calcula RMS medio dos chunks
            rms_values = [
                float(np.sqrt(np.mean(np.asarray(chunk, dtype=np.float32) ** 2)))
                for chunk in recent
            ]
            avg_rms = float(np.mean(rms_values)) if rms_values else 0.0
            max_rms = float(max(rms_values)) if rms_values else 0.0
            debug.log(f"detector_palavra: rms avg={avg_rms:.5f} max={max_rms:.5f} "
                       f"chunks={len(recent)} chunk_sz={chunk_size}")

            if avg_rms <= RMS_THRESHOLD and max_rms <= RMS_THRESHOLD * 2:
                time.sleep(WINDOW_SECONDS * 0.4)
                continue

            # Captura a janela de audio
            with assistente.buf_lock:
                end = len(assistente.buf)
                start = max(0, recent_snapshot_start)
                full_chunks = assistente.buf[start:end]

            if not full_chunks:
                debug.log("detector_palavra: chunks sumiram entre locks")
                time.sleep(0.3)
                continue

            audio_list = []
            total_samples = 0
            for c in full_chunks:
                arr = np.asarray(c, dtype=np.float32).ravel()
                audio_list.append(arr)
                total_samples += len(arr)
            audio = np.concatenate(audio_list)

            # Mantem apenas os ultimos WINDOW_SAMPLES
            if len(audio) > WINDOW_SAMPLES:
                audio = audio[-WINDOW_SAMPLES:]

            # Remove ruido de fundo (opcional: normaliza e tira DC)
            audio = audio - np.mean(audio)

            # Trima silencio
            audio_trimmed = _trim_silence(audio)
            if len(audio_trimmed) < int(SR * 0.25):
                debug.log(f"detector_palavra: audio curto apos trim {len(audio_trimmed)} samples")
                del audio
                del audio_trimmed
                time.sleep(0.3)
                continue

            debug.log(f"detector_palavra: audio bruto={len(audio)} ({len(audio)/SR:.1f}s) "
                       f"trimado={len(audio_trimmed)} ({len(audio_trimmed)/SR:.1f}s)")

            # Cooldown
            now = time.monotonic()
            if now - _last_trigger < _COOLDOWN_SECS:
                debug.log(f"detector_palavra: cooldown {now - _last_trigger:.1f}s")
                del audio, audio_trimmed
                time.sleep(0.5)
                continue

            # Roda Whisper
            debug.log(f"detector_palavra: enviando {len(audio_trimmed)/SR:.2f}s para whisper...")
            match = _transcrever_checar(audio_trimmed)
            del audio, audio_trimmed

            if match:
                _last_trigger = time.monotonic()
                debug.log("detector_palavra: *** WAKE WORD DETECTED ***")
                if _callback:
                    _callback(trigger=True)
                time.sleep(1.0)
            else:
                time.sleep(0.5)

        except Exception as e:
            import traceback, io
            tb = io.StringIO()
            traceback.print_exc(file=tb)
            debug.log(f"detector_palavra error: {e}\n{tb.getvalue()}")
            time.sleep(1.0)


def start(callback):
    global _running, _callback, _thread
    if _thread is not None and _thread.is_alive():
        debug.log("detector_palavra: already running")
        return
    _callback = callback
    _running = True
    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    debug.log("detector_palavra: started")


def stop():
    global _running
    _running = False
    debug.log("detector_palavra: stopped")