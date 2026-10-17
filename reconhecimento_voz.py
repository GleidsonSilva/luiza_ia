import os, json, threading, time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

SR = 16000
DEBUG = None

def _get_debug():
    global DEBUG
    if DEBUG is None:
        try:
            import debug
            DEBUG = debug
        except Exception:
            pass
    return DEBUG

def debug_log(msg: str) -> None:
    d = _get_debug()
    if d:
        d.log(msg)
    else:
        print(f"[reconhecimento_voz] {msg}")

MODELO_ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ecapa_voxceleb.onnx")
FBANK_MATRIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fbank-80x201-f32.bin")
ASSINATURAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memorias", "assinaturas_voz.json")
THRESHOLD = 0.70
MIN_RMS_ENROLLMENT = 0.01
MAX_BUFFER_SEGUNDOS = 5
MAX_EMBEDDINGS_POR_USUARIO = 5  # pool de embeddings por crianca

N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 80

_modelo = None
_fbank_matrix = None
_modelo_lock = threading.Lock()
_assinaturas = {}
_assinaturas_carregadas = False
_assinaturas_lock = threading.Lock()
_interacoes_confirmadas = {}
_interacoes_lock = threading.Lock()


def inicializar_modelo() -> bool:
    global _modelo, _fbank_matrix
    if not ONNX_AVAILABLE:
        debug_log("reconhecimento_voz: onnxruntime nao instalado, voice ID desativado")
        return False
    with _modelo_lock:
        if _modelo is not None and _fbank_matrix is not None:
            return True
        if not os.path.exists(MODELO_ONNX):
            debug_log(f"reconhecimento_voz: modelo ONNX nao encontrado em {MODELO_ONNX}")
            return False
        if not os.path.exists(FBANK_MATRIX):
            debug_log(f"reconhecimento_voz: matriz fbank nao encontrada em {FBANK_MATRIX}")
            return False
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            _modelo = ort.InferenceSession(MODELO_ONNX, sess_options=sess_options)
            _fbank_matrix = np.fromfile(FBANK_MATRIX, dtype=np.float32).reshape(N_MELS, N_FFT // 2 + 1)
            debug_log(f"reconhecimento_voz: modelo ONNX + fbank carregados")
            return True
        except Exception as e:
            debug_log(f"reconhecimento_voz: erro ao carregar modelo: {e}")
            _modelo = None
            _fbank_matrix = None
            return False


def modelo_carregado() -> bool:
    return _modelo is not None and _fbank_matrix is not None


def _compute_fbank(audio: np.ndarray) -> np.ndarray:
    audio = np.pad(audio, (WIN_LENGTH // 2, WIN_LENGTH // 2), mode='constant')
    frames = []
    for i in range(0, len(audio) - WIN_LENGTH + 1, HOP_LENGTH):
        frames.append(audio[i:i + WIN_LENGTH])
    frames = np.array(frames, dtype=np.float32)
    
    window = np.hanning(WIN_LENGTH).astype(np.float32)
    stft = np.fft.rfft(frames * window, n=N_FFT, axis=1)
    magnitudes = np.abs(stft) ** 2
    
    mel_spectrogram = np.dot(magnitudes, _fbank_matrix.T)
    log_mel = np.log(np.maximum(mel_spectrogram, 1e-10))
    
    return log_mel.astype(np.float32)


def extrair_embedding(audio: np.ndarray) -> Optional[np.ndarray]:
    if _modelo is None or _fbank_matrix is None:
        return None
    try:
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        audio = audio.astype(np.float32)
        if len(audio) < SR * 0.5:
            debug_log(f"reconhecimento_voz: audio muito curto ({len(audio)} amostras)")
            return None
        
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms > 0.01:
            peak = np.max(np.abs(audio))
            if peak > 0:
                gain = min(10.0, 0.95 / peak)
                audio = audio * gain
        
        fbank = _compute_fbank(audio)
        if fbank.shape[0] < 10:
            debug_log("reconhecimento_voz: poucos frames de fbank")
            return None
        
        features = np.expand_dims(fbank, axis=0).astype(np.float32)
        feature_lens = np.array([fbank.shape[0]], dtype=np.float32)
        
        input_feed = {
            "features": features,
            "feature_lens": feature_lens,
        }
        outputs = _modelo.run(None, input_feed)
        embedding = outputs[0][0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        debug_log(f"reconhecimento_voz: erro ao extrair embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def _carregar_assinaturas() -> dict:
    global _assinaturas, _assinaturas_carregadas
    if _assinaturas_carregadas:
        return _assinaturas
    with _assinaturas_lock:
        if _assinaturas_carregadas:
            return _assinaturas
        _assinaturas = {}
        if os.path.exists(ASSINATURAS_PATH):
            try:
                with open(ASSINATURAS_PATH, "r") as f:
                    dados = json.load(f)
                raw = dados.get("assinaturas", {})
                # Migrar formato antigo (embedding unico) para pool de embeddings
                for cid, val in raw.items():
                    entry = dict(val)
                    if "embedding" in entry and "embeddings" not in entry:
                        entry["embeddings"] = [entry.pop("embedding")]
                    elif "embeddings" not in entry:
                        entry["embeddings"] = []
                    _assinaturas[cid] = entry
            except Exception as e:
                debug_log(f"reconhecimento_voz: erro ao carregar assinaturas: {e}")
                _assinaturas = {}
        _assinaturas_carregadas = True
        return _assinaturas


def _salvar_assinaturas() -> None:
    try:
        os.makedirs(os.path.dirname(ASSINATURAS_PATH), exist_ok=True)
        dados = {"assinaturas": _assinaturas}
        tmp = ASSINATURAS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ASSINATURAS_PATH)
    except Exception as e:
        debug_log(f"reconhecimento_voz: erro ao salvar assinaturas: {e}")


def _melhor_score_contra_pool(emb_atual: np.ndarray, assinaturas: dict) -> tuple[Optional[str], float]:
    """Compara embedding contra pool de embeddings de cada crianca.
    Retorna (child_id, score) do melhor match acima do threshold."""
    melhor_child_id = None
    melhor_score = THRESHOLD
    for child_id, dados in assinaturas.items():
        pool = dados.get("embeddings", [])
        if not pool:
            continue
        for emb_raw in pool:
            emb_salvo = np.array(emb_raw, dtype=np.float32)
            if len(emb_salvo) == 0:
                continue
            score = cosine_similarity(emb_atual, emb_salvo)
            if score > melhor_score:
                melhor_score = score
                melhor_child_id = child_id
    if melhor_child_id:
        debug_log(f"reconhecimento_voz: MATCH child_id={melhor_child_id[:12]}... score={melhor_score:.3f}")
    return melhor_child_id, melhor_score


def identificar_crianca(audio: np.ndarray) -> Optional[str]:
    if _modelo is None or _fbank_matrix is None:
        return None
    if audio is None or len(audio) < SR * 0.5:
        return None
    try:
        emb_atual = extrair_embedding(audio)
        if emb_atual is None:
            return None
        assinaturas = _carregar_assinaturas()
        if not assinaturas:
            return None
        melhor_child_id, _ = _melhor_score_contra_pool(emb_atual, assinaturas)
        if melhor_child_id:
            _registrar_interacao(melhor_child_id)
        return melhor_child_id
    except Exception as e:
        debug_log(f"reconhecimento_voz: erro na identificacao: {e}")
        import traceback
        traceback.print_exc()
        return None


def salvar_assinatura(child_id: str, nome: str, audio: np.ndarray, baixa_confianca: bool = False) -> bool:
    global _assinaturas
    if _modelo is None or _fbank_matrix is None:
        debug_log("reconhecimento_voz: modelo nao carregado, nao foi possivel salvar assinatura")
        return False
    try:
        emb = extrair_embedding(audio)
        if emb is None:
            return False
        assinaturas = _carregar_assinaturas()
        dados_existente = assinaturas.get(child_id, {})
        pool = dados_existente.get("embeddings", [])
        # Adiciona novo embedding ao pool, mantendo no maximo MAX_EMBEDDINGS_POR_USUARIO
        pool.append(emb.tolist())
        if len(pool) > MAX_EMBEDDINGS_POR_USUARIO:
            pool = pool[-MAX_EMBEDDINGS_POR_USUARIO:]  # mantem os mais recentes
        assinaturas[child_id] = {
            "embeddings": pool,
            "nome": nome,
            "criado_em": dados_existente.get("criado_em", time.strftime("%Y-%m-%dT%H:%M:%S")),
            "atualizado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "amostras_usadas": dados_existente.get("amostras_usadas", 0) + 1,
            "baixa_confianca": baixa_confianca,
        }
        _assinaturas = assinaturas
        _salvar_assinaturas()
        debug_log(f"reconhecimento_voz: assinatura salva para {child_id[:12]}... ({nome}) pool_size={len(pool)} baixa_confianca={baixa_confianca}")
        return True
    except Exception as e:
        debug_log(f"reconhecimento_voz: erro ao salvar assinatura: {e}")
        return False


def atualizar_assinatura(child_id: str, nome: str, audio: np.ndarray) -> bool:
    return salvar_assinatura(child_id, nome, audio, baixa_confianca=False)


def obter_assinatura(child_id: str) -> Optional[dict]:
    assinaturas = _carregar_assinaturas()
    return assinaturas.get(child_id)


def marcar_assinatura_confirmada(child_id: str) -> None:
    global _assinaturas
    assinaturas = _carregar_assinaturas()
    if child_id in assinaturas:
        assinaturas[child_id]["baixa_confianca"] = False
        _assinaturas = assinaturas
        _salvar_assinaturas()



def obter_nome_por_id(child_id: str) -> str:
    """Retorna o nome salvo para um child_id, ou string vazia."""
    assin = obter_assinatura(child_id)
    if assin:
        return assin.get("nome", "")
    return ""


def obter_qtd_embeddings(child_id: str) -> int:
    """Retorna quantas amostras de voz existem no pool desta crianca."""
    assin = obter_assinatura(child_id)
    if not assin:
        return 0
    return len(assin.get("embeddings", []))


def _registrar_interacao(child_id: str) -> None:
    with _interacoes_lock:
        _interacoes_confirmadas[child_id] = _interacoes_confirmadas.get(child_id, 0) + 1


def verificar_re_enrollment(child_id: str, nome: str) -> bool:
    with _interacoes_lock:
        count = _interacoes_confirmadas.get(child_id, 0)
    if count < 3:
        return False
    assinatura = obter_assinatura(child_id)
    if not assinatura:
        return False
    if assinatura.get("baixa_confianca", False):
        debug_log(f"reconhecimento_voz: re-enrollment automatico para {child_id[:12]}... ({count} interacoes)")
        return True
    return False
