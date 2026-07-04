import sys, os, tempfile, select, time
import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper
import requests
from kokoro_onnx import Kokoro
import memoria

SR = 16000
MAX_DURATION = 15
GAIN = 8.0
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
MAX_HISTORICO = 5

os.chdir(os.path.dirname(os.path.abspath(__file__)))
kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

SYSTEM_PROMPT = """### PERSONA
Voce eh a Luiza, uma amiga virtual de 8 anos.
Voce adora brincar, fazer perguntas e inventar historias.
NUNCA se comporte como assistente ou robo.
Nao use emojis ou simbolos.

### COMO FALAR
- Frases curtas (maximo 15 palavras cada)
- Vocabulario simples, como uma crianca de 8 anos
- Uma ideia por vez
- Nao de listas ou topicos numerados
- Termine com uma pergunta para a crianca continuar conversando

### COMO BRINCAR
- Toda brincadeira funciona em turnos: voce fala, a crianca responde, voce responde de novo
- Explique a brincadeira em 1 frase antes de comecar
- Sempre termine sua vez com uma pergunta ou esperando a resposta da crianca
- Nao fale por ela nem responda pela crianca
- Se ela errar, de uma dica em vez de dar a resposta
- Nao finja que a crianca respondeu — aguarde a resposta de verdade
- Cadeia de Palavras: voce fala uma palavra, crianca responde com outra que comeca com a ultima silaba
- Rima Comigo: voce fala uma palavra, crianca responde com outra que rima
- Complete a Frase: voce comeca uma frase, crianca termina de um jeito criativo
- Historia Coletiva: alternem partes da historia, cada um fala um pedaco

### BRINCADEIRAS QUE VOCE CONHECE
Forca, adivinhacao, 20 perguntas, cadeia de palavras,
o que sou eu, verdadeiro ou falso, rima comigo,
categorias/stop, complete a frase, historia coletiva,
detetive, palavra secreta

### REGRAS
- Responda com NO MAXIMO 3 frases curtas
- Nao explique as regras do jogo a menos que a crianca peca
- Se a crianca estiver triste ou com medo, acolha com carinho
- Nao finja ser outra pessoa ou personagem"""

mem = memoria.carregar()
historico = []

PITCH_SHIFT = 5.0  # semitons (positivo = mais agudo, infantil)

def pitch_shift(x: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    ratio = 2.0 ** (n_semitones / 12.0)
    fft_size = 2048
    hop_size = fft_size // 4
    original_len = len(x)
    x = np.pad(x, (0, fft_size))
    win = np.hanning(fft_size)

    n_frames = 1 + (len(x) - fft_size) // hop_size
    X = np.zeros((fft_size, n_frames), dtype=complex)
    for t in range(n_frames):
        start = t * hop_size
        X[:, t] = np.fft.fft(x[start:start + fft_size] * win)

    mag = np.abs(X)
    phase = np.unwrap(np.angle(X), axis=1)
    delta_phase = np.diff(phase, axis=1)
    expected = (2 * np.pi * np.arange(fft_size)[:, None] * hop_size / fft_size)
    inst_freq = (2 * np.pi * np.arange(fft_size)[:, None] / fft_size +
                 (delta_phase - expected[:, :n_frames-1]) / hop_size)

    Y = np.zeros_like(X, dtype=complex)
    for k in range(fft_size):
        ka = k / ratio
        if ka >= fft_size:
            continue
        kl, kr = int(ka), min(int(ka) + 1, fft_size - 1)
        f = ka - kl
        mag_k = (1 - f) * mag[kl] + f * mag[kr]
        f_synth = ((1 - f) * inst_freq[kl] + f * inst_freq[kr]) * ratio
        ph = np.zeros(n_frames)
        ph[0] = (1 - f) * phase[kl, 0] + f * phase[kr, 0]
        for t in range(1, n_frames):
            ph[t] = ph[t-1] + f_synth[t-1] * hop_size
        Y[k] = mag_k * np.exp(1j * ph)

    y = np.zeros(len(x))
    for t in range(n_frames):
        frame = np.fft.ifft(Y[:, t]).real * win
        s = t * hop_size
        y[s:s + fft_size] += frame

    y = y[:original_len]
    peak = np.max(np.abs(y))
    return (y / peak * 0.95).astype(np.float32) if peak > 0 else y.astype(np.float32)

def trim_silence(audio: np.ndarray, threshold=0.005) -> np.ndarray:
    mask = np.abs(audio) > threshold
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return audio
    return audio[idxs[0]:idxs[-1] + 1]

def transcrever(audio: np.ndarray) -> str:
    audio = trim_silence(audio)
    if len(audio) < SR * 0.3:
        return ""
    audio = audio * GAIN
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95

    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio, SR)

    r = mlx_whisper.transcribe(
        tmp, language="pt",
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        initial_prompt="ola vamos brincar qual e o seu nome voce gosta de jogar que historia legal"
    )
    os.remove(tmp)
    return r["text"].strip()

def perguntar(texto: str) -> str:
    global historico
    memorias = memoria.formatar_para_prompt(mem)
    system = SYSTEM_PROMPT
    if memorias:
        system += f"\n\nCoisas que voce sabe sobre seu amigo:\n{memorias}"

    messages = [{"role": "system", "content": system}]
    messages.extend(historico)
    messages.append({"role": "user", "content": texto})

    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False
    })
    resposta = r.json()["message"]["content"].strip()

    historico.append({"role": "user", "content": texto})
    historico.append({"role": "assistant", "content": resposta})
    if len(historico) > MAX_HISTORICO * 2:
        historico = historico[-(MAX_HISTORICO * 2):]

    memoria.extrair_fatos(texto + " " + resposta, mem)
    memoria.extrair_temas(texto + " " + resposta, mem)
    memoria.extrair_jogos(texto + " " + resposta, mem)
    memoria.salvar(mem)

    return resposta

def falar(texto: str):
    samples, sr = kokoro.create(texto, voice="pf_dora", speed=1.0, lang="pt-br")
    samples = pitch_shift(samples, sr, PITCH_SHIFT)
    sd.play(samples, sr)
    sd.wait()

def key_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False

nome_crianca = mem.get("crianca", {}).get("nome", "")
if nome_crianca:
    print(f"Ola {nome_crianca}! A Luiza ja esta aqui!")
else:
    print("Ola! Eu sou a Luiza! Qual o seu nome?")

print("ENTER para falar, ENTER de novo para parar")

buf = []
stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda i, f, t, s: buf.append(i.copy()))
stream.start()

while True:
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
    input("\nPressione ENTER para falar com a Luiza...")

    idx = len(buf)
    print("Falando... (ENTER para parar)", end="", flush=True)
    inicio = time.time()

    while True:
        time.sleep(0.05)
        if key_pressed():
            print("")
            break
        if time.time() - inicio > MAX_DURATION:
            print(" (tempo maximo)")
            break

    full = np.concatenate(buf[idx:])

    print("Entendendo...")
    texto = transcrever(full)

    if not texto:
        print("Nao entendi. Pode falar de novo?")
        continue

    print(f"  Voce: {texto}")

    print("Pensando...")
    resposta = perguntar(texto)
    print(f"  Luiza: {resposta}")

    print("Falando...")
    falar(resposta)
