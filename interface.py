import threading, random, time
import numpy as np
import assistente
import debug

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Container, Horizontal
from textual_canvas import Canvas
from textual.color import Color

W, H = 120, 72
FX, FY = 60, 43

PELE = Color(255, 200, 160)
OLHO = Color(70, 148, 80)
SOBRANCELHA = Color(94, 60, 45)
BRANCO = Color(255, 255, 255)
BLUSH = Color(255, 130, 160)
BOCA_COR = Color(235, 70, 75)
CABELO = Color(255, 215, 80)
C_LACO = Color(220, 40, 40)
C_LACO_ESCURO = Color(160, 20, 20)
FUNDO = Color(0, 0, 0)
WALL = Color(245, 225, 200)
WINDOW_GLASS = Color(170, 210, 240)
MOON = Color(255, 250, 220)
STAR = Color(255, 255, 200)
FRAME = Color(160, 100, 70)
RUG = Color(210, 160, 180)
DRESS = Color(180, 180, 240)

MOUTH_CYCLE = ["smile", "slight", "open_small", "open_medium", "open_large",
               "open_medium", "open_small", "slight"]
STATUS_MAP = {
    "idle":          "[#555]⬤ Pronta[/]",
    "ouvindo":       "[#4CAF50]⬤ Ouvindo...[/]",
    "transcrevendo": "[#FFC107]⬤ Entendendo...[/]",
    "pensando":      "[#2196F3]⬤ Pensando...[/]",
    "falando":       "[#9C27B0]⬤ Falando...[/]",
}

def _filled_oval(c, cx, cy, rx, ry, color):
    for dy in range(-ry, ry + 1):
        y = cy + dy
        if y < 0 or y >= H:
            continue
        for dx in range(-rx, rx + 1):
            x = cx + dx
            if x < 0 or x >= W:
                continue
            if (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1:
                c.set_pixel(x, y, color)

def _filled_circle(c, cx, cy, r, color):
    for dy in range(-r, r + 1):
        y = cy + dy
        if y < 0 or y >= H:
            continue
        for dx in range(-r, r + 1):
            x = cx + dx
            if x < 0 or x >= W:
                continue
            if dx * dx + dy * dy <= r * r:
                c.set_pixel(x, y, color)

def _olho(c, cx, cy, estilo, fdx=0, fdy=0):
    _filled_circle(c, cx, cy, 6, BRANCO)
    c.draw_circle(cx, cy, 6, SOBRANCELHA)
    if estilo == "normal":
        _filled_circle(c, cx, cy, 3, OLHO)
        c.set_pixel(cx + 1, cy - 1, BRANCO)
    elif estilo == "happy":
        _filled_circle(c, cx, cy, 3, OLHO)
        c.set_pixel(cx + 1, cy - 1, BRANCO)
        c.draw_line(cx - 5, cy + 5, cx + 5, cy + 5, SOBRANCELHA)
    elif estilo == "focused":
        _filled_circle(c, cx + fdx, cy + fdy, 2, OLHO)
    elif estilo == "up":
        _filled_circle(c, cx - 1, cy - 2, 2, OLHO)
    elif estilo == "olhando":
        _filled_circle(c, cx + fdx, cy + fdy, 3, OLHO)
        c.set_pixel(cx + 1 + fdx, cy - 1 + fdy, BRANCO)

def _boca_curva(c, cx, cy, abertura, largura):
    for dx in range(-largura, largura + 1):
        dy = int((dx * dx) / (largura * largura / abertura) - abertura / 2)
        c.set_pixel(cx + dx, cy - dy, BOCA_COR)
        c.set_pixel(cx + dx, cy - dy + 1, BOCA_COR)

def _boca_retang(c, cx, cy, w, h):
    for dy in range(h):
        for dx in range(-w, w + 1):
            c.set_pixel(cx + dx, cy + dy, BOCA_COR)

def _boca(c, cx, cy, forma):
    if forma == "smile":
        _boca_curva(c, cx, cy, 6, 12)
    elif forma == "slight":
        _boca_curva(c, cx, cy, 4, 9)
    elif forma == "open_small":
        _boca_retang(c, cx, cy, 3, 4)
    elif forma == "open_medium":
        _boca_retang(c, cx, cy, 5, 6)
    elif forma == "open_large":
        _boca_retang(c, cx, cy, 6, 6)
    elif forma == "thinking":
        for dx in range(-6, 7):
            c.set_pixel(cx + dx, cy, BOCA_COR)
            c.set_pixel(cx + dx, cy + 1, BOCA_COR)
        c.set_pixel(cx - 6, cy + 1, BOCA_COR)
        c.set_pixel(cx + 6, cy + 1, BOCA_COR)
        c.set_pixel(cx - 6, cy + 2, BOCA_COR)
        c.set_pixel(cx + 6, cy + 2, BOCA_COR)
    elif forma == "surprised":
        _filled_circle(c, cx, cy, 5, BOCA_COR)
    elif forma == "open_wide":
        _boca_retang(c, cx, cy, 7, 7)

def _blush(c):
    _filled_circle(c, FX - 22, FY + 2, 5, BLUSH)
    _filled_circle(c, FX + 22, FY + 2, 5, BLUSH)

def _palpebra(c, cx, cy):
    for ox in range(-6, 7):
        oy = (ox * ox) // 7
        c.set_pixel(cx + ox, cy - 7 + oy, SOBRANCELHA)

def _olho_fechado(c, cx, cy):
    _filled_circle(c, cx, cy, 7, PELE)
    c.draw_line(cx - 7, cy, cx + 7, cy, SOBRANCELHA)

def _sobrancelhas(c, estilo):
    sy = FY - 16
    if estilo == "neutra":
        for ox in range(-7, 8):
            oy = (ox * ox) // 12
            c.set_pixel(FX - 16 + ox, sy - 4 + oy, SOBRANCELHA)
            c.set_pixel(FX + 16 + ox, sy - 4 + oy, SOBRANCELHA)
    elif estilo == "franzida":
        c.draw_line(FX - 26, sy + 2, FX - 10, sy - 2, SOBRANCELHA)
        c.draw_line(FX + 10, sy - 2, FX + 26, sy + 2, SOBRANCELHA)
    elif estilo == "curiosa":
        c.draw_line(FX - 24, sy - 2, FX - 8, sy + 2, SOBRANCELHA)
        c.draw_line(FX + 8, sy + 2, FX + 24, sy - 2, SOBRANCELHA)

def _tapete(c):
    _filled_oval(c, FX, 67, 30, 4, RUG)

def _janela(c):
    for y in range(3, 21):
        for x in range(103, 120):
            c.set_pixel(x, y, WINDOW_GLASS)
    c.draw_line(103, 3, 119, 3, FRAME)
    c.draw_line(103, 3, 103, 20, FRAME)
    c.draw_line(119, 3, 119, 20, FRAME)
    c.draw_line(103, 20, 119, 20, FRAME)
    c.draw_line(111, 3, 111, 20, FRAME)
    c.draw_line(103, 11, 119, 11, FRAME)

def _lua_estrelas(c):
    _filled_circle(c, 16, 10, 6, MOON)
    _filled_circle(c, 13, 8, 5, WALL)
    for sx, sy in [(30,5), (25,18), (35,12), (8,22), (20,25)]:
        c.set_pixel(sx, sy, STAR)
        c.set_pixel(sx + 1, sy + 1, STAR)
        c.set_pixel(sx - 1, sy + 1, STAR)
        c.set_pixel(sx, sy + 2, STAR)
        c.set_pixel(sx + 1, sy - 1, STAR)

 
def _corpo(c):
    for x in range(56, 64):
        for y in range(57, 59):
            c.set_pixel(x, y, PELE)
    for y in range(59, H):
        for x in range(FX - 12, FX + 13):
            c.set_pixel(x, y, DRESS)

def _cabelo_atras(c):
    _filled_oval(c, FX, FY - 18, 50, 19, CABELO)
    _filled_oval(c, FX - 40, FY - 10, 14, 20, CABELO)
    _filled_oval(c, FX + 40, FY - 10, 14, 20, CABELO)

def _franja(c):
    for x in range(33, 90, 6):
        _filled_oval(c, x, FY - 22, 4, 5, CABELO)

def _lacinho(c):
    _filled_oval(c, FX - 12, FY - 32, 10, 5, C_LACO)
    _filled_oval(c, FX + 12, FY - 32, 10, 5, C_LACO)
    _filled_circle(c, FX, FY - 32, 3, C_LACO_ESCURO)

EYE_L = FX - 16
EYE_R = FX + 16
EYE_Y = FY - 10
MOUTH_Y = FY + 12

def desenhar_rosto(c, expressao, frame_boca, piscando=False, olho_dir=(0,0)):
    c.clear()
    _tapete(c)
    _janela(c)
    _lua_estrelas(c)
    _corpo(c)
    _cabelo_atras(c)
    _filled_oval(c, FX, FY, 36, 28, PELE)
    _franja(c)
    _lacinho(c)

    if expressao == "idle":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y, "olhando", *olho_dir)
        _olho(c, EYE_R, EYE_Y, "olhando", *olho_dir)
        _palpebra(c, EYE_L, EYE_Y)
        _palpebra(c, EYE_R, EYE_Y)
        _blush(c)
        _boca(c, FX, MOUTH_Y, "smile")
    elif expressao == "ouvindo":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y, "normal")
        _olho(c, EYE_R, EYE_Y, "normal")
        _palpebra(c, EYE_L, EYE_Y)
        _palpebra(c, EYE_R, EYE_Y)
        _blush(c)
        _boca(c, FX, MOUTH_Y, "slight")
    elif expressao == "transcrevendo":
        _sobrancelhas(c, "franzida")
        fdx, fdy = random.choice([(-1,-2), (1,-2), (-1,2), (1,2)])
        _olho(c, EYE_L, EYE_Y, "focused", fdx, fdy)
        _olho(c, EYE_R, EYE_Y, "focused", fdx, fdy)
        _palpebra(c, EYE_L, EYE_Y)
        _palpebra(c, EYE_R, EYE_Y)
        _blush(c)
        _boca(c, FX, MOUTH_Y, "slight")
    elif expressao == "pensando":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y - 3, "up")
        _olho(c, EYE_R, EYE_Y - 3, "up")
        _palpebra(c, EYE_L, EYE_Y - 3)
        _palpebra(c, EYE_R, EYE_Y - 3)
        _boca(c, FX, MOUTH_Y, "surprised")
    elif expressao == "falando":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y, "happy")
        _olho(c, EYE_R, EYE_Y, "happy")
        _palpebra(c, EYE_L, EYE_Y)
        _palpebra(c, EYE_R, EYE_Y)
        _blush(c)
        _boca(c, FX, MOUTH_Y, MOUTH_CYCLE[frame_boca])

    if piscando:
        _olho_fechado(c, EYE_L, EYE_Y - 1)
        _olho_fechado(c, EYE_R, EYE_Y - 1)


class LuizaApp(App[None]):
    CSS = """
    Screen {
        background: #f5e1c8;
        padding: 0;
    }

    #face-wrapper {
        height: 38;
        align: center middle;
        background: #f5e1c8;
        padding: 0;
    }

    Canvas {
        background: #f5e1c8;
        width: 120;
        height: 38;
        padding: 0;
    }

    #status {
        height: 1;
        content-align: center middle;
        margin-bottom: 1;
    }

    .bubble-row {
        height: 3;
        margin: 0 4;
    }

    .label {
        height: 1;
        color: #555;
        text-style: bold;
        margin: 0 4;
        margin-top: 1;
    }

    .label-luiza {
        color: #D4726A;
    }

    .bubble {
        background: #fcf0e0;
        border: solid #d0c0a8;
        padding: 0 2;
        height: 3;
        margin: 0 4;
    }

    #stt-text, #tts-text {
        background: #fcf0e0 !important;
    }

    .bubble-luiza {
        border: solid #D4726A;
    }

    #hint {
        height: 1;
        color: #777;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("space", "toggle_recording", "Falar"),
        ("enter", "toggle_recording"),
        ("escape", "quit", "Sair"),
    ]

    _ultimo_toggle: float = 0.0

    def action_quit(self):
        assistente.parar_stream()
        self.exit()

    def action_toggle_recording(self):
        agora = time.monotonic()
        if self.processando:
            return
        if not self.gravando:
            if agora - self._ultimo_toggle < 0.3:
                return
            self._ultimo_toggle = agora
            self._inicio_gravacao = agora
            self._iniciar_gravacao()
        elif agora - self._inicio_gravacao >= 2.0:
            self._parar_gravacao()

    expressao = reactive("idle")
    _frame_boca = 0
    _piscando = False

    def __init__(self):
        super().__init__()
        self.gravando = False
        self.processando = False
        self.recording_idx = 0
        self._olho_dir = (0, 0)
        self._silencio_count = 0
        self._inicio_gravacao = 0.0

    def compose(self) -> ComposeResult:
        with Container(id="face-wrapper"):
            yield Canvas(W, H, canvas_color=WALL, id="rosto")
        yield Static(STATUS_MAP["idle"], id="status")
        yield Static("Você:", classes="label")
        yield Static(" ", id="stt-text", classes="bubble")
        yield Static("Luiza:", classes="label label-luiza")
        yield Static(" ", id="tts-text", classes="bubble bubble-luiza")
        yield Static("\\[Espaço/Enter] Falar  \\[Esc] Sair", id="hint")

    def on_mount(self):
        assistente.iniciar_stream()
        self._boca_timer = self.set_interval(1 / 8, self._avancar_boca,
                                             pause=True)
        self._silencio_timer = self.set_interval(0.5, self._verificar_silencio,
                                                  pause=True)
        self._agendar_piscar()
        self._agendar_olhar()
        self._atualizar_rosto()

    def _agendar_piscar(self):
        self.set_timer(random.uniform(8, 15), self._piscar)

    def _piscar(self):
        self._piscando = True
        self._atualizar_rosto()
        self.set_timer(0.2, self._abrir_olhos)

    def _abrir_olhos(self):
        self._piscando = False
        self._atualizar_rosto()

    DIRECOES = [(0,0), (-2,0), (2,0), (0,-2), (0,1),
                (-2,-1), (2,-1), (-2,1), (2,1)]

    def _agendar_olhar(self):
        self.set_timer(random.uniform(3, 9), self._olhar_aleatorio)

    def _olhar_aleatorio(self):
        if self.expressao == "idle":
            self._olho_dir = random.choice(self.DIRECOES)
            self._atualizar_rosto()
        self._agendar_olhar()

    def _avancar_boca(self):
        self._frame_boca = (self._frame_boca + 1) % len(MOUTH_CYCLE)
        if self.expressao == "falando":
            self._atualizar_rosto()

    def _atualizar_rosto(self):
        canvas = self.query_one("#rosto", Canvas)
        with canvas.batch_refresh():
            desenhar_rosto(canvas, self.expressao, self._frame_boca,
                           self._piscando, self._olho_dir)

    def _mudar_expressao(self, expressao):
        self.expressao = expressao
        self._frame_boca = 0
        self._atualizar_rosto()
        status = self.query_one("#status", Static)
        status.update(STATUS_MAP.get(expressao, ""))
        if expressao == "falando":
            self._boca_timer.resume()
        else:
            self._boca_timer.pause()

    def _iniciar_gravacao(self):
        self.gravando = True
        with assistente.buf_lock:
            self.recording_idx = len(assistente.buf)
        self._silencio_count = 0
        self._silencio_timer.resume()
        self._mudar_expressao("ouvindo")
        self.query_one("#stt-text", Static).update("")
        self.query_one("#tts-text", Static).update("")

    def _parar_gravacao(self):
        self._silencio_timer.pause()
        self.gravando = False
        self.processando = True
        self._mudar_expressao("transcrevendo")
        with assistente.buf_lock:
            audio = np.concatenate(assistente.buf[self.recording_idx:])
            del assistente.buf[:self.recording_idx]
            self.recording_idx = 0
        threading.Thread(target=self._processar, args=(audio,),
                         daemon=True).start()

    def _verificar_silencio(self):
        if not self.gravando:
            return
        with assistente.buf_lock:
            recent = assistente.buf[self.recording_idx:]
        if not recent:
            return
        chunks = []
        n = 0
        for chunk in reversed(recent):
            chunks.append(chunk)
            n += len(chunk)
            if n >= 16000:
                break
        if n < 4000:
            return
        audio = np.concatenate(list(reversed(chunks)))
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.008:
            self._silencio_count += 1
            if self._silencio_count >= 10:
                debug.log(f"silence detected after {self._silencio_count} checks, rms={rms:.5f}")
                self._parar_gravacao()
        else:
            self._silencio_count = 0

    def _processar(self, audio):
        try:
            debug.log(f"_processar: audio shape={audio.shape}")
            texto = assistente.transcrever(audio)
            debug.log(f"_processar: texto='{texto}'")
            if not texto:
                self.call_from_thread(self._voltar_idle)
                return

            self.call_from_thread(self._atualizar_stt, texto)
            self.call_from_thread(self._mudar_expressao, "pensando")

            debug.log("chamando perguntar...")
            resposta = assistente.perguntar(texto)
            debug.log(f"resposta='{resposta}'")

            self.call_from_thread(self._mudar_expressao, "falando")
            self.call_from_thread(self._atualizar_tts, resposta)

            assistente.falar(resposta)
        except Exception as e:
            import traceback
            import io
            tb = io.StringIO()
            traceback.print_exc(file=tb)
            debug.log(f"ERRO: {e}\n{tb.getvalue()}")
            self.call_from_thread(self._atualizar_tts, f"Erro: {e}")
        finally:
            debug.log("_processar finalizando")
            self.call_from_thread(self._voltar_idle)
            self.processando = False

    def _atualizar_stt(self, texto):
        self.query_one("#stt-text", Static).update(texto)

    def _atualizar_tts(self, texto):
        self.query_one("#tts-text", Static).update(texto)

    def _voltar_idle(self):
        self._mudar_expressao("idle")
