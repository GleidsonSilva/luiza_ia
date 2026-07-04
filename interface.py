import threading, random
import numpy as np
import assistente

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

MOUTH_CYCLE = ["smile", "slight", "open_small", "open_medium", "open_large",
               "open_medium", "open_small", "slight"]
STATUS_MAP = {
    "idle":          "[#888]⬤ Pronta[/]",
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

def _sobrancelhas(c, estilo):
    sy = FY - 16
    if estilo == "neutra":
        c.draw_line(FX - 24, sy, FX - 8, sy, SOBRANCELHA)
        c.draw_line(FX + 8, sy, FX + 24, sy, SOBRANCELHA)
    elif estilo == "franzida":
        c.draw_line(FX - 26, sy + 2, FX - 10, sy - 2, SOBRANCELHA)
        c.draw_line(FX + 10, sy - 2, FX + 26, sy + 2, SOBRANCELHA)
    elif estilo == "curiosa":
        c.draw_line(FX - 24, sy - 2, FX - 8, sy + 2, SOBRANCELHA)
        c.draw_line(FX + 8, sy + 2, FX + 24, sy - 2, SOBRANCELHA)

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

def desenhar_rosto(c, expressao, frame_boca):
    c.clear()
    _cabelo_atras(c)
    _filled_oval(c, FX, FY, 36, 25, PELE)
    _franja(c)
    _lacinho(c)

    if expressao == "idle":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y, "normal")
        _olho(c, EYE_R, EYE_Y, "normal")
        _blush(c)
        _boca(c, FX, MOUTH_Y, "smile")
    elif expressao == "ouvindo":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y, "normal")
        _olho(c, EYE_R, EYE_Y, "normal")
        _blush(c)
        _boca(c, FX, MOUTH_Y, "slight")
    elif expressao == "transcrevendo":
        _sobrancelhas(c, "franzida")
        fdx, fdy = random.choice([(-1,-2), (1,-2), (-1,2), (1,2)])
        _olho(c, EYE_L, EYE_Y, "focused", fdx, fdy)
        _olho(c, EYE_R, EYE_Y, "focused", fdx, fdy)
        _blush(c)
        _boca(c, FX, MOUTH_Y, "slight")
    elif expressao == "pensando":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y - 3, "up")
        _olho(c, EYE_R, EYE_Y - 3, "up")
        _boca(c, FX, MOUTH_Y, "surprised")
    elif expressao == "falando":
        _sobrancelhas(c, "neutra")
        _olho(c, EYE_L, EYE_Y, "happy")
        _olho(c, EYE_R, EYE_Y, "happy")
        _blush(c)
        _boca(c, FX, MOUTH_Y, MOUTH_CYCLE[frame_boca])


class LuizaApp(App[None]):
    CSS = """
    Screen {
        background: #000000;
    }

    #face-wrapper {
        height: 38;
        align: center middle;
    }

    Canvas {
        background: #000000;
        width: 120;
        height: 38;
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
        color: #888;
        text-style: bold;
        margin: 0 4;
        margin-top: 1;
    }

    .label-luiza {
        color: #D4726A;
    }

    .bubble {
        background: #FFFFFF;
        border: solid #CCC;
        padding: 0 2;
        height: 3;
        margin: 0 4;
    }

    .bubble-luiza {
        border: solid #D4726A;
    }

    #hint {
        height: 1;
        color: #AAA;
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("space", "toggle_recording", "Falar"),
        ("escape", "quit", "Sair"),
    ]

    expressao = reactive("idle")
    _frame_boca = 0

    def __init__(self):
        super().__init__()
        self.gravando = False
        self.recording_idx = 0

    def compose(self) -> ComposeResult:
        with Container(id="face-wrapper"):
            yield Canvas(W, H, canvas_color=FUNDO, id="rosto")
        yield Static(STATUS_MAP["idle"], id="status")
        yield Static("Você:", classes="label")
        yield Static("", id="stt-text", classes="bubble")
        yield Static("Luiza:", classes="label label-luiza")
        yield Static("", id="tts-text", classes="bubble bubble-luiza")
        yield Static("[Espaço] Falar  [Esc] Sair", id="hint")

    def on_mount(self):
        self._boca_timer = self.set_interval(1 / 8, self._avancar_boca,
                                             pause=True)
        self._atualizar_rosto()

    def _avancar_boca(self):
        self._frame_boca = (self._frame_boca + 1) % len(MOUTH_CYCLE)
        if self.expressao == "falando":
            self._atualizar_rosto()

    def _atualizar_rosto(self):
        canvas = self.query_one("#rosto", Canvas)
        with canvas.batch_refresh():
            desenhar_rosto(canvas, self.expressao, self._frame_boca)

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

    def action_toggle_recording(self):
        if self.gravando:
            self._parar_gravacao()
        else:
            self._iniciar_gravacao()

    def _iniciar_gravacao(self):
        self.gravando = True
        self.recording_idx = len(assistente.buf)
        self._mudar_expressao("ouvindo")
        self.query_one("#stt-text", Static).update("")
        self.query_one("#tts-text", Static).update("")

    def _parar_gravacao(self):
        self.gravando = False
        self._mudar_expressao("transcrevendo")
        audio = np.concatenate(assistente.buf[self.recording_idx:])
        threading.Thread(target=self._processar, args=(audio,),
                         daemon=True).start()

    def _processar(self, audio):
        texto = assistente.transcrever(audio)
        if not texto:
            self.call_from_thread(self._voltar_idle)
            return

        self.call_from_thread(self._atualizar_stt, texto)
        self.call_from_thread(self._mudar_expressao, "pensando")

        resposta = assistente.perguntar(texto)

        self.call_from_thread(self._mudar_expressao, "falando")
        self.call_from_thread(self._atualizar_tts, resposta)

        assistente.falar(resposta)

        self.call_from_thread(self._voltar_idle)

    def _atualizar_stt(self, texto):
        self.query_one("#stt-text", Static).update(texto)

    def _atualizar_tts(self, texto):
        self.query_one("#tts-text", Static).update(texto)

    def _voltar_idle(self):
        self._mudar_expressao("idle")
