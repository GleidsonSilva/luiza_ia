import math, threading, random, time, re
import numpy as np
import sounddevice as sd
import assistente
import memoria
import debug
import reconhecimento_voz

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Container, Horizontal
from textual_canvas import Canvas
from textual.color import Color


class VADBuffer:
    def __init__(self, max_chunks=32):
        from collections import deque
        self._buf = deque(maxlen=max_chunks)
        self._sum_sq = 0.0
        self._count = 0

    def update(self, chunk):
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        self._sum_sq += float(np.dot(arr, arr))
        self._count += int(arr.size)
        if len(self._buf) == self._buf.maxlen:
            old = self._buf[0]
            self._sum_sq -= float(np.dot(old, old))
            self._count -= int(old.size)
        self._buf.append(arr)

    @property
    def count(self):
        return self._count

    @property
    def rms(self):
        if self._count == 0:
            return 0.0
        return float(np.sqrt(self._sum_sq / self._count))

    def clear(self):
        self._buf.clear()
        self._sum_sq = 0.0
        self._count = 0


W, H = 120, 72
FX, FY = 60, 43

PELE = Color(255, 200, 160)
NARIZ = Color(235, 180, 145)
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

MOUTH_CYCLE = ["smile", "slight", "slight", "open_small", "open_small",
               "open_medium", "open_large", "open_medium", "open_small",
               "open_small", "slight", "slight"]
STATUS_MAP = {
    "idle":          "[#555]😊 Pronta[/]",
    "ouvindo":       "[#4CAF50]🎤 Ouvindo...[/]",
    "transcrevendo": "[#FFC107]🤔 Entendendo...[/]",
    "pensando":      "[#2196F3]💭 Pensando...[/]",
    "falando":       "[#9C27B0]🗣️ Falando...[/]",
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

def _olho(c, cx, cy, estilo, fdx=0, fdy=0, rms=0.0, t=0.0):
    _filled_circle(c, cx, cy, 6, BRANCO)
    c.draw_circle(cx, cy, 6, SOBRANCELHA)
    if estilo == "normal":
        pr = 3 + int(min(1.5, rms * 3))
        _filled_circle(c, cx + fdx, cy + fdy, pr, OLHO)
        c.set_pixel(cx + 1 + fdx, cy - 1 + fdy, BRANCO)
        c.set_pixel(cx + 3 + fdx, cy - 2 + fdy, BRANCO)
        c.set_pixel(cx - 1 + fdx, cy + 1 + fdy, Color(120, 180, 130))
    elif estilo == "happy":
        pr = 3 + int(min(1.5, rms * 3))
        _filled_circle(c, cx + fdx, cy + fdy, pr, OLHO)
        c.set_pixel(cx + 1 + fdx, cy - 1 + fdy, BRANCO)
        c.set_pixel(cx + 3 + fdx, cy - 2 + fdy, BRANCO)
        c.set_pixel(cx - 1 + fdx, cy + 1 + fdy, Color(120, 180, 130))
        if rms > 0.02:
            c.set_pixel(cx + fdx, cy - 2 + fdy, Color(120, 180, 130))
    elif estilo == "happy_talk":
        pr = 4 + int(min(1.0, rms * 2))
        _filled_circle(c, cx + fdx, cy + fdy, pr, OLHO)
        c.set_pixel(cx + 1 + fdx, cy - 1 + fdy, BRANCO)
        c.set_pixel(cx + 3 + fdx, cy - 3 + fdy, BRANCO)
        c.set_pixel(cx - 2 + fdx, cy - 2 + fdy, BRANCO)
        c.set_pixel(cx - 1 + fdx, cy + 1 + fdy, Color(120, 180, 130))
        if rms > 0.02:
            c.set_pixel(cx + fdx, cy - 3 + fdy, Color(120, 180, 130))
            c.set_pixel(cx + 2 + fdx, cy + fdy, Color(120, 180, 130))
            c.set_pixel(cx - 2 + fdx, cy + fdy, Color(120, 180, 130))
    elif estilo == "focused":
        _filled_circle(c, cx + fdx, cy + fdy, 2, OLHO)
        c.set_pixel(cx + 1 + fdx, cy - 1 + fdy, BRANCO)
    elif estilo == "up":
        _filled_circle(c, cx - 1, cy - 2, 2, OLHO)
        c.set_pixel(cx, cy - 3, BRANCO)
    elif estilo == "olhando":
        _filled_circle(c, cx + fdx, cy + fdy, 3, OLHO)
        c.set_pixel(cx + 1 + fdx, cy - 1 + fdy, BRANCO)
        c.set_pixel(cx + 3 + fdx, cy - 2 + fdy, BRANCO)
        c.set_pixel(cx + 2 + fdx, cy - 2 + fdy, SOBRANCELHA)
        c.set_pixel(cx - 1 + fdx, cy - 2 + fdy, SOBRANCELHA)
    CILIO = Color(70, 45, 30)
    for ox in range(-4, 6, 2):
        dy = int(math.sqrt(max(0, 36 - ox * ox)))
        c.set_pixel(cx + ox, cy - dy - 1, CILIO)
        c.set_pixel(cx + ox, cy - dy - 2, CILIO)

def _boca_curva(c, cx, cy, abertura, largura, assimetrico=0):
    dy_max = int(abertura / 2)
    for dx in range(-largura, largura + 1):
        dy = int((dx * dx) / (largura * largura / abertura) - abertura / 2)
        asy = int(assimetrico * dx / largura) if assimetrico else 0
        c.set_pixel(cx + dx, cy - dy - 1 + asy, Color(
            max(0, BOCA_COR.r - 30),
            max(0, BOCA_COR.g - 30),
            max(0, BOCA_COR.b - 30)
        ))
        c.set_pixel(cx + dx, cy - dy + asy, BOCA_COR)
        c.set_pixel(cx + dx, cy - dy + 1 + asy, BOCA_COR)
        if dx > -largura:
            pd = int(((dx-1)*(dx-1)) / (largura*largura/abertura) - abertura/2)
            pa = int(assimetrico * (dx-1) / largura) if assimetrico else 0
            prev_top = cy - pd + pa
            curr_top = cy - dy + asy
            if curr_top > prev_top:
                c.set_pixel(cx + dx, prev_top, BOCA_COR)
            if curr_top < prev_top:
                c.set_pixel(cx + dx - 1, curr_top, BOCA_COR)
        if abs(dx) <= largura // 3:
            hl = Color(
                min(255, BOCA_COR.r + 40),
                min(255, BOCA_COR.g + 40),
                min(255, BOCA_COR.b + 40)
            )
            c.set_pixel(cx + dx, cy - dy + 2 + asy, hl)

    flick_y = cy - dy_max - 2
    c.set_pixel(cx - largura - 1, flick_y, BOCA_COR)
    c.set_pixel(cx - largura - 2, flick_y + 1, BOCA_COR)
    c.set_pixel(cx + largura + 1, flick_y, BOCA_COR)
    c.set_pixel(cx + largura + 2, flick_y + 1, BOCA_COR)

def _boca_retang(c, cx, cy, w, h, inner=False, curve=True):
    shadow = Color(
        max(0, BOCA_COR.r - 40),
        max(0, BOCA_COR.g - 40),
        max(0, BOCA_COR.b - 40)
    )
    mid = h // 2
    for dy in range(h):
        cap = int(w - abs(dy - mid))
        actual_w = max(1, cap)
        actual_w = min(actual_w, w)
        for dx in range(-actual_w, actual_w + 1):
            c.set_pixel(cx + dx, cy + dy, BOCA_COR)
            if dy == h - 1 and dx % 2 == 0 and curve:
                c.set_pixel(cx + dx, cy + dy + 1, shadow)
    if inner and h > 2:
        teeth = Color(255, 245, 230)
        for dx in range(-w + 1, w):
            c.set_pixel(cx + dx, cy + 1, teeth)

def _boca(c, cx, cy, forma):
    if forma == "smile":
        _boca_curva(c, cx, cy, 6, 13)
    elif forma == "smile_wide":
        _boca_curva(c, cx, cy, 7, 15, assimetrico=1)
    elif forma == "slight":
        _boca_curva(c, cx, cy, 4, 10)
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

def _boca_altura(forma):
    if forma == "smile": return 6
    if forma == "slight": return 4
    if forma == "open_small": return 4
    if forma == "open_medium": return 6
    if forma == "open_large": return 6
    if forma == "thinking": return 2
    if forma == "surprised": return 5
    if forma == "open_wide": return 7
    return 3

def _boca_dinamica(c, cx, cy, base_forma, rms, frame):
    base = _boca_altura(base_forma)
    h = max(2, base + int(rms * 14))
    h = min(7, h)
    w = max(3, min(8, int(h * 1.2 + rms * 8)))
    _boca_retang(c, cx, cy, w, h, inner=rms > 0.04)
    if h >= 4:
        mid = h // 2
        for sign in (-1, 1):
            c.set_pixel(cx + sign * (w + 1), cy + mid, BOCA_COR)
            c.set_pixel(cx + sign * (w + 1), cy + mid - 1, BOCA_COR)

def _blush(c):
    _filled_circle(c, FX - 22, FY + 2, 3, BLUSH)
    _filled_circle(c, FX + 22, FY + 2, 3, BLUSH)

def _nariz(c):
    c.set_pixel(FX - 1, FY + 2, NARIZ)
    c.set_pixel(FX, FY + 2, NARIZ)

def _palpebra(c, cx, cy, soft=False, squint=0):
    cor = Color(210, 165, 130) if soft else SOBRANCELHA
    lash = Color(
        max(0, cor.r - 40),
        max(0, cor.g - 40),
        max(0, cor.b - 40)
    )
    for ox in range(-7, 8):
        oy = (ox * ox) // 7
        base_y = cy - 4 + squint + oy
        c.set_pixel(cx + ox, base_y, cor)
        if not soft and ox % 2 == 0:
            c.set_pixel(cx + ox, base_y + 1, Color(
                max(0, cor.r - 20),
                max(0, cor.g - 20),
                max(0, cor.b - 20)
            ))
        c.set_pixel(cx + ox, base_y - 1, lash)

def _olho_fechado(c, cx, cy):
    _filled_circle(c, cx, cy, 7, PELE)
    for x in range(cx - 7, cx + 8):
        oy = ((x - cx) ** 2) // 7
        c.set_pixel(x, cy - 7 + oy, SOBRANCELHA)

def _sobrancelha_traca(c, x1, x2, y1, y2, espessura=1):
    length = max(1, abs(x2 - x1))
    step = 1 if x2 >= x1 else -1
    for x in range(x1, x2 + step, step):
        t = (x - x1) / length
        y = int(y1 + t * (y2 - y1))
        for dy in range(-espessura, espessura):
            c.set_pixel(x, y + dy, SOBRANCELHA)

def _sobrancelhas(c, estado, t=0.0, rms=0.0):
    sy = FY - 16
    if estado == "neutra":
        oy_neutra = int(0.6 * math.sin(t * 0.7))
        for ox in range(-7, 8):
            oy = (ox * ox) // 20
            c.set_pixel(FX - 16 + ox, sy - 3 + oy + oy_neutra, SOBRANCELHA)
            c.set_pixel(FX - 16 + ox, sy - 4 + oy + oy_neutra, SOBRANCELHA)
            c.set_pixel(FX + 16 + ox, sy - 3 + oy + oy_neutra, SOBRANCELHA)
            c.set_pixel(FX + 16 + ox, sy - 4 + oy + oy_neutra, SOBRANCELHA)
    elif estado == "franzida":
        _sobrancelha_traca(c, FX - 26, FX - 10, sy - 1, sy - 3, 1)
        _sobrancelha_traca(c, FX + 10, FX + 26, sy - 3, sy - 1, 1)
    elif estado == "curiosa":
        _sobrancelha_traca(c, FX - 24, FX - 8, sy - 4, sy - 2, 1)
        _sobrancelha_traca(c, FX + 8, FX + 24, sy - 2, sy - 4, 1)
    elif estado == "triste":
        _sobrancelha_traca(c, FX - 24, FX - 10, sy, sy + 2, 1)
        _sobrancelha_traca(c, FX + 10, FX + 24, sy + 2, sy, 1)
    elif estado == "brava":
        _sobrancelha_traca(c, FX - 24, FX - 10, sy - 4, sy, 1)
        _sobrancelha_traca(c, FX + 10, FX + 24, sy, sy - 4, 1)
    elif estado == "feliz":
        oy_anim = int(1.5 * math.sin(t * 2.5))
        for side, sign in [(FX - 16, -1), (FX + 16, 1)]:
            for x in range(side - 7, side + 8):
                base_y = (sy - 4)
                oy = ((x - side) ** 2) // 20
                c.set_pixel(x, base_y + oy + oy_anim, SOBRANCELHA)
                c.set_pixel(x, base_y + oy + 1 + oy_anim, SOBRANCELHA)
                if abs(x - side) > 4:
                    c.set_pixel(x, base_y + oy - 1 + oy_anim, Color(
                        max(0, SOBRANCELHA.r - 20),
                        max(0, SOBRANCELHA.g - 20),
                        max(0, SOBRANCELHA.b - 20)
                    ))

def _tapete(c):
    _filled_oval(c, FX, 67, 30, 4, RUG)
    for y in range(65, 70, 2):
        for x in range(FX - 28, FX + 29):
            c.set_pixel(x, y, Color(180, 130, 150))

def _janela(c, now=None):
    if now is None:
        now = time.monotonic()
    hour = time.localtime(time.time()).tm_hour
    if hour >= 6 and hour < 12:
        glass = Color(210, 225, 250)
        frame = Color(170, 110, 80)
    elif hour >= 12 and hour < 20:
        glass = WINDOW_GLASS
        frame = FRAME
    else:
        glass = Color(18, 22, 45)
        frame = Color(80, 55, 35)
    for y in range(3, 21):
        for x in range(103, 120):
            c.set_pixel(x, y, glass)
    c.draw_line(103, 3, 119, 3, frame)
    c.draw_line(103, 3, 103, 20, frame)
    c.draw_line(119, 3, 119, 20, frame)
    c.draw_line(103, 20, 119, 20, frame)
    c.draw_line(111, 3, 111, 20, frame)
    c.draw_line(103, 11, 119, 11, frame)
    if hour >= 6 and hour < 20:
        c.set_pixel(116, 6, Color(255, 240, 180))
        c.set_pixel(117, 5, Color(255, 240, 180))
        c.set_pixel(115, 5, Color(255, 240, 180))
    else:
        stars = [(106, 5), (113, 8), (107, 14), (115, 16), (109, 19), (118, 7)]
        phase = now * 3
        for idx, (sx, sy) in enumerate(stars):
            blink = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(phase + idx * 1.3))
            col = Color(
                int(STAR.r * blink),
                int(STAR.g * blink),
                int(STAR.b * blink)
            )
            if blink > 0.7:
                c.set_pixel(sx + 1, sy, col)
                c.set_pixel(sx - 1, sy, col)
            c.set_pixel(sx, sy, col)

def _lua_estrelas(c, now=0.0):
    _filled_circle(c, 16, 10, 6, MOON)
    _filled_circle(c, 13, 8, 5, WALL)
    _filled_circle(c, 14, 9, 2, Color(235, 235, 210))
    _filled_circle(c, 18, 12, 1, Color(235, 235, 210))
    for idx, (sx, sy) in enumerate([(30,5), (25,18), (35,12), (8,22), (20,25)]):
        phase = now * 3 + idx * 1.3
        blink = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(phase))
        star_color = Color(
            int(STAR.r * blink),
            int(STAR.g * blink),
            int(STAR.b * blink)
        )
        c.set_pixel(sx, sy, star_color)
        c.set_pixel(sx + 1, sy + 1, star_color)
        c.set_pixel(sx - 1, sy + 1, star_color)
        c.set_pixel(sx, sy + 2, star_color)
        c.set_pixel(sx + 1, sy - 1, star_color)

def _pensamento(c, bubbles):
    for bx, by, br, age in bubbles:
        progress = age / 16.0
        b = int(255 - progress * 180)
        b = max(60, b)
        alpha = int(255 - progress * 200)
        alpha = max(55, alpha)
        cor = Color(b, b, b, alpha)
        _filled_circle(c, int(bx), int(by), int(br + 0.5), cor)

def _barras_audio(c, rms=0.0, now=0.0):
    if rms <= 0:
        return
    bars_per_side = 4
    bar_w = 2
    gap = 1
    base_h = 3
    y = 55
    frame = int(now * 18)
    t = (frame // 10) * 1.2
    group_w = bars_per_side * bar_w + (bars_per_side - 1) * gap
    for side, sign in [("L", -1), ("R", 1)]:
        cx = FX + sign * 29
        start_x = cx - group_w // 2
        for i in range(bars_per_side):
            phase = t + i * 1.2 + (0.5 if side == "R" else 0)
            wave = math.sin(phase) * 0.5 + 0.5
            audio_influence = min(1.0, rms * 40.0)
            h = int(base_h + wave * 2 + audio_influence * 4)
            h = max(1, min(12, h))
            x = start_x + i * (bar_w + gap)
            for dy in range(-h, h + 1):
                for dx in range(bar_w):
                    c.set_pixel(x + dx, y + dy, STAR)

  
def _corpo(c):
    for x in range(56, 64):
        for y in range(57, 59):
            c.set_pixel(x, y, PELE)
    shadow = Color(
        max(0, DRESS.r - 25),
        max(0, DRESS.g - 25),
        max(0, DRESS.b - 25)
    )
    for y in range(59, H):
        for x in range(FX - 12, FX + 13):
            c.set_pixel(x, y, DRESS)
    for y in range(62, H, 3):
        for x in range(FX - 12, FX + 13):
            c.set_pixel(x, y, shadow)

def _cabelo_atras(c):
    _filled_oval(c, FX, FY - 18, 50, 19, CABELO)
    _filled_oval(c, FX - 40, FY - 10, 14, 20, CABELO)
    _filled_oval(c, FX + 40, FY - 10, 14, 20, CABELO)
    _filled_oval(c, FX - 32, FY + 8, 9, 18, CABELO)
    _filled_oval(c, FX + 32, FY + 8, 9, 18, CABELO)

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

def desenhar_rosto(c, expressao, frame_boca, piscando=0, olho_dir=(0,0), audio_rms=0.0, now=None, mouth_h=None, bubbles=None, bocejando=False, bocejo_frame=0, trans_frame=-1, trans_total=0):
    if now is None:
        now = time.monotonic()
    breath_y = int(0.5 * math.sin(now * 2.1))
    eye_l = FX - 16
    eye_r = FX + 16
    eye_y = FY - 10
    mouth_y = FY + 12
    c.clear()
    _tapete(c)
    _janela(c, now=now)
    _lua_estrelas(c, now=now)
    _corpo(c)
    _cabelo_atras(c)
    _filled_oval(c, FX, FY + breath_y, 36, 28, PELE)
    _franja(c)
    _lacinho(c)

    boca_forma = "smile"
    if bocejando:
        if bocejo_frame <= 8:
            boca_forma = "open_medium"
        elif bocejo_frame <= 16:
            boca_forma = "open_large"
        elif bocejo_frame <= 24:
            boca_forma = "open_medium"
        else:
            boca_forma = "smile"

    if expressao == "idle":
        _sobrancelhas(c, "neutra", t=now)
        _olho(c, eye_l, eye_y, "olhando", *olho_dir, rms=audio_rms, t=now)
        _olho(c, eye_r, eye_y, "olhando", *olho_dir, rms=audio_rms, t=now)
        if bocejando:
            _olho_fechado(c, eye_l, eye_y)
            _olho_fechado(c, eye_r, eye_y)
        else:
            _palpebra(c, eye_l, eye_y, soft=True)
            _palpebra(c, eye_r, eye_y, soft=True)
        _blush(c)
        _nariz(c)
        _boca(c, FX, mouth_y, boca_forma)
    elif expressao == "ouvindo":
        _sobrancelhas(c, "feliz", t=now)
        _olho(c, eye_l, eye_y, "normal", rms=audio_rms, t=now)
        _olho(c, eye_r, eye_y, "normal", rms=audio_rms, t=now)
        _palpebra(c, eye_l, eye_y, soft=True)
        _palpebra(c, eye_r, eye_y, soft=True)
        _blush(c)
        _nariz(c)
        _boca(c, FX, mouth_y, "slight")
        _barras_audio(c, rms=audio_rms, now=now)
    elif expressao == "transcrevendo":
        _sobrancelhas(c, "curiosa", t=now)
        _olho(c, eye_l, eye_y - 3, "up", rms=0.0, t=now)
        _olho(c, eye_r, eye_y - 3, "up", rms=0.0, t=now)
        _palpebra(c, eye_l, eye_y - 3)
        _palpebra(c, eye_r, eye_y - 3)
        _blush(c)
        _nariz(c)
        _boca(c, FX, mouth_y, "surprised")
    elif expressao == "pensando":
        _sobrancelhas(c, "franzida", t=now)
        dirs = [(-1,-2), (1,-2), (-1,2), (1,2)]
        fdx, fdy = dirs[(int(now // 3)) % 4]
        _olho(c, eye_l, eye_y, "focused", fdx, fdy, rms=0.0, t=now)
        _olho(c, eye_r, eye_y, "focused", fdx, fdy, rms=0.0, t=now)
        _palpebra(c, eye_l, eye_y, soft=True)
        _palpebra(c, eye_r, eye_y, soft=True)
        _blush(c)
        _nariz(c)
        _boca(c, FX, mouth_y, "slight")
    elif expressao == "falando":
        _sobrancelhas(c, "feliz", t=now)
        _olho(c, eye_l, eye_y, "happy", 0, 0, rms=audio_rms, t=now)
        _olho(c, eye_r, eye_y, "happy", 0, 0, rms=audio_rms, t=now)
        _palpebra(c, eye_l, eye_y, soft=True)
        _palpebra(c, eye_r, eye_y, soft=True)
        _blush(c)
        _nariz(c)
        if mouth_h is not None:
            mh = int(mouth_h)
            w = max(3, min(8, int(mouth_h * 1.2 + audio_rms * 8)))
            _boca_retang(c, FX, mouth_y, w, mh, inner=audio_rms > 0.04)
            if mh >= 4:
                mid = mh // 2
                for sign in (-1, 1):
                    c.set_pixel(FX + sign * (w + 1), mouth_y + mid, BOCA_COR)
                    c.set_pixel(FX + sign * (w + 1), mouth_y + mid - 1, BOCA_COR)
        else:
            _boca_dinamica(c, FX, mouth_y, MOUTH_CYCLE[frame_boca], audio_rms, frame_boca)
        _barras_audio(c, rms=audio_rms, now=now)
    elif expressao == "triste":
        _sobrancelhas(c, "triste", t=now)
        _olho(c, eye_l, eye_y, "normal", fdy=1, rms=0.0, t=now)
        _olho(c, eye_r, eye_y, "normal", fdy=1, rms=0.0, t=now)
        _palpebra(c, eye_l, eye_y)
        _palpebra(c, eye_r, eye_y)
        _blush(c)
        _nariz(c)
        _boca(c, FX, mouth_y, "slight")
    elif expressao == "brava":
        _sobrancelhas(c, "brava", t=now)
        _olho(c, eye_l, eye_y, "focused", fdx=-1, fdy=-1, rms=0.0, t=now)
        _olho(c, eye_r, eye_y, "focused", fdx=1, fdy=-1, rms=0.0, t=now)
        _palpebra(c, eye_l, eye_y)
        _palpebra(c, eye_r, eye_y)
        _blush(c)
        _nariz(c)
        _boca(c, FX, mouth_y, "slight")
    elif expressao == "animada":
        _sobrancelhas(c, "animada", t=now)
        _olho(c, eye_l, eye_y, "happy", rms=max(audio_rms, 0.15), t=now)
        _olho(c, eye_r, eye_y, "happy", rms=max(audio_rms, 0.15), t=now)
        _palpebra(c, eye_l, eye_y, soft=True)
        _palpebra(c, eye_r, eye_y, soft=True)
        _blush(c)
        _nariz(c)
        _boca(c, FX, mouth_y, "smile_wide")

    if trans_frame >= 0 and trans_frame < trans_total * 0.4:
        _olho_fechado(c, eye_l, eye_y - 1)
        _olho_fechado(c, eye_r, eye_y - 1)
        for dx in range(-4, 5):
            c.set_pixel(FX + dx, mouth_y, SOBRANCELHA)
            c.set_pixel(FX + dx, mouth_y + 1, SOBRANCELHA)
        if expressao == "falando":
            _barras_audio(c, rms=audio_rms, now=now)

    if piscando >= 1:
        _olho_fechado(c, eye_l, eye_y - 1)
        _olho_fechado(c, eye_r, eye_y - 1)


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
        width: 1fr;
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

    .bubble {
        background: #fcf0e0;
        border: solid #d0c0a8;
        color: #4a3f35;
        padding: 0 2;
        height: 5;
        margin: 0;
        overflow: hidden;
        border-top: round #d0c0a8;
        border-bottom: round #d0c0a8;
        width: 70%;
        content-align: left middle;
    }

    .bubble-luiza {
        border: solid #D4726A;
        border-top: round #D4726A;
        border-bottom: round #D4726A;
    }

    .bubble-erro {
        border: solid #F44336;
        color: #B71C1C;
        background: #fff0f0;
    }

    .bubble-row {
        align: center middle;
        width: 1fr;
        height: auto;
    }

    #hint {
        height: 1;
        color: #777;
        text-align: center;
        margin-top: 1;
    }

    #ctx-usage {
        height: 1;
        text-align: right;
        margin-top: 0;
        width: 1fr;
        color: #999;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("space", "toggle_recording", "Falar"),
        ("enter", "toggle_recording"),
        ("escape", "quit", "Sair"),
    ]

    _ultimo_toggle: float = 0.0
    _wake_enabled: bool = True

    def _wake_check(self, trigger=False):
        if trigger:
            agora = time.monotonic()
            if self.processando or self.gravando:
                return
            if agora - self._ultimo_toggle < 1.0:
                return
            self.call_from_thread(self.action_toggle_recording)
        elif self._wake_enabled:
            return not self.gravando and not self.processando and self.expressao == "idle"
        return True

    def action_quit(self):
        if self._wake_enabled:
            try:
                import detector_palavra
                detector_palavra.stop()
            except Exception:
                pass
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
            self._iniciar_gravacao_com_saudacao(from_wake=True)
        elif agora - self._inicio_gravacao >= 3.5:
            self._parar_gravacao()

    expressao = reactive("idle")
    _frame_boca = 0
    _piscando = 0
    _typing_timers: dict[str, object] = {}

    def __init__(self):
        super().__init__()
        self.gravando = False
        self.processando = False
        self.recording_idx = 0
        self._olho_dir = (0, 0)
        self._silencio_count = 0
        self._inicio_gravacao = 0.0
        self._boca_delay_timer = None
        self._vad = VADBuffer()
        self._audio_rms = 0.0
        self._mouth_h = 2.0
        self._mouth_target_h = 2.0
        self._bubbles = []
        self._bubble_frame = 0
        self._bocejando = False
        self._trans_frame = -1
        self._trans_total = 0
        self._PHRASES_START = ["oi", "estou ouvindo", "pode falar", "diz aí", "sim?", "pois não?", "olá", "opa"]
        self._PHRASES_WAKE = ["oi", "sim?", "pois não?", "olá", "opa"]
        self._PHRASES_LISTEN = ["estou ouvindo", "pode falar", "diz aí"]
        self._auto_listen_event = threading.Event()
        self._auto_listen_enabled = False
        self._min_listen_time = 5.0
        self._voice_detected = False

    def compose(self) -> ComposeResult:
        with Container(id="face-wrapper"):
            yield Canvas(W, H, canvas_color=WALL, id="rosto")
        yield Static(STATUS_MAP["idle"], id="status")
        with Horizontal(classes="bubble-row"):
            yield Static("Você:", id="stt-text", classes="bubble")
        with Horizontal(classes="bubble-row"):
            yield Static("[#7a1f18]Luiza:[/]", id="tts-text", classes="bubble bubble-luiza")
        yield Static("", id="ctx-usage")
        yield Static("\[Espaço/Enter] Falar  \[Diga 'Luiza'] Ativar  \[Esc] Sair", id="hint")

    def on_mount(self):
        assistente._identidade_confirmada = False
        assistente.iniciar_stream()
        if self._wake_enabled:
            import detector_palavra
            detector_palavra.start(callback=self._wake_check)
        self._boca_timer = self.set_interval(1 / 18, self._avancar_boca,
                                             pause=True)
        self._silencio_timer = self.set_interval(0.5, self._verificar_silencio,
                                                  pause=True)
        self._bubble_timer = self.set_interval(0.08, self._avancar_bubble,
                                                  pause=True)
        self._trans_timer = self.set_interval(1 / 18, self._step_transicao,
                                                  pause=True)
        assistente.tool_status_callback = lambda name: self.call_from_thread(self._usando_ferramenta, name)
        self._agendar_piscar()
        self._agendar_olhar()
        self._agendar_bocejo()
        self._atualizar_rosto()
        self._saudacao_inicial()

    def _agendar_piscar(self):
        self.set_timer(random.uniform(8, 15), self._piscar)

    def _piscar(self):
        self._piscando = 1
        self._atualizar_rosto()
        self.set_timer(0.05, self._piscar_passo2)

    def _piscar_passo2(self):
        self._piscando = 2
        self._atualizar_rosto()
        self.set_timer(0.1, self._piscar_passo3)

    def _piscar_passo3(self):
        self._piscando = 1
        self._atualizar_rosto()
        self.set_timer(0.05, self._abrir_olhos)

    def _abrir_olhos(self):
        self._piscando = 0
        self._atualizar_rosto()
        self._agendar_piscar()

    def _step_transicao(self):
        self._trans_frame += 1
        if self._trans_frame >= self._trans_total:
            self._trans_frame = -1
            self._trans_timer.pause()
        self._atualizar_rosto()

    def _agendar_bocejo(self):
        self.set_timer(random.uniform(25, 40), self._iniciar_bocejo)

    def _iniciar_bocejo(self):
        if self.expressao != "idle":
            self._agendar_bocejo()
            return
        self._bocejando = True
        self._bocejo_frame = 0
        self._bocejo_timer = self.set_interval(1 / 18, self._avancar_bocejo)

    def _avancar_bocejo(self):
        if self.expressao != "idle":
            self._bocejando = False
            self._bocejo_timer.stop()
            self._bocejo_timer = None
            return
        self._bocejo_frame += 1
        if self._bocejo_frame <= 8:
            self._atualizar_rosto()
        elif self._bocejo_frame <= 24:
            self._atualizar_rosto()
        elif self._bocejo_frame <= 32:
            self._atualizar_rosto()
        else:
            self._bocejando = False
            self._bocejo_timer.stop()
            self._bocejo_timer = None
            self._atualizar_rosto()
            self._agendar_bocejo()

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
            base = _boca_altura(MOUTH_CYCLE[self._frame_boca])
            self._mouth_target_h = max(1, base + int(self._audio_rms * 12))
            self._mouth_target_h = min(7, self._mouth_target_h)
            self._atualizar_rosto()

    def _avancar_bubble(self):
        if self.expressao == "pensando":
            self._atualizar_rosto()
            return
        new = []
        for bx, by, br, age in self._bubbles:
            age += 1
            if age >= 16:
                continue
            new.append((bx, by - 1, min(br + 0.3, 2.5), age))
        self._bubbles = new
        self._bubble_frame += 1
        if self._bubble_frame % 5 == 0 and len(self._bubbles) < 5:
            rx = random.randint(-3, 3)
            self._bubbles.append((FX + 10 + rx, FY - 28, 1.0, 0))
        if self.expressao == "transcrevendo":
            self._atualizar_rosto()

    def _atualizar_rosto(self):
        if self.expressao == "falando" and self._mouth_target_h is not None:
            self._mouth_h += (self._mouth_target_h - self._mouth_h) * 0.55
        canvas = self.query_one("#rosto", Canvas)
        with canvas.batch_refresh():
            desenhar_rosto(canvas, self.expressao, self._frame_boca,
                           self._piscando, self._olho_dir,
                           audio_rms=self._audio_rms, now=time.monotonic(),
                           mouth_h=self._mouth_h if self.expressao == "falando" else None,
                           bubbles=self._bubbles if self.expressao == "transcrevendo" else None,
                           bocejando=self._bocejando,
                           bocejo_frame=self._bocejo_frame if self._bocejando else 0,
                           trans_frame=self._trans_frame,
                           trans_total=self._trans_total)

    def _mudar_expressao(self, expressao):
        if self._boca_delay_timer:
            self._boca_delay_timer.cancel()
            self._boca_delay_timer = None
        self.expressao = expressao
        self._frame_boca = 0
        self._mouth_h = 2.0
        self._mouth_target_h = 2.0
        if self._bocejando:
            self._bocejando = False
            if self._bocejo_timer:
                self._bocejo_timer.stop()
                self._bocejo_timer = None
        dur = {"ouvindo": 8, "transcrevendo": 10, "pensando": 10,
               "falando": 8, "idle": 10, "triste": 12}.get(expressao, 6)
        self._trans_frame = 0
        self._trans_total = dur
        self._trans_timer.resume()
        self._atualizar_rosto()
        status = self.query_one("#status", Static)
        status.update(STATUS_MAP.get(expressao, ""))
        if expressao == "idle":
            try:
                self.query_one("#tts-text", Static).remove_class("bubble-erro")
            except Exception:
                pass
        if expressao == "falando":
            self._boca_delay_timer = self.set_timer(0.5, self._iniciar_animacao_boca)
            self._bubble_timer.pause()
        elif expressao == "transcrevendo":
            self._bubbles = []
            self._bubble_frame = 0
            self._bubble_timer.resume()
            self._boca_timer.pause()
        elif expressao == "pensando":
            self._bubbles = []
            self._boca_timer.pause()
            self._bubble_timer.resume()
        else:
            self._bubbles = []
            self._boca_timer.pause()
            self._bubble_timer.pause()

    def _start_typing(self, widget_id, text, speed=0.04, is_error=False, label=""):
        self._cancel_typing(widget_id)
        target = self.query_one(f"#{widget_id}", Static)
        if is_error:
            target.add_class("bubble-erro")
        else:
            target.remove_class("bubble-erro")
        full_text = f"{label}{text}" if text else label
        idx = 0
        def _step():
            nonlocal idx
            idx += 1
            target.update(full_text[:idx])
            if idx < len(full_text):
                self._typing_timers[widget_id] = self.set_timer(speed, _step)
            else:
                self._typing_timers.pop(widget_id, None)
        _step()

    def _cancel_typing(self, widget_id=None):
        if widget_id is None:
            widget_id = next(iter(self._typing_timers), None)
        if widget_id and widget_id in self._typing_timers:
            timer = self._typing_timers.pop(widget_id)
            timer.stop()

    def _iniciar_animacao_boca(self):
        self._boca_delay_timer = None
        self._boca_timer.resume()

    def _falar_frase_inicio(self, from_wake=None):
        if from_wake is None:
            from_wake = not self._auto_listen_event.is_set()
        if from_wake:
            frase = random.choice(self._PHRASES_WAKE)
        else:
            frase = random.choice(self._PHRASES_LISTEN)
        try:
            assistente.falar(frase)
        except Exception as e:
            debug.log(f"_falar_frase_inicio: erro TTS: {e}")

    def _iniciar_gravacao_com_saudacao(self, from_wake):
        def _falar():
            try:
                self._falar_frase_inicio(from_wake=from_wake)
            except Exception as e:
                debug.log(f"_iniciar_gravacao_com_saudacao: erro TTS: {e}")
            finally:
                min_dur = 5.0 if from_wake else 15.0
                self.call_from_thread(self._iniciar_gravacao, min_dur)
        threading.Thread(target=_falar, daemon=True).start()

    def _iniciar_gravacao(self, min_duration=5.0):
        self._min_listen_time = min_duration
        self._inicio_gravacao = time.monotonic()
        self.gravando = True
        self._voice_detected = False
        self._vad.clear()
        with assistente.buf_lock:
            self.recording_idx = len(assistente.buf)
            debug.log(f"_iniciar_gravacao: buf_len={len(assistente.buf)} recording_idx={self.recording_idx}")
        self._silencio_count = 0
        self._silencio_timer.resume()
        self._mudar_expressao("ouvindo")
        self._start_typing("stt-text", "", label="Você: ")
        self._start_typing("tts-text", "", label="[#7a1f18]Luiza:[/] ")

    def _parar_gravacao(self):
        if not self.gravando:
            return
        self._silencio_timer.pause()
        self.gravando = False
        self.processando = True
        self._mudar_expressao("transcrevendo")
        with assistente.buf_lock:
            effective_idx = min(self.recording_idx, len(assistente.buf))
            chunks = assistente.buf[effective_idx:] if effective_idx < len(assistente.buf) else []
            debug.log(f"_parar_gravacao: buf_len={len(assistente.buf)} recording_idx={self.recording_idx} "
                       f"effective_idx={effective_idx} chunks={len(chunks)}")
        if not chunks:
            # Tenta esperar um pouco o buffer encher
            for _ in range(5):
                time.sleep(0.2)
                with assistente.buf_lock:
                    effective_idx = min(self.recording_idx, len(assistente.buf))
                    chunks = assistente.buf[effective_idx:] if effective_idx < len(assistente.buf) else []
                    debug.log(f"_parar_gravacao: retry buf_len={len(assistente.buf)} chunks={len(chunks)}")
                if chunks:
                    break
        if not chunks:
            debug.log("_parar_gravacao: sem audio novo, voltando ao repouso")
            self.processando = False
            self.recording_idx = 0
            self._mudar_expressao("idle")
            return
        audio = np.concatenate(chunks)
        with assistente.buf_lock:
            del assistente.buf[:effective_idx]
            self.recording_idx = 0
        threading.Thread(target=self._processar, args=(audio,),
                         daemon=True).start()

    def _finalizar_escuta_sem_voz(self):
        self._silencio_timer.pause()
        self.gravando = False
        self.processando = False
        self.recording_idx = 0
        frase = "Quando quiser conversar é só me chamar"
        def _falar():
            try:
                assistente.falar(frase)
            except Exception as e:
                debug.log(f"_finalizar_escuta_sem_voz: erro TTS: {e}")
            finally:
                self.call_from_thread(self._atualizar_tts, frase)
                self.call_from_thread(self._mudar_expressao, "idle")
        threading.Thread(target=_falar, daemon=True).start()

    def _saudacao_inicial(self):
        hora = time.localtime().tm_hour
        if hora < 12:
            saudacao = "Bom dia"
        elif hora < 18:
            saudacao = "Boa tarde"
        else:
            saudacao = "Boa noite"
        frase = f"{saudacao}! Como vai?"
        def _falar():
            try:
                assistente.falar(frase)
            except Exception as e:
                debug.log(f"_saudacao_inicial: erro TTS: {e}")
                self.call_from_thread(self._mudar_expressao, "idle")
                return
            self.call_from_thread(self._atualizar_tts, frase)
            self.call_from_thread(self._iniciar_gravacao, 15.0)
        threading.Thread(target=_falar, daemon=True).start()

    def _verificar_silencio(self):
        if not self.gravando:
            return
        try:
            agora = time.monotonic()
            if agora - self._inicio_gravacao < 0.5:
                return
            with assistente.buf_lock:
                recent = assistente.buf[self.recording_idx:]
            if not recent:
                return
            for chunk in recent:
                self._vad.update(chunk)
            if self._vad.count < 4000:
                return
            rms = self._vad.rms
            if rms >= 0.015:
                self._voice_detected = True
                self._silencio_count = 0
            else:
                self._silencio_count += 1
                if self._silencio_count >= 4:
                    if self._voice_detected:
                        debug.log(f"voice ended, silence 2s, rms={rms:.5f}")
                        self._parar_gravacao()
                    elif agora - self._inicio_gravacao >= self._min_listen_time:
                        debug.log(f"timeout, no voice, rms={rms:.5f}")
                        self._finalizar_escuta_sem_voz()
            self._audio_rms = rms
        except Exception as e:
            import traceback
            import io
            tb = io.StringIO()
            traceback.print_exc(file=tb)
            debug.log(f"_verificar_silencio ERRO: {e}\n{tb.getvalue()}")

    def _processar(self, audio):
        t = threading.Thread(target=self._processar_impl, args=(audio,), daemon=True)
        t.start()
        t.join(timeout=180.0)
        if t.is_alive():
            sd.stop()
            debug.log("_processar: timeout")
            try:
                t.join(timeout=5.0)
            except Exception:
                pass
            self.processando = False
            self.call_from_thread(self._voltar_idle)

    def _processar_impl(self, audio):
        t0_total = time.time()
        self._auto_listen_enabled = False
        try:
            debug.log(f"_processar: audio shape={audio.shape}")
            texto = assistente.transcrever(audio)
            debug.log(f"_processar: stt={time.time()-t0_total:.2f}s texto='{texto}'")
            if not texto:
                self.call_from_thread(self._voltar_idle)
                return

            if texto and reconhecimento_voz.modelo_carregado():
                child_id_voz = reconhecimento_voz.identificar_crianca(audio)
                if child_id_voz and child_id_voz != assistente.child_id_ativo:
                    assistente.child_id_ativo = child_id_voz
                    assistente.mem = memoria.carregar(child_id_voz)
                    debug.log(f"voice_id: crianca identificada por voz: {child_id_voz[:12]}...")

            self.call_from_thread(self._atualizar_stt, texto)
            self.call_from_thread(self._mudar_expressao, "pensando")

            debug.log("chamando perguntar...")
            t1 = time.time()
            resposta = assistente.perguntar(texto)
            debug.log(f"_processar: llm={time.time()-t1:.2f}s total={time.time()-t0_total:.2f}s resposta='{resposta}'")
            self.call_from_thread(self._atualizar_ctx)

            time.sleep(1)

            self.call_from_thread(self._mudar_expressao, "falando")
            self.call_from_thread(self._atualizar_tts, resposta)

            assistente.falar(resposta)
            self._auto_listen_enabled = True
            debug.log(f"_processar: fim={time.time()-t0_total:.2f}s")
        except Exception as e:
            import traceback
            import io
            tb = io.StringIO()
            traceback.print_exc(file=tb)
            debug.log(f"ERRO: {e}\n{tb.getvalue()}")
            self.call_from_thread(self._atualizar_tts_erro, f"Erro: {e}")
        finally:
            debug.log("_processar finalizando")
            if self._auto_listen_enabled:
                self._auto_listen_event.set()
            self.call_from_thread(self._voltar_idle)
            self.processando = False

    def _atualizar_stt(self, texto):
        if texto:
            texto = re.sub(r'\s+', ' ', texto).strip()
        self._start_typing("stt-text", texto or "", label="Você: ")

    def _atualizar_tts(self, texto):
        if texto:
            texto = re.sub(r'\s+', ' ', texto).strip()
        self._start_typing("tts-text", texto or "", label="[#7a1f18]Luiza:[/] ")

    def _atualizar_tts_erro(self, texto):
        texto = re.sub(r'\s+', ' ', texto).strip()
        self._start_typing("tts-text", texto, is_error=True, label="[#7a1f18]Luiza:[/] ")

    def _atualizar_ctx(self):
        s = assistente.ultimo_ctx_status
        tokens = s.get("total_tokens", 0)
        limite = s.get("context_limit", 1)
        pct = s.get("percentual", 0.0)
        pct = max(0.0, min(1.0, pct))
        k_tokens = tokens / 1024
        k_limite = limite / 1024
        if pct >= 0.95:
            cor = "#F44336"
        elif pct >= 0.85:
            cor = "#FFC107"
        else:
            cor = "#999"
        texto = f"[{cor}] {k_tokens:.1f}K / {k_limite:.0f}K  {pct:.0%} ctx[/]"
        try:
            self.query_one("#ctx-usage", Static).update(texto)
        except Exception:
            pass

    def _usando_ferramenta(self, nome):
        nomes_amigaveis = {
            "lembrar_nome": "Lembrando quem é você...",
            "salvar_memoria": "Guardando informação...",
            "carregar_memoria": "Buscando lembranças...",
            "buscar_clima": "Olhando o clima...",
            "buscar_web": "Pesquisando...",
            "calculadora": "Calculando...",
            "pergunta_matematica": "Preparando pergunta...",
        }
        label = nomes_amigaveis.get(nome, f"Usando {nome}...")
        try:
            status = self.query_one("#status", Static)
            status.update(f"[#FF9800]🔧 {label}[/]")
        except Exception:
            pass

    def _voltar_idle(self):
        if self._auto_listen_event.is_set():
            self._auto_listen_event.clear()
            self._auto_listen_enabled = False
            self._iniciar_gravacao_com_saudacao(from_wake=False)
            return
        self._mudar_expressao("idle")
