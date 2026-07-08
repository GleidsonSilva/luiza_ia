# Spec 06 — Layout responsivo

## Problema
Hoje `W=120`, `H=72`, `FX=60`, `FY=43` são constantes. Se o terminal for menor, a interface quebra. Se for maior, subutiliza o espaço.

## O que implementir

### 1. Detecção do tamanho do terminal
Em `LuizaApp.on_mount()` (ou antes, em `__init__`), obter:
```python
import shutil
term = shutil.get_terminal_size()
term_w, term_h = term.columns, term.lines
```

### 2. Cálculo das constantes proporcionais
Definir **fatores de escala** baseados no terminal vs. resolução base (120x72):

```python
ESCALA_X = min(1.0, term_w / 120)
ESCALA_Y = min(1.0, term_h / 72)
ESCALA = min(ESCALA_X, ESCALA_Y)

W = int(120 * ESCALA)
H = int(72 * ESCALA)
FX = int(60 * ESCALA)
FY = int(43 * ESCALA)
```

Manter resolução base inteira; arredondar para baixo para evitar estouro.

### 3. Adaptação das funções de desenho
Todas as funções que usam coordenadas absolutas precisam ser ajustadas:

- `_filled_oval`, `_filled_circle`, `_olho`, `_boca_*`, `_sobrancelhas`, `_corpo`, `_cabelo_atras`, `_franja`, `_lacinho`, `_tapete`, `_janela`, `_lua_estrelas`, `_barras_audio` — TODAS usam valores hardcoded como `FX`, `FY`, `W`, `H`

**Abordagem recomendada:**
- Já usam `FX`, `FY`, `W`, `H` para o posicionamento central, mas muitos valores secundários são hardcoded (ex: `FX - 16` para olho esquerdo, `FY - 10` para altura dos olhos)
- Criar **constantes derivadas** que são calculadas com base em `FX`/`FY`/escala:

```python
EYE_L = int(FX - 16 * ESCALA)
EYE_R = int(FX + 16 * ESCALA)
EYE_Y = int(FY - 10 * ESCALA)
MOUTH_Y = int(FY + 12 * ESCALA)
# etc. para todos os deslocamentos usados nas funções
```

### 4. Canvas width/height dinâmico
No `compose`, passar `W` e `H` calculados:
```python
yield Canvas(W, H, canvas_color=WALL, id="rosto")
```

No CSS, remover `width: 120; height: 38;` e usar valores dinâmicos ou `1fr`.

### 5. Fallback mínimo
Se `term_w < 60` ou `term_h < 40`, usar resolução mínima (`W=60, H=40, ESCALA=0.5`). Se for menor que isso, mostrar mensagem de erro: "Terminal muito pequeno (mínimo 60x40)".

### 6. Reação a resize
Usar `self.screen.size` em Textual ou `on_resize` event para recalcular e redesenhar quando o terminal for redimensionado.

```python
def on_resize(self, event):
    self._recalcular_tamanhos()
    self._atualizar_rosto()
```

## Critérios de aceite
- [ ] Interface escala para terminais de 60x40 a 200x100
- [ ] Proporções faciais mantidas (não distorcidas)
- [ ] Canvas ocupa proporção correta na tela
- [ ] Resize em tempo de execução funciona
- [ ] Fallback para terminais muito pequenos
- [ ] Nenhum elemento ultrapassa os limites do canvas

## Código afetado
- `interface.py` — variáveis globais, `compose`, `on_mount`, `on_resize`, e praticamente todas as funções de desenho
- `CSS` inline — remover `width: 120; height: 38` fixos
