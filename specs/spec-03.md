# Spec 03 — Spinner de carregamento durante processamento

## Problema
Nos estados `"transcrevendo"` e `"pensando"`, o feedback visual é apenas o texto no `#status`. Não há indicação animada de que algo está acontecendo.

## O que implementar

### 1. Spinner baseado em caracteres no Canvas
Desenhar um spinner animado no canto inferior direito do canvas (posição `(110, 70)`), usando caracteres Braille ou formas geométricas desenhadas pixel a pixel.

Opção recomendada (mais simples): desenhar um círculo pontilhado de 8 pontos girando.

#### Frames do spinner (8 frames):

```
Frame 0: ponto em (0, -3)  — cima
Frame 1: ponto em (2, -2)  — cima-direita
Frame 2: ponto em (3, 0)   — direita
Frame 3: ponto em (2, 2)   — baixo-direita
Frame 4: ponto em (0, 3)   — baixo
Frame 5: ponto em (-2, 2)  — baixo-esquerda
Frame 6: ponto em (-3, 0)  — esquerda
Frame 7: ponto em (-2, -2) — cima-esquerda
```

Cada ponto: `_filled_circle(c, 110 + dx, 70 + dy, 1, STAR)`
Apenas o frame ativo tem brilho máximo; os demais ficam com opacidade reduzida.

### 2. Timer de animação
Criar `self._spinner_frame = 0` e `self._spinner_timer = self.set_interval(0.08, self._avancar_spinner)`.

O timer é ativado nos estados `"transcrevendo"` e `"pensando"` e pausado nos demais.

### 3. Integração com `_avancar_boca` / `_atualizar_rosto`
O spinner é redesenhado a cada tick do timer, chamando `_atualizar_rosto()`.

### 4. Touch visual (opcional)
Em vez de apenas um círculo, usar 4 pontos girando como uma "estrela cadente" — desenha trail (rastro) com opacidade decrescente.

## Critérios de aceite
- [ ] Spinner aparece nos estados "transcrevendo" e "pensando"
- [ ] Spinner some nos demais estados
- [ ] Animação é suave (pelo menos 8–12fps)
- [ ] Não causa flicker em outros elementos

## Código afetado
- `interface.py` — nova função `_spinner`, timer `_spinner_timer`, contador `_spinner_frame`
