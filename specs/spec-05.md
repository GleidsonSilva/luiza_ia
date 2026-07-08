# Spec 05 — Transições suaves entre expressões

## Problema
Quando o estado muda (ex: `"idle"` → `"ouvindo"`, `"ouvindo"` → `"pensando"`), a expressão facial troca instantaneamente. Não há morphing ou fade, o que parece abrupto.

## O que implementar

### 1. Mecanismo de transição por interpolação

Em vez de trocar `expressao` diretamente e redesenhar:

```python
# estado_transicao = (expressao_anterior, expressao_nova, progresso)
self._transicao: tuple[str, str, float] | None = None
```

Quando `_mudar_expressao(nova)` for chamado:
1. Salvar `self.expressao` como `expressao_anterior`
2. Definir `self._transicao = (anterior, nova, 0.0)`
3. Iniciar `self._transicao_timer = self.set_interval(1/60, self._step_transicao)`

### 2. Interpolação visual (morphing)

O morphing acontece em 8 passos (~130ms a 60fps):

**Progresso 0.0 → 1.0:**
- **Olhos:** pupilas interpolam posição entre a posição da expressão anterior e a nova
- **Boca:** forma da boca interpola entre os dois estados:
  - Se `"smile"` → `"open_small"`: arco se achata e abre gradualmente
  - Implementar `_boca_interp(c, cx, cy, forma_a, forma_b, t)` que mistura as duas formas
- **Sobrancelhas:** ângulo/inclinação interpola entre os dois estados
- **Blush:** fade in/out (opacidade)

### 3. Transições específicas por par

| De → Para | Duração (frames) | Comportamento |
|-----------|-----------------|---------------|
| idle → ouvindo | 6 | Olhos abrem mais, sobrancelhas sobem, boca de sorriso para leve |
| ouvindo → transcrevendo | 8 | Olhos viram para cima, boca abre surpresa, sobrancelhas curvam |
| transcrevendo → pensando | 8 | Olhos focam, sobrancelhas franzem, boca fecha |
| pensando → falando | 6 | Olhos brilham, sobrancelhas felizes, boca abre |
| falando → idle | 10 | Relaxamento completo: boca fecha, olhos suavizam |
| qualquer → triste | 12 | Transição mais lenta (tristeza é gradual) |

Se o par não estiver mapeado, usar interpolação genérica de 6 frames.

### 4. Estado transiente
- `_expressao_transicao = True` — nesse estado, o rosto é redesenhado a cada tick do timer
- Quando `progresso >= 1.0`, finalizar: `self.expressao = nova`, limpar `_transicao`, pausar timer
- Chamar `_atualizar_rosto()` a cada passo

### 5. Limpeza
Se uma nova transição começar enquanto outra está em andamento, cancelar a anterior imediatamente.

## Critérios de aceite
- [ ] Toda troca de expressão tem transição suave visível
- [ ] Transições não bloqueiam a UI (assíncronas)
- [ ] Boca, olhos e sobrancelhas mudam gradualmente
- [ ] Transições específicas respeitam durações diferentes
- [ ] Não causa flicker nem artefatos visuais

## Código afetado
- `interface.py` — `_mudar_expressao`, novo `_transicao_timer`, `_step_transicao`, `_boca_interp`
- Possível novo arquivo `expressoes.py` com lógica de interpolação (opcional para manter `interface.py` enxuto)
