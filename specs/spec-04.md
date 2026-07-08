# Spec 04 — Microexpressões no estado idle

## Problema
No estado `"idle"`, Luiza só pisca os olhos e mexe o olhar aleatoriamente. O rosto fica parado na mesma expressão, o que parece "congelado".

## O que implementar

Três microexpressões independentes, cada uma com seu próprio timer:

### 1. Respiração (sutil, contínua)
- Leve movimento senoidal no corpo (`_corpo`) e ombros
- Deslocar o desenho do corpo em `y` num ciclo de ~3s: `dy = int(1.5 * sin(t * 2.1))`
- A face (círculo da pele) também sobe/desce ligeiramente: `dy = int(0.5 * sin(t * 2.1))`
- **Timer:** integrado ao `_atualizar_rosto` (já chamado a cada frame) — usar `time.time()` para calcular o offset

### 2. Bocejo (ocasional, ~a cada 25–40s)
- **Gatilho:** timer com `random.uniform(25, 40)`, reinicia após executar
- **Duração:** ~1.5s (90 frames a 60fps)
- **Animação:**
  - 0–0.3s: olhos semifecham (pálpebras descem 2px), boca abre gradualmente para `open_medium`
  - 0.3–0.8s: boca `open_large`, olhos quase fechados
  - 0.8–1.5s: retorno ao sorriso normal
- **Implementação:** criar `_animar_bocejo(progresso)` chamada por um `set_interval` fino ou por `_avancar_boca` modificado

### 3. Inclinação da cabeça (suave, contínua)
- A face inteira oscila lateralmente em `x` com amplitude de 1px e período de 4–6s
- `head_tilt_x = int(1.0 * sin(t * 1.2))` — afeta `FX` de todos os elementos faciais
- **Atenção:** ajustar **todas** as funções de desenho para aceitar um offset central, ou passar `FX` modificado para elas

### 4. Piscada mais natural
- Já existe (`_piscar` a cada 8–15s), mas a piscada atual é instantânea (0.2s fechado)
- **Melhoria:** fazer a pálpebra descer progressivamente em 3 sub-passos (aberto → semi → fechado → semi → aberto), cada um com ~50ms

## Critérios de aceite
- [ ] Respiração visível no corpo e ombros
- [ ] Bocejo ocorre esporadicamente (25–40s) com animação suave
- [ ] Cabeça inclina levemente com oscilação contínua
- [ ] Piscada tem transição gradual (não instantânea)
- [ ] Nenhuma microexpressão interfere nos estados ativos (ouvindo, falando, etc.)
- [ ] Performance não degrada (todas contínuas mas baratas computacionalmente)

## Código afetado
- `interface.py` — funções `_respirar`, `_animar_bocejo`, parâmetros de offset em `desenhar_rosto` e funções auxiliares
