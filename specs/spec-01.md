# Spec 01 — Reposicionar e ampliar barras de áudio

## Localização atual
`_barras_audio()` desenha em `(bx=100, by=50)` — canto inferior direito da tela, com altura máxima de 6px, o que a torna quase imperceptível.

## O que mudar

### 1. Nova posição e proporção
- **Posição:** centralizado abaixo do rosto, em `y = 58` (logo acima do corpo/vestido)
- **Largura:** ocupar ~80px centrados em `FX` (60), começando em `FX - 40`
- **Altura máxima:** aumentar de 6 para **12px**
- **Número de barras:** passar de 4 para **8 barras**

### 2. Funcionamento
- Cada barra tem largura de 2px com 1px de espaço entre elas
- A altura de cada barra = `base(3) + wave_sin(phase + i*1.2)*0.5 + audio_influence*4`
- `audio_influence = min(1.0, rms * 40.0)` (igual ao atual)
- `t = time.time() * 6` (um pouco mais lento que os 8 atuais para suavizar)

### 3. Onde desenhar
Chamar `_barras_audio(c, rms)` **apenas nos estados** `"ouvindo"` e `"falando"` (já é o caso hoje — manter assim).

### 4. Cores
- Manter `STAR` (amarelo claro) como cor das barras
- Opcional: graduar cor da base pro topo (ex: mais claro em cima)

## Critérios de aceite
- [ ] Barras aparecem centralizadas abaixo do rosto
- [ ] Barras são visivelmente maiores que antes
- [ ] Barras reagem ao volume de áudio
- [ ] Não aparece em estados idle / pensando / transcrevendo

## Código afetado
- `interface.py` — função `_barras_audio`
