# Spec 02 — VU meter em tempo real (nível de áudio)

## Problema
Quando o usuário está falando (estado `"ouvindo"`), não há indicador visual claro de que o som está sendo captado. As barras de áudio só mostram uma reação genérica, sem um medidor de intensidade.

## O que implementar

### 1. Barra VU horizontal abaixo do rosto
Desenhar uma barra horizontal larga, centralizada em `FX`, na linha `y = 60` (entre as barras de áudio e o corpo).

- **Largura:** 60px (de `FX - 30` a `FX + 30`)
- **Altura:** 4px
- **Preenchimento:** proporcional a `_audio_rms`, com escala logarítmica suave
  - `fill = min(1.0, rms * 50)`
  - Preencher da esquerda para a direita até `FX - 30 + int(60 * fill)`

### 2. Cores graduadas por intensidade
- RMS < 0.01 (silêncio): `Color(100, 180, 100)` — verde fraco
- RMS 0.01–0.05: `Color(180, 220, 80)` — amarelo
- RMS > 0.05: `Color(220, 80, 80)` — vermelho

Usar `_audio_rms` (já atualizado a cada 500ms pelo `_verificar_silencio`) para definir a cor.

### 3. Onde desenhar
Criar função `_vu_meter(c, rms)` e chamar dentro de `desenhar_rosto` quando `expressao == "ouvindo"`.

### 4. Frame arredondado (opcional)
Desenhar uma borda fina ao redor da barra com `Color(120, 120, 120)`.

## Critérios de aceite
- [ ] Barra VU aparece no estado "ouvindo"
- [ ] Barra preenche proporcionalmente ao volume captado
- [ ] Cor muda de verde → amarelo → vermelho conforme intensidade
- [ ] Some quando sai do estado "ouvindo"
- [ ] Não interfere com outros elementos visuais

## Código afetado
- `interface.py` — nova função `_vu_meter`, chamada em `desenhar_rosto`
