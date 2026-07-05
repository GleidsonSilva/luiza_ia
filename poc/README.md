# POC: voz mais infantil (APENAS pitch, sem aumentar velocidade)

Branch: `poc/voz-infantil`

Textos-base iguais, mesma voz `pf_dora` do app, sem speed extra.

Arquivos:
- `base_pf_dora.wav` — voz atual do app.
- `abordagem_1_scipy_pitch.wav` — pitch mais alto por resampling linear com `scipy`, mantendo duração idêntica.
- `abordagem_2_phasevocoder_pitch.wav` — pitch mais alto por phase vocoder, mantendo duração idêntica.
- `abordagem_3_voice_blend_pf_dora_af_heart.wav` — blending no Kokoro entre `pf_dora` + voz mais aguda, sem edição pós-processamento.

Execução:
```bash
bash poc/gerar_poc.sh
```

Avaliação:
- `afplay poc/base_pf_dora.wav`
- `afplay poc/abordagem_1_scipy_pitch.wav`
- `afplay poc/abordagem_2_phasevocoder_pitch.wav`
- `afplay poc/abordagem_3_voice_blend_pf_dora_af_heart.wav`
