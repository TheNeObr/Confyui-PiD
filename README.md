# ComfyUI PiD

Custom node para usar o `nvidia/PiD` no ComfyUI no fluxo proprio do PiD, sem depender de `PiD Conditioning` ou `KSampler`.

## O que este pacote faz

- Carrega os checkpoints oficiais do PiD para `flux`, `sd3` e `flux2`.
- Permite pre-encode do prompt para reduzir custo repetido do text encoder.
- Permite encode de imagem com o encoder do proprio PiD para gerar latent compativel.
- Faz decode de `LATENT -> IMAGE` no fluxo normal ou em tiles para resolucoes maiores.

## Nodes mantidos

- `PiD Load Model`
  - Seleciona `backbone` e `checkpoint_variant`.
  - Mantem apenas o runtime ativo do modelo atual.

- `PiD Encode Prompt`
  - Entradas: `PID_MODEL`, `prompt`.
  - Saida: `PID_PROMPT`.

- `PiD Encode Image`
  - Entradas: `PID_MODEL`, `IMAGE`.
  - Saida: `LATENT`.

- `PiD Decode Latent`
  - Entradas: `PID_MODEL`, `LATENT`, `prompt`, `pid_inference_steps`, `seed`, `degrade_sigma`.
  - Entrada opcional: `PID_PROMPT`.
  - Saida: `IMAGE`.

- `PiD Decode Latent Tiled`
  - Entradas: `PID_MODEL`, `LATENT`, `tile_size`, `tile_overlap`, `prompt`, `pid_inference_steps`, `seed`, `degrade_sigma`.
  - Entrada opcional: `PID_PROMPT`.
  - Saida: `IMAGE`.

## Fluxo recomendado

1. Carregue o modelo com `PiD Load Model`.
2. Se for reutilizar o mesmo prompt, use `PiD Encode Prompt`.
3. Use um `LATENT` compativel com o backbone, ou gere um com `PiD Encode Image`.
4. Faça o decode com `PiD Decode Latent` ou `PiD Decode Latent Tiled`.

## Instalacao

1. Coloque esta pasta dentro de `ComfyUI/custom_nodes/`.
2. Instale as dependencias no ambiente do ComfyUI:

```bash
pip install -r requirements.txt
```

3. Reinicie o ComfyUI.

## Backbones suportados

- `flux`
- `sd3`
- `flux2`

## Observacoes importantes

- O backbone escolhido no `PiD Load Model` precisa combinar com a familia do latent.
- O primeiro carregamento baixa pesos do repositorio [`nvidia/PiD`](https://huggingface.co/nvidia/PiD).
- O runtime do PiD tambem carrega o text encoder `Efficient-Large-Model/gemma-2-2b-it`, entao o primeiro uso exige bastante VRAM, RAM e disco.
- O node usa CUDA. Nao ha suporte pratico a CPU neste wrapper.
- Este pacote nao mantem nodes de `conditioner`, `KSampler` ou nodes experimentais fora do fluxo principal do PiD.
- O modelo `nvidia/PiD` tem termos de uso proprios da NVIDIA. Confira a licenca no card do modelo antes de distribuir ou usar em producao.

## Validacao local

Os testes unitarios deste repositorio validam:

- contrato dos nodes do ComfyUI;
- conversao de `LATENT` para `IMAGE`;
- validacao de canais por backbone;
- encode de imagem e cache de prompt;
- chamada correta do runtime com mocks.

Executar:

```bash
python -m unittest discover -s tests -v
```
