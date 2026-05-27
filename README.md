# ComfyUI PiD

Custom node for using `nvidia/PiD` in ComfyUI with the native PiD workflow, without relying on `PiD Conditioning` or the core `KSampler`.

## What This Package Does

- Loads the official PiD checkpoints for `flux`, `sd3`, and `flux2`.
- Supports prompt pre-encoding to reduce repeated text encoder cost.
- Supports image encoding with PiD's own encoder to generate compatible latents.
- Decodes `LATENT -> IMAGE` in the standard flow or in tiles for larger resolutions.

## Included Nodes

- `PiD Load Model`
  - Selects `backbone` and `checkpoint_variant`.
  - Keeps only the current model runtime active.

- `PiD Encode Prompt`
  - Inputs: `PID_MODEL`, `prompt`.
  - Output: `PID_PROMPT`.

- `PiD Encode Image`
  - Inputs: `PID_MODEL`, `IMAGE`.
  - Output: `LATENT`.

- `PiD Decode Latent`
  - Inputs: `PID_MODEL`, `LATENT`, `prompt`, `pid_inference_steps`, `seed`, `degrade_sigma`.
  - Optional input: `PID_PROMPT`.
  - Output: `IMAGE`.

- `PiD Decode Latent Tiled`
  - Inputs: `PID_MODEL`, `LATENT`, `tile_size`, `tile_overlap`, `tile_batch_size`, `prompt`, `pid_inference_steps`, `seed`, `degrade_sigma`.
  - Optional input: `PID_PROMPT`.
  - Output: `IMAGE`.

- `PiD KSampler`
  - Inputs: `PID_MODEL`, `LATENT`, `prompt`, `pid_inference_steps`, `seed`, `degrade_sigma`, `keep_model_loaded_on_gpu`, `use_tiled`, `tile_size`, `tile_overlap`, `tile_batch_size`.
  - Optional input: `PID_PROMPT`.
  - Output: `IMAGE`.

## Recommended Flow

1. Load the model with `PiD Load Model`.
2. If you plan to reuse the same prompt, use `PiD Encode Prompt`.
3. Use a `LATENT` compatible with the selected backbone, or generate one with `PiD Encode Image`.
4. Decode with `PiD Decode Latent`, `PiD Decode Latent Tiled`, or `PiD KSampler`.

## Example

- Example workflow: [`workflow_pid_flux2_2kto4k_tiled.json`](./workflow_pid_flux2_2kto4k_tiled.json)
- Example output image:

![PiD example output](./pid_512-2048_flux1_upscale_example.png)

- Example comparison image showing input and model result:

![PiD input and result comparison](./compare.png)

## Installation

1. Place this folder inside `ComfyUI/custom_nodes/`.
2. Install the dependencies in the ComfyUI environment:

```bash
pip install -r requirements.txt
```

3. Restart ComfyUI.

## Supported Backbones

- `flux`
- `sd3`
- `flux2`

## Important Notes

- The backbone selected in `PiD Load Model` must match the latent family.
- The first load downloads weights from the [`nvidia/PiD`](https://huggingface.co/nvidia/PiD) repository.
- The PiD runtime also loads the `Efficient-Large-Model/gemma-2-2b-it` text encoder, so the first run requires a significant amount of VRAM, RAM, and disk space.
- Development and validation for this custom node were tested on an NVIDIA RTX 3090.
- The recommended minimum GPU memory for practical use is 16 GB of VRAM.
- Environments with 12 GB or 8 GB of VRAM were not tested, but they may still work depending on the workflow and settings used.
- This node uses CUDA. There is no practical CPU support in this wrapper.
- This package does not keep `conditioner`, core `KSampler`, or extra experimental nodes outside the main PiD flow.
- `upstream-pid` is a vendored minimal subset of NVIDIA PiD kept only for the runtime pieces this node uses, not the full original project.
- `PiD Decode Latent Tiled` and `PiD KSampler` use `tile_batch_size` to process multiple same-sized tiles per call and reduce overhead.
- `PiD KSampler` includes `keep_model_loaded_on_gpu` so you can decide whether the PiD network stays resident on GPU after sampling.
- The decode node shows only `height x width` in a compact read-only `resolution` field below the preview, without creating extra graph outputs.
- Small tiles such as `256` are decoded with extra internal context before the final crop to reduce green tint and color collapse.
- The `nvidia/PiD` model has its own NVIDIA usage terms. Check the model card license before distributing or using it in production.

## Local Validation

The unit tests in this repository validate:

- ComfyUI node contracts;
- `LATENT` to `IMAGE` conversion;
- channel validation per backbone;
- image encoding and prompt cache behavior;
- correct runtime delegation with mocks.

Run:

```bash
python -m unittest discover -s tests -v
```
