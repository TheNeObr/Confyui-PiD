# ComfyUI PiD (Version V2)

> [!IMPORTANT]
> **This is the V2 branch** of the ComfyUI PiD custom node. It features mathematically perfect variable-overlap tile blending, resolution widget corrections, speed optimizations, and unified sampling logic.

Custom node for using `nvidia/PiD` in ComfyUI with the native PiD workflow, without relying on `PiD Conditioning` or the core `KSampler`.

## Key Improvements in V2

1.  **Seamless Spatial Tile Blending**: Fully resolves vertical/horizontal grid lines and seams in tiled outputs. Uses a mathematically correct variable-overlap cosine blending mask (`_tile_weight_mask_var`) that handles uneven grids and asymmetric overlaps perfectly.
2.  **Flux2 Quality Fix**: Increases the minimum inference context size for the `flux2` backbone to 512px, preventing color shifts, contrast collapse, and flat textures at tile boundaries.
3.  **UI Resolution Widget Fix**: Corrects the inverted resolution display below the preview from `height x width` to `width x height`.
4.  **Timescale Scope Bug Fix**: Fixes a `NameError` crash (`cannot access free variable 'timescale'`) that occurred when running single-step SDE/velocity models.
5.  **Removed Experimental Frequency Anchoring**: Removed older Fourier/frequency anchoring methods which caused vertical lines, grid patterns, and artifacting.
6.  **Optimized Tiled Batch Execution**: Fully integrates parallel processing of tiles via `tile_batch_size` to scale performance with available VRAM.

## What This Package Does

- Loads the official PiD checkpoints for `flux`, `sd3`, `flux2`, `flux2-klein-4b`, `flux2-klein-9b`, `sdxl`, `qwenimage`, `qwenimage-2512`, `zimage`, and `zimage-turbo`.
- Supports prompt pre-encoding to reduce repeated text encoder cost, including optional external `CLIP` input with ComfyUI `pixeldit` format.
- Supports image encoding with PiD's own encoder to generate compatible latents.
- Keeps the runtime lazy so loading the node does not immediately force the full PiD network and text stack onto GPU.
- Uses a lighter VAE-only path for image encoding and an on-demand internal Gemma 2 loader when no external `CLIP` is connected.
- Preserves the original framing during image encode/decode by aligning with padding instead of destructive crop shifts.
- Decodes `LATENT -> IMAGE` in the standard flow or in tiles for larger resolutions.
- Shows CLI progress for `PiD KSampler` with steps, `it/s`, and ETA, while tiled preview updates progressively in the node.

## Included Nodes

- `PiD Load Model`
  - Selects `backbone` and `checkpoint_variant`.
  - Prepares the PiD checkpoint and VAE assets for the selected backbone.
  - The full runtime stays lazy and only loads when prompt encoding or decoding actually needs it.

- `PiD Encode Prompt`
  - Inputs: `PID_MODEL`, `prompt`.
  - Optional input: `CLIP`.
  - Accepts an external ComfyUI `CLIP` configured for `pixeldit`, avoiding the internal Gemma 2 download when provided.
  - Output: `PID_PROMPT`.

- `PiD Encode Image`
  - Inputs: `PID_MODEL`, `IMAGE`, `encode_tile_size`.
  - Uses only the PiD VAE path to produce a compatible latent.
  - Stores the original image geometry inside the produced latent so the final decode can restore the original framing automatically.
  - Supports `disabled`, `512`, and `1024` tiled encode modes to reduce the encode memory spike on larger images.
  - Output: `LATENT`.

- `PiD Decode Latent`
  - Inputs: `pid_model`, `latent`, `prompt`, `negative_prompt`, `cfg_scale`, `pid_inference_steps`, `seed`, `degrade_sigma`, `lq_conditioning_boost`, `source_denoise_strength`, `source_detail_noise_boost`, `sde_noise_strength`, `color_match`.
  - Optional inputs: `pid_prompt`, `clip`, `image_ref`.
  - Output: `IMAGE`.

- `PiD Decode Latent Tiled`
  - Inputs: `pid_model`, `latent`, `tile_size`, `tile_overlap`, `tile_batch_size`, `seam_refine`, `seam_refine_strength`, `prompt`, `negative_prompt`, `cfg_scale`, `pid_inference_steps`, `seed`, `degrade_sigma`, `lq_conditioning_boost`, `source_denoise_strength`, `source_detail_noise_boost`, `sde_noise_strength`, `tiled_sde_noise_boost`, `color_match`.
  - Optional inputs: `pid_prompt`, `clip`, `image_ref`.
  - Output: `IMAGE`.

- `PiD KSampler`
  - Inputs: `pid_model`, `latent`, `prompt`, `negative_prompt`, `cfg_scale`, `pid_inference_steps`, `seed`, `degrade_sigma`, `lq_conditioning_boost`, `source_denoise_strength`, `source_detail_noise_boost`, `keep_model_loaded_on_gpu`, `use_tiled`, `tile_size`, `tile_overlap`, `tile_batch_size`, `seam_refine`, `seam_refine_strength`, `sde_noise_strength`, `tiled_sde_noise_boost`, `color_match`.
  - Optional inputs: `pid_prompt`, `clip`, `image_ref`.
  - Shows step progress, `it/s`, and ETA in the CLI during generation.
  - In tiled mode, shows an incremental low-resolution preview during tile restoration and replaces it with the final blended composition when done.
  - Output: `IMAGE`.

## Schedulers and Noise Control

### SDE Sampler
The node uses a stochastic sampler (SDE) by default. Since the distilled DMD2 checkpoints are trained strictly under SDE assumptions, the deterministic ODE sampler has been removed to avoid generating noisy, destroyed images.

### Scheduler Options
- **`original`**: The default non-linear timestep list from the loaded student model (traditionally optimized for 4 steps). When running with custom step counts, the node uses linear interpolation over the original schedule rather than rounding indices, preventing duplicate or wasted timesteps.
- **`uniform`**: A linear schedule where timesteps are spaced evenly from the maximum student timestep down to `0.0`.
- **`cosine`**: A schedule based on a cosine curve. It progresses faster in early timesteps and slows down at the end, giving the model more steps to refine details near the target output.
- **`quadratic`**: A quadratic decay curve. Timesteps drop quickly at the beginning and slowly decrease towards the end of the sampling path.

### SDE Noise Strength (`sde_noise_strength`)
- A float slider from `0.0` to `1.0` that interpolates between the conservative deterministic path and full SDE sampling. Lower values now preserve more of the source trajectory instead of replacing it with a weak random color field.
  - **`1.0`**: Full SDE noise injection (default).
  - **`0.0`**: Uses a deterministic update for the intermediate steps instead of injecting random noise. This should reduce extra micro-detail and keep the decode more conservative, not produce a random color field.
  - **`0.1` - `0.5`**: Keeps most of the deterministic update while blending in a small amount of SDE variation for subtle restoration.

### Tiled SDE Noise Boost (`tiled_sde_noise_boost`)
- Applied only when tiled sampling is active. The default `1.15` compensates for the more conservative local tile context without changing direct decoding.
- Increase gradually toward `1.30` or `1.50` when tiled restoration preserves too much of the input. Values above `1.50` may introduce grain or seams.

### Seam Refinement (`seam_refine`)
- Runs an optional second tiled restoration pass after the primary tiled image has been assembled.
- The second pass shifts the tile grid by half a stride, uses the first result as a low-denoise source image, and blends back only the overlap bands and their intersections.
- **V2 Note**: Thanks to the new mathematically perfect variable-overlap cosine blending, **seam refinement is no longer necessary for most outputs**. Turning `seam_refine` off (`False`) is highly recommended as it immediately saves **~44% of total generation time**.

### Restoration Control
- `restoration_strength` was removed because it blended the generated RGB output with the source after sampling. When reconstruction geometry differed, that post-process produced shadows and duplicated edges.
- Use `degrade_sigma` for the model's internal LQ conditioning gate. `0.0` keeps the strongest source conditioning; larger values progressively reduce that conditioning and allow a freer reconstruction.
- Use `lq_conditioning_boost` when `degrade_sigma = 0.0` is still too transformative. It increases source preservation inside the same trained LQ gate before prediction. Start at `0.25`, then try `0.50`; values above `1.0` are intentionally aggressive extrapolation.
- Use `source_denoise_strength` for img2img-style restoration strength inside the sampler. `1.0` starts from full noise as before; lower values start closer to the resized source image and process a shorter noise range. `0.0` returns the resized source without generative reconstruction.
- Use `source_detail_noise_boost` to recover texture while keeping a low `source_denoise_strength`. It scales only the residual SDE detail noise after source init. Start at `1.0`, then try `1.25` to `1.75`; reduce it when grain or color artifacts appear.
- Use `sde_noise_strength` for sampling freedom. Lower values keep the trajectory more conservative without compositing two different RGB images.

## V2 Tiling & Performance Optimization Guide

PiD V2 introduces high-performance tiled sampling and math-perfect blending to completely resolve vertical/horizontal grid seams while giving you control over generation speeds.

### 1. Variable-Overlap Cosine Blending (Grid Lines Fix)
*   **The Problem**: Standard tiling often leaves visible seams, vertical/horizontal stripes, or differences in restoration quality between adjacent blocks.
*   **The V2 Solution**: PiD V2 uses a mathematically rigorous **variable-overlap cosine-faded blending mask** (`_tile_weight_mask_var`). It dynamically calculates the exact top, bottom, left, and right overlaps for each tile on the grid (even when boundaries result in asymmetric tile alignments). The 2D blending weight is generated by multiplying 1D horizontal and vertical cosine ramps, ensuring that the sum of all tile weights at any given pixel is exactly `1.0`. Edge margins are left unfaded to prevent dark borders.
*   **The Result**: Tiles blend together seamlessly in a single pass without needing any post-processing.

### 2. Resolution Widget Inversion Fix
*   The read-only resolution widget underneath the node preview now correctly lists the dimensions as `Width x Height` (the standard ComfyUI format), resolving the inverted resolution text in the UI.

### 3. V2 Speed & VRAM Tuning (How to avoid long generation times)
When working with huge resolutions (e.g., upscaling/restoring 5K to 8K images, which is ~46 Megapixels), a single generation can take upwards of 30 minutes if under-optimized. Use the following guide to speed up generations to **less than 3 minutes**:

*   **Increase `tile_batch_size` (VRAM Scaling)**:
    *   By default, `tile_batch_size` is set to `1`, meaning the GPU processes tiles one-by-one.
    *   If you have a GPU with good VRAM (e.g., RTX 3080/3090/4080/4090 with 12GB–24GB), increase `tile_batch_size` to **`2`**, **`4`**, or **`8`**.
    *   PyTorch will process these tiles in parallel, cutting down CPU/PyTorch overhead and utilizing the GPU fully. This can speed up generation by **2x to 4x**.
*   **Disable `seam_refine`**:
    *   Because V2's variable-overlap cosine blending makes tile edges invisible in the first pass, **`seam_refine` is no longer needed**.
    *   Unchecking/disabling `seam_refine` prevents a second grid-shifted generation pass, immediately saving **~44% of processing time**.
*   **Use `cfg_scale: 1.0` (Double your speed)**:
    *   When `cfg_scale` is greater than 1.0, the model must run two passes per step (conditional and unconditional).
    *   Setting `cfg_scale` to **`1.0`** disables Classifier-Free Guidance (CFG), saving half the network evaluation cost and **cutting generation time in half (50% faster)**.
*   **Understand Tile Auto-Adjustment**:
    *   For the `flux2` backbone, the system auto-adjusts requested `512px` tiles to `1024px` because Flux2 requires a larger context window to avoid color/contrast collapse on borders.
    *   This is actually a speed benefit: a 5Kx8K image needs only **88 tiles** of 1024px, compared to **315 tiles** of 512px. Processing fewer, larger tiles in batch parallel (using `tile_batch_size > 1`) is significantly faster than processing hundreds of small tiles.

## Recommended Flow

1. Load the model with `PiD Load Model`.
2. If you plan to reuse the same prompt, use `PiD Encode Prompt`.
3. Use a `LATENT` compatible with the selected backbone, or generate one with `PiD Encode Image`.
4. Decode with `PiD Decode Latent`, `PiD Decode Latent Tiled`, or `PiD KSampler`.

## Example

- V2 Tiled Workflow: [`PID TILED WORFLOW V2.json`](./PID%20TILED%20WORFLOW%20V2.json)
- Example output image:

![PiD example output](./pid_512-2048_flux1_upscale_examplev2.png)

- Example comparison image showing input and model result:

![PiD input and result comparison](./compare.png)

## Installation

1. Place this folder inside `ComfyUI/custom_nodes/`.
2. Install the dependencies in the ComfyUI environment:

```bash
pip install -r requirements.txt
```

3. Restart ComfyUI.
4. When the PiD weights are downloaded automatically, they are stored inside this custom node folder under `upstream-pid/checkpoints/`.
5. In a typical ComfyUI installation, that means the files will be saved to:

```text
ComfyUI/custom_nodes/ComfyUI-PiD/upstream-pid/checkpoints/
```

## Supported Backbones

- `flux`
- `sd3`
- `flux2`
- `flux2-klein-4b`
- `flux2-klein-9b`
- `sdxl`
- `qwenimage`
- `qwenimage-2512`
- `zimage`
- `zimage-turbo`

## Official Resources And Disclaimer

- Official model repository: [`nvidia/PiD` on Hugging Face](https://huggingface.co/nvidia/PiD)
- Official PiD source repository: [`nv-tlabs/PiD` on GitHub](https://github.com/nv-tlabs/PiD)
- The released PiD models are subject to NVIDIA's own model license and usage terms. Always review the official model card and license before downloading, using, sharing, or adapting any of the provided weights.
- This custom node is developed for personal use and experimentation in a controlled environment.
- Running this node, downloading the models, and using them on your own hardware is the sole responsibility of each user.
- Any use outside the applicable model license terms, usage restrictions, or deployment limitations is the sole responsibility of the user who chooses to do so.

## Important Notes

- The backbone selected in `PiD Load Model` must match the latent family.
- `dinov2` and `siglip` are intentionally not supported in this v2 branch.
- `flux2` with `checkpoint_variant = 2kto4k` uses the upstream `_2606` checkpoint, which replaced the earlier Flux2 2kto4k weight to fix color drift.
- `checkpoint_variant = auto` selects the first valid official variant for the chosen backbone. If an older workflow asks for an unsupported variant on a single-variant backbone, the node automatically switches to that backbone's only valid variant instead of failing during generation.
- `sdxl`, `qwenimage`, and `qwenimage-2512` currently support only the upstream `2kto4k` PiD checkpoint.
- `zimage` and `zimage-turbo` reuse the official PiD Flux checkpoint and Flux VAE path, matching the upstream alias.
- `flux2-klein-4b` and `flux2-klein-9b` reuse the official PiD Flux2 checkpoint and Flux2 VAE path, matching the upstream alias.
- The current architecture stays on the stable PiD wrapper path for actual decode quality, while trimming unnecessary load/prefetch behavior around it.
- The first load downloads weights from the [`nvidia/PiD`](https://huggingface.co/nvidia/PiD) repository.
- Downloaded PiD checkpoint files are stored locally in `upstream-pid/checkpoints/` inside this custom node directory.
- `PiD Load Model` downloads the PiD checkpoint and VAE assets, but it does not pre-download the internal Gemma 2 text encoder anymore.
- The Gemma 2 weights now come from [`Comfy-Org/Lumina_Image_2.0_Repackaged`](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/tree/main/split_files/text_encoders) using the `gemma_2_2b_fp16.safetensors` file instead of the older non-safetensors model path.
- The tokenizer file is cached locally together with the text encoder so the PiD runtime can load the native `pixeldit` CLIP format from disk on later runs.
- Downloaded internal Gemma 2 files are stored under `upstream-pid/checkpoints/text_encoders/gemma-2-2b-it/` inside this custom node directory.
- The internal Gemma 2 is only downloaded on demand when PiD needs to encode prompts without an external ComfyUI `CLIP`.
- If you connect an external ComfyUI `CLIP` loaded with `type = pixeldit`, PiD uses that `clip` input and does not need the internal Gemma 2 at prompt time.
- `PiD Encode Image` now uses a lighter VAE-only runtime instead of forcing the full PiD runtime for image encode.
- The automatic encode correction now uses alignment-safe central padding instead of destructive crop-down sizing, which avoids output shift in image comparisons.
- The latent produced by `PiD Encode Image` stores the original geometry and decode nodes crop the final result back to the original framing automatically for a more pixel-perfect comparison workflow.
- `PiD Encode Image` includes optional tiled encode modes (`512` and `1024`) to reduce VRAM spikes during latent creation.
- Custom rectangular latent resolutions stay rectangular during decode instead of being expanded to a square canvas.
- Development and validation for this custom node were tested on an NVIDIA RTX 3090.
- The recommended minimum GPU memory for practical use is 16 GB of VRAM.
- Environments with 12 GB or 8 GB of VRAM were not tested, but they may still work depending on the workflow and settings used.
- This node uses CUDA. There is no practical CPU support in this wrapper.
- This package does not keep `conditioner`, core `KSampler`, or extra experimental nodes outside the main PiD flow.
- `upstream-pid` is a vendored minimal subset of NVIDIA PiD kept only for the runtime pieces this node uses, not the full original project.
- `PiD Decode Latent Tiled` and `PiD KSampler` use `tile_batch_size` to process multiple same-sized tiles per call and reduce overhead.
- `PiD KSampler` includes `keep_model_loaded_on_gpu` so you can decide whether the PiD network stays resident on GPU after sampling.
- `PiD KSampler` now prints CLI progress with `current/total`, percent, `it/s`, and ETA.
- Tiled preview stays in the preview size configured by ComfyUI, updates incrementally during restoration, and publishes a final blended preview at the end.
- The decode node shows only `height x width` in a compact read-only `resolution` field below the preview, without creating extra graph outputs.
- In tiled sampling, `tile_size` is the effective inference-window limit. A requested `512` tile is processed as a `512` tile instead of being silently expanded to a larger direct-decode window.
- Tiled sampling keeps the full-frame diffusion state and blend accumulators in system RAM, transferring only the active tile window to CUDA to prevent VRAM usage from scaling with the complete output canvas.
- Optional `seam_refine` performs a second shifted-grid pass over the assembled result and feather-blends only the original overlap bands. This improves continuity without increasing the configured inference-window size.
- Green artifacts can also appear in non-tiled decoding when aspect-ratio changes or custom resolutions push the latent outside the model's most stable size/alignment range. This is not limited to `4:3`; the bigger issue is usually latent/grid alignment and using a checkpoint variant outside the resolution range where it is most stable.
- As a practical rule, `2k` is usually the safer choice around a `512` base workflow, while `2kto4k` is usually the safer choice around a `1024` base workflow.
- Using the `2k` checkpoint variant with `1024` in direct non-tiled decoding can still be accepted by the model, but it is more likely to produce green artifacts, color collapse, or unstable results than the same workflow at `512`.
- Tiled decoding is often more stable at larger resolutions because the model processes smaller local regions instead of one large full-frame decode, so `1024` and larger outputs tend to behave better in tiled mode than in direct mode.
- For best stability, prefer generating or encoding directly at the target aspect ratio and keep dimensions aligned to the backbone: multiples of `32` are safer for `flux`, `sd3`, `sdxl`, `qwenimage`, `qwenimage-2512`, `zimage`, and `zimage-turbo`; multiples of `64` are safer for `flux2` and the `flux2-klein` aliases.
- For `flux2` and the `flux2-klein` aliases, the encode path is especially strict because the VAE uses an effective `16x` spatial compression and an internal `2x2` patchification step, so dimensions that drift away from the safe multiples are more likely to fail or produce unstable colors.
- The `nvidia/PiD` model has its own NVIDIA usage terms. Check the model card license before distributing or using it in production.

