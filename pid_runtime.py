from __future__ import annotations

import gc
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

NODE_ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = NODE_ROOT / "upstream-pid"
UPSTREAM_CONFIG = "pid/_src/configs/pid/config.py"
HF_REPO_ID = "nvidia/PiD"

SUPPORTED_BACKBONES = ("flux", "sd3", "flux2")
SUPPORTED_VARIANTS = ("2k", "2kto4k")
SUPPORTED_LATENT_INTERPOLATIONS = ("nearest-exact", "bilinear", "bicubic", "area")
LATENT_CHANNELS = {"flux": 16, "sd3": 16, "flux2": 128}
LATENT_COMPRESSION = {"flux": 8, "sd3": 8, "flux2": 16}
MAX_PROMPT_CACHE_ITEMS = 8

_HANDLE_CACHE: dict[tuple[str, str], "PiDHandle"] = {}
_PROMPT_CACHE: "OrderedDict[tuple[str, str, str], PiDPrompt]" = OrderedDict()
_ACTIVE_RUNTIME: "_PiDLoadedRuntime | None" = None


@dataclass(frozen=True)
class PiDHandle:
    backbone: str
    ckpt_type: str
    pid_scale: int
    latent_channels: int
    latent_compression: int
    input_caption_key: str
    checkpoint_path: str
    device: str = "cuda"


@dataclass
class PiDPrompt:
    caption_embs: torch.Tensor
    attention_mask: torch.Tensor | None
    prompt: str = ""


@dataclass
class _PiDLoadedRuntime:
    cache_key: tuple[str, str]
    model: Any


@dataclass
class _DecodeProgress:
    total: int
    node_id: str | None = None
    current: int = 0

    def __post_init__(self):
        try:
            from comfy.utils import ProgressBar

            self._bar = ProgressBar(max(1, int(self.total)), node_id=self.node_id)
            self._bar.update_absolute(0, self.total)
        except Exception:
            self._bar = None

    def update(self, advance: int = 1, preview=None) -> None:
        self.current = min(self.total, self.current + int(advance))
        if self._bar is not None:
            self._bar.update_absolute(self.current, self.total, preview)


@dataclass(frozen=True)
class _TileDecodeJob:
    start_y: int
    start_x: int
    end_y: int
    end_x: int
    out_y: int
    out_x: int


def _ensure_upstream_path() -> None:
    upstream_str = str(UPSTREAM_ROOT)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)


@contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


def _asset_patterns(backbone: str, ckpt_type: str) -> list[str]:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported PiD backbone: {backbone!r}")
    if ckpt_type not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported PiD variant: {ckpt_type!r}")

    if backbone == "flux":
        vae_patterns = ["checkpoints/ae.safetensors"]
    elif backbone == "sd3":
        vae_patterns = ["checkpoints/sd3_vae/*"]
    else:
        vae_patterns = ["checkpoints/flux2_ae.safetensors"]

    ckpt_name = {
        ("flux", "2k"): "PiD_res2k_sr4x_official_flux_distill_4step",
        ("flux", "2kto4k"): "PiD_res2kto4k_sr4x_official_flux_distill_4step",
        ("sd3", "2k"): "PiD_res2k_sr4x_official_sd3_distill_4step",
        ("sd3", "2kto4k"): "PiD_res2kto4k_sr4x_official_sd3_distill_4step",
        ("flux2", "2k"): "PiD_res2k_sr4x_official_flux2_distill_4step",
        ("flux2", "2kto4k"): "PiD_res2kto4k_sr4x_official_flux2_distill_4step",
    }[(backbone, ckpt_type)]

    return [f"checkpoints/{ckpt_name}/*", *vae_patterns]


def _tokenizer_overrides(backbone: str) -> list[str]:
    if backbone == "flux":
        vae_path = UPSTREAM_ROOT / "checkpoints" / "ae.safetensors"
    elif backbone == "sd3":
        vae_path = UPSTREAM_ROOT / "checkpoints" / "sd3_vae" / "vae" / "diffusion_pytorch_model.safetensors"
    elif backbone == "flux2":
        vae_path = UPSTREAM_ROOT / "checkpoints" / "flux2_ae.safetensors"
    else:
        raise ValueError(f"Unsupported PiD backbone: {backbone!r}")

    return [f"+model.config.tokenizer.vae_pth={vae_path.resolve().as_posix()}"]


def _ensure_assets(backbone: str, ckpt_type: str) -> None:
    patterns = _asset_patterns(backbone, ckpt_type)
    missing = [pattern for pattern in patterns if not list(UPSTREAM_ROOT.glob(pattern))]
    if not missing:
        return

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(UPSTREAM_ROOT),
        allow_patterns=patterns,
    )


def _empty_cuda_cache() -> None:
    try:
        import comfy.model_management

        comfy.model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()


def _set_module_device(module: Any, device: str) -> None:
    if module is None or not hasattr(module, "to"):
        return
    module.to(device)


def _set_module_eval(module: Any) -> None:
    if module is None or not hasattr(module, "eval"):
        return
    module.eval()


def _disable_module_grads(module: Any) -> None:
    if module is None or not hasattr(module, "requires_grad_"):
        return
    module.requires_grad_(False)


def _offload_aux_modules(model: Any) -> None:
    _set_module_device(getattr(model, "text_encoder", None), "cpu")
    _set_module_device(getattr(model, "vae_encoder", None), "cpu")
    _empty_cuda_cache()


def _prepare_model_for_inference(model: Any) -> None:
    precision = getattr(model, "precision", torch.float32)
    if hasattr(model, "net") and model.net is not None:
        model.net = model.net.to(device="cuda", dtype=precision)
        _set_module_eval(model.net)
        _disable_module_grads(model.net)
    if getattr(model, "text_encoder", None) is not None:
        _set_module_eval(model.text_encoder)
        _disable_module_grads(model.text_encoder)
    if getattr(model, "vae_encoder", None) is not None:
        _set_module_eval(model.vae_encoder)
        _disable_module_grads(model.vae_encoder)
    _set_module_eval(model)
    _offload_aux_modules(model)


def _clear_prompt_cache(prefix: tuple[str, str] | None = None) -> None:
    if prefix is None:
        _PROMPT_CACHE.clear()
        return
    for key in [key for key in _PROMPT_CACHE if key[:2] == prefix]:
        _PROMPT_CACHE.pop(key, None)


def _release_active_runtime() -> None:
    global _ACTIVE_RUNTIME

    runtime = _ACTIVE_RUNTIME
    if runtime is None:
        return

    model = runtime.model
    for module_name in ("net", "text_encoder", "vae_encoder"):
        _set_module_device(getattr(model, module_name, None), "cpu")

    _ACTIVE_RUNTIME = None
    _clear_prompt_cache(runtime.cache_key)
    del model
    gc.collect()
    _empty_cuda_cache()


def _load_runtime(backbone: str, ckpt_type: str) -> tuple[PiDHandle, Any]:
    global _ACTIVE_RUNTIME

    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported PiD backbone: {backbone!r}")
    if ckpt_type not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported PiD variant: {ckpt_type!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("PiD precisa de CUDA para carregar e inferir no ComfyUI.")

    cache_key = (backbone, ckpt_type)
    if _ACTIVE_RUNTIME is not None and _ACTIVE_RUNTIME.cache_key == cache_key:
        return _HANDLE_CACHE[cache_key], _ACTIVE_RUNTIME.model

    _release_active_runtime()
    _ensure_upstream_path()
    _ensure_assets(backbone, ckpt_type)

    from pid._src.inference.checkpoint_registry import get_pid_checkpoint
    from pid._src.utils.model_loader import load_model_from_checkpoint

    pid_checkpoint = get_pid_checkpoint(backbone, ckpt_type)
    checkpoint_path = (UPSTREAM_ROOT / pid_checkpoint.checkpoint_path).resolve()
    with _pushd(UPSTREAM_ROOT):
        model, _config = load_model_from_checkpoint(
            experiment_name=pid_checkpoint.experiment,
            checkpoint_path=str(checkpoint_path),
            config_file=UPSTREAM_CONFIG,
            enable_fsdp=False,
            experiment_opts=_tokenizer_overrides(backbone),
            strict=False,
        )

    _prepare_model_for_inference(model)
    handle = PiDHandle(
        backbone=backbone,
        ckpt_type=ckpt_type,
        pid_scale=pid_checkpoint.pid_scale,
        latent_channels=LATENT_CHANNELS[backbone],
        latent_compression=LATENT_COMPRESSION[backbone],
        input_caption_key=model.config.input_caption_key,
        checkpoint_path=str(checkpoint_path),
        device="cuda",
    )
    _HANDLE_CACHE[cache_key] = handle
    _ACTIVE_RUNTIME = _PiDLoadedRuntime(cache_key=cache_key, model=model)
    return handle, model


def load_pid_model(backbone: str, ckpt_type: str) -> PiDHandle:
    handle, _model = _load_runtime(backbone, ckpt_type)
    return handle


def _get_model(handle: PiDHandle) -> Any:
    _handle, model = _load_runtime(handle.backbone, handle.ckpt_type)
    return model


def _normalize_pid_prompt(pid_prompt: Any) -> PiDPrompt:
    if isinstance(pid_prompt, PiDPrompt):
        return pid_prompt
    if isinstance(pid_prompt, dict) and "caption_embs" in pid_prompt:
        return PiDPrompt(
            caption_embs=pid_prompt["caption_embs"],
            attention_mask=pid_prompt.get("attention_mask"),
            prompt=str(pid_prompt.get("prompt", "")),
        )
    raise TypeError("O prompt do PiD precisa ser um PiDPrompt ou dict com 'caption_embs'.")


def _encode_prompt_once(handle: PiDHandle, prompt: str) -> PiDPrompt:
    cache_key = (handle.backbone, handle.ckpt_type, prompt)
    cached = _PROMPT_CACHE.get(cache_key)
    if cached is not None:
        _PROMPT_CACHE.move_to_end(cache_key)
        return cached

    model = _get_model(handle)
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError("O modelo PiD carregado nao possui text_encoder.")

    _set_module_device(text_encoder, handle.device)
    with torch.inference_mode():
        caption_embs, attention_mask = model._encode_text_raw([prompt])
    _offload_aux_modules(model)

    prompt_value = PiDPrompt(
        caption_embs=caption_embs.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        attention_mask=attention_mask.detach().to(device="cpu", dtype=torch.int64).contiguous(),
        prompt=prompt,
    )
    _PROMPT_CACHE[cache_key] = prompt_value
    _PROMPT_CACHE.move_to_end(cache_key)
    while len(_PROMPT_CACHE) > MAX_PROMPT_CACHE_ITEMS:
        _PROMPT_CACHE.popitem(last=False)
    return prompt_value


def encode_prompt(handle: PiDHandle, prompt: str) -> dict[str, torch.Tensor | str]:
    encoded = _encode_prompt_once(handle, prompt)
    return {
        "caption_embs": encoded.caption_embs,
        "attention_mask": encoded.attention_mask,
        "prompt": encoded.prompt,
    }


def _extract_latent_tensor(latent: Any) -> torch.Tensor:
    if isinstance(latent, dict):
        if "samples" not in latent:
            raise KeyError("O LATENT recebido nao contem a chave 'samples'.")
        latent_tensor = latent["samples"]
    elif torch.is_tensor(latent):
        latent_tensor = latent
    else:
        raise TypeError("O input LATENT precisa ser um dict do ComfyUI ou um torch.Tensor.")

    if latent_tensor.ndim != 4:
        raise ValueError(f"Esperado latent 4D [B,C,H,W], recebido {tuple(latent_tensor.shape)}.")
    return latent_tensor


def _extract_image_tensor(image: Any) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("O input IMAGE precisa ser um torch.Tensor do ComfyUI.")
    if image.ndim != 4:
        raise ValueError(f"Esperado IMAGE 4D [B,H,W,C], recebido {tuple(image.shape)}.")
    if image.shape[-1] != 3:
        raise ValueError(f"Esperado IMAGE com 3 canais em [B,H,W,C], recebido {tuple(image.shape)}.")
    return image


def encode_image_to_latent(handle: PiDHandle, image: Any) -> dict[str, torch.Tensor]:
    model = _get_model(handle)
    image_tensor = _extract_image_tensor(image)

    chw_image = image_tensor.permute(0, 3, 1, 2).contiguous()
    vae_input = (chw_image * 2.0 - 1.0).clamp(-1.0, 1.0).to(device=handle.device, dtype=torch.float32)

    _set_module_device(getattr(model, "vae_encoder", None), handle.device)
    with torch.inference_mode():
        latent = model.encode_lq_latent(vae_input)
    _offload_aux_modules(model)

    if latent.ndim != 4:
        raise ValueError(f"Encoder do PiD retornou latent invalido: {tuple(latent.shape)}.")
    if latent.shape[1] != handle.latent_channels:
        raise ValueError(
            f"Encoder do PiD para '{handle.backbone}' retornou {latent.shape[1]} canais, "
            f"mas o modelo espera {handle.latent_channels}."
        )

    return {"samples": latent.float().cpu()}


def _resize_spatial_tensor(tensor: torch.Tensor, size: tuple[int, int], interpolation: str) -> torch.Tensor:
    if tensor.ndim < 3:
        return tensor

    original_dtype = tensor.dtype
    work_tensor = tensor.float()
    kwargs: dict[str, Any] = {"size": size, "mode": interpolation}
    if interpolation in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    resized = F.interpolate(work_tensor, **kwargs)
    return resized.to(dtype=original_dtype)


def resize_latent(latent: Any, latent_scale: float, interpolation: str) -> Any:
    if interpolation not in SUPPORTED_LATENT_INTERPOLATIONS:
        raise ValueError(f"Interpolacao de latent nao suportada: {interpolation!r}")
    if latent_scale <= 0:
        raise ValueError("latent_scale precisa ser maior que zero.")

    latent_tensor = _extract_latent_tensor(latent)
    if abs(float(latent_scale) - 1.0) < 1e-8:
        return dict(latent) if isinstance(latent, dict) else latent_tensor

    new_h = max(1, int(round(latent_tensor.shape[-2] * float(latent_scale))))
    new_w = max(1, int(round(latent_tensor.shape[-1] * float(latent_scale))))
    resized_samples = _resize_spatial_tensor(latent_tensor, (new_h, new_w), interpolation)

    if torch.is_tensor(latent):
        return resized_samples

    resized_latent = dict(latent)
    resized_latent["samples"] = resized_samples
    if "noise_mask" in resized_latent and torch.is_tensor(resized_latent["noise_mask"]):
        noise_mask = resized_latent["noise_mask"]
        if noise_mask.ndim == 2:
            noise_mask = noise_mask.unsqueeze(0).unsqueeze(0)
            resized_noise_mask = _resize_spatial_tensor(noise_mask, (new_h, new_w), "nearest-exact").squeeze(0).squeeze(0)
        elif noise_mask.ndim == 3:
            resized_noise_mask = _resize_spatial_tensor(noise_mask.unsqueeze(1), (new_h, new_w), "nearest-exact").squeeze(1)
        elif noise_mask.ndim == 4:
            resized_noise_mask = _resize_spatial_tensor(noise_mask, (new_h, new_w), "nearest-exact")
        else:
            resized_noise_mask = noise_mask
        resized_latent["noise_mask"] = resized_noise_mask
    return resized_latent


def _repeat_pid_prompt(pid_prompt: PiDPrompt, repeat_blocks: int, base_batch: int) -> PiDPrompt:
    if repeat_blocks <= 1:
        return pid_prompt

    caption_embs = pid_prompt.caption_embs
    attention_mask = pid_prompt.attention_mask
    prompt_batch = int(caption_embs.shape[0])
    target_batch = base_batch * repeat_blocks

    if prompt_batch == 1:
        caption_embs = caption_embs.expand(target_batch, -1, -1).contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask.expand(target_batch, -1).contiguous()
    elif prompt_batch == base_batch:
        caption_embs = caption_embs.repeat((repeat_blocks, 1, 1)).contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((repeat_blocks, 1)).contiguous()
    elif prompt_batch != target_batch:
        raise ValueError(
            f"caption_embs precisa ter batch 1, {base_batch} ou {target_batch}, mas recebeu {prompt_batch}."
        )

    return PiDPrompt(caption_embs=caption_embs, attention_mask=attention_mask, prompt=pid_prompt.prompt)


def _make_decode_noise(
    batch_size: int,
    output_h: int,
    output_w: int,
    device: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn((batch_size, 3, output_h, output_w), device=device, dtype=dtype, generator=gen)


def _iter_tile_jobs(
    latent_tensor: torch.Tensor,
    compression: int,
    pid_scale: int,
    tile_latent: int,
    overlap_latent: int,
) -> list[_TileDecodeJob]:
    starts_y = _compute_tile_starts(int(latent_tensor.shape[-2]), tile_latent, overlap_latent)
    starts_x = _compute_tile_starts(int(latent_tensor.shape[-1]), tile_latent, overlap_latent)
    jobs: list[_TileDecodeJob] = []
    for start_y in starts_y:
        for start_x in starts_x:
            end_y = min(start_y + tile_latent, int(latent_tensor.shape[-2]))
            end_x = min(start_x + tile_latent, int(latent_tensor.shape[-1]))
            jobs.append(
                _TileDecodeJob(
                    start_y=start_y,
                    start_x=start_x,
                    end_y=end_y,
                    end_x=end_x,
                    out_y=start_y * compression * pid_scale,
                    out_x=start_x * compression * pid_scale,
                )
            )
    return jobs


def _decode_samples(
    handle: PiDHandle,
    latent_tensor: torch.Tensor,
    pid_prompt: PiDPrompt,
    cfg_scale: float,
    pid_inference_steps: int,
    seed: int,
    degrade_sigma: float,
    progress: _DecodeProgress | None = None,
    preview_enabled: bool = False,
    noise: torch.Tensor | None = None,
    progress_advance: int = 1,
) -> torch.Tensor:
    if latent_tensor.shape[1] != handle.latent_channels:
        raise ValueError(
            f"Backbone '{handle.backbone}' espera {handle.latent_channels} canais no latent, "
            f"mas recebeu {latent_tensor.shape[1]}."
        )

    model = _get_model(handle)
    device = handle.device
    baseline_h = int(latent_tensor.shape[-2]) * handle.latent_compression
    baseline_w = int(latent_tensor.shape[-1]) * handle.latent_compression
    output_h = baseline_h * handle.pid_scale
    output_w = baseline_w * handle.pid_scale
    batch_size = int(latent_tensor.shape[0])

    precision = getattr(model, "precision", torch.float32)
    caption_embs = pid_prompt.caption_embs
    if caption_embs.ndim != 3:
        raise ValueError(f"caption_embs invalido para PiD: {tuple(caption_embs.shape)}.")
    if caption_embs.shape[0] == 1 and batch_size > 1:
        caption_embs = caption_embs.expand(batch_size, -1, -1)
    elif caption_embs.shape[0] != batch_size:
        raise ValueError(
            f"caption_embs precisa ter batch 1 ou {batch_size}, mas recebeu {caption_embs.shape[0]}."
        )

    caption_embs = caption_embs.to(device=device, dtype=precision)
    latent_state = latent_tensor.to(device=device, dtype=precision)
    sigma_tensor = torch.full((batch_size,), float(degrade_sigma), device=device, dtype=torch.float32)

    gen = torch.Generator(device=device).manual_seed(int(seed))
    if noise is None:
        noise = torch.randn((batch_size, 3, output_h, output_w), device=device, dtype=precision, generator=gen)
    else:
        expected_shape = (batch_size, 3, output_h, output_w)
        if tuple(noise.shape) != expected_shape:
            raise ValueError(f"Ruido inicial invalido para PiD: esperado {expected_shape}, recebido {tuple(noise.shape)}.")
        noise = noise.to(device=device, dtype=precision)
    autocast_ctx = torch.autocast("cuda", dtype=model.autocast_dtype) if getattr(model, "autocast_dtype", None) else nullcontext()
    net = model.net
    net.eval()

    def _preview_tuple_from_x0(x0: torch.Tensor):
        if not preview_enabled:
            return None
        image = (x0[:1].float().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
        arr = (image[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
        preview = Image.fromarray(arr)
        try:
            from comfy.cli_args import args

            preview_size = int(getattr(args, "preview_size", 512))
        except Exception:
            preview_size = 512
        if preview_size > 0:
            preview.thumbnail((preview_size, preview_size))
        return ("JPEG", preview, preview_size)

    with torch.inference_mode():
        effective_steps = int(pid_inference_steps) if pid_inference_steps is not None else int(model.config.student_sample_steps)
        student_sample_type = getattr(model.config, "student_sample_type", "sde")
        prediction_type = getattr(model.config, "prediction_type", "velocity")
        student_timestep = float(getattr(model.config, "student_timestep", 1.0))
        if effective_steps == 1:
            t_student = torch.full((batch_size,), student_timestep, device=device, dtype=torch.float32)
            t_student_scaled = t_student * model.fm_trainer.timescale
            with autocast_ctx:
                v_student = net(
                    noise,
                    t_student_scaled,
                    caption_embs,
                    lq_video_or_image=None,
                    lq_latent=latent_state,
                    degrade_sigma=sigma_tensor,
                )
                x0_student = model._velocity_to_x0(noise, v_student, t_student)
            if progress is not None:
                progress.update(advance=progress_advance, preview=_preview_tuple_from_x0(x0_student))
        else:
            t_list = model._get_t_list(device=torch.device(device), num_steps=effective_steps)
            x = noise
            timescale = model.fm_trainer.timescale
            with autocast_ctx:
                for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
                    t_cur_batch = t_cur.expand(batch_size)
                    t_cur_scaled = t_cur_batch * timescale

                    v_pred = net(
                        x,
                        t_cur_scaled,
                        caption_embs,
                        lq_video_or_image=None,
                        lq_latent=latent_state,
                        degrade_sigma=sigma_tensor,
                    )

                    if t_next.item() > 0:
                        if student_sample_type == "ode":
                            v_for_step = model._net_output_to_velocity(x, v_pred, t_cur_batch, prediction_type)
                            dt = t_next - t_cur
                            x = x + dt * v_for_step
                            preview_x0 = model._velocity_to_x0(x, v_pred, t_cur_batch)
                        else:
                            x0_pred = model._velocity_to_x0(x, v_pred, t_cur_batch)
                            eps_infer = torch.randn(
                                x0_pred.shape,
                                device=x0_pred.device,
                                dtype=x0_pred.dtype,
                                generator=gen,
                            )
                            s = [batch_size] + [1] * (x.ndim - 1)
                            t_next_bcast = t_next.reshape(1).expand(s)
                            x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
                            preview_x0 = x0_pred
                    else:
                        x = model._velocity_to_x0(x, v_pred, t_cur_batch)
                        preview_x0 = x

                    if progress is not None:
                        progress.update(advance=progress_advance, preview=_preview_tuple_from_x0(preview_x0))
            x0_student = x
    return x0_student.clamp(-1, 1).unsqueeze(2)


def _samples_to_comfy_image(samples: torch.Tensor) -> torch.Tensor:
    image = samples.squeeze(2).float().clamp(-1, 1)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return image.permute(0, 2, 3, 1).cpu()


def _compute_tile_starts(total: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0:
        raise ValueError("tile_size precisa ser maior que zero.")
    if overlap < 0:
        raise ValueError("tile_overlap nao pode ser negativo.")
    if overlap >= tile:
        raise ValueError("tile_overlap precisa ser menor que tile_size.")
    if total <= tile:
        return [0]

    stride = tile - overlap
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _tile_weight_mask(
    height: int,
    width: int,
    overlap: int,
    top_edge: bool,
    bottom_edge: bool,
    left_edge: bool,
    right_edge: bool,
    device: torch.device,
) -> torch.Tensor:
    weight_y = torch.ones((height,), dtype=torch.float32, device=device)
    weight_x = torch.ones((width,), dtype=torch.float32, device=device)

    if overlap > 0:
        ramp_y = torch.linspace(1.0 / overlap, 1.0, overlap, dtype=torch.float32, device=device)
        ramp_x = torch.linspace(1.0 / overlap, 1.0, overlap, dtype=torch.float32, device=device)

        if not top_edge:
            weight_y[:overlap] = torch.minimum(weight_y[:overlap], ramp_y)
        if not bottom_edge:
            weight_y[-overlap:] = torch.minimum(weight_y[-overlap:], ramp_y.flip(0))
        if not left_edge:
            weight_x[:overlap] = torch.minimum(weight_x[:overlap], ramp_x)
        if not right_edge:
            weight_x[-overlap:] = torch.minimum(weight_x[-overlap:], ramp_x.flip(0))

    return (weight_y[:, None] * weight_x[None, :]).unsqueeze(0).unsqueeze(-1)


def decode_latent(
    handle: PiDHandle,
    latent: Any,
    prompt: str,
    cfg_scale: float,
    pid_inference_steps: int,
    seed: int,
    degrade_sigma: float,
    pid_prompt: Any = None,
    unique_id: str | None = None,
) -> torch.Tensor:
    prompt_value = _normalize_pid_prompt(pid_prompt) if pid_prompt is not None else _encode_prompt_once(handle, prompt)
    effective_steps = int(pid_inference_steps) if pid_inference_steps is not None else 4
    progress = _DecodeProgress(total=max(1, effective_steps), node_id=unique_id)
    samples = _decode_samples(
        handle=handle,
        latent_tensor=_extract_latent_tensor(latent),
        pid_prompt=prompt_value,
        cfg_scale=cfg_scale,
        pid_inference_steps=pid_inference_steps,
        seed=seed,
        degrade_sigma=degrade_sigma,
        progress=progress,
        preview_enabled=True,
    )
    return _samples_to_comfy_image(samples)


def decode_latent_tiled(
    handle: PiDHandle,
    latent: Any,
    prompt: str,
    cfg_scale: float,
    pid_inference_steps: int,
    seed: int,
    degrade_sigma: float,
    tile_size: int,
    tile_overlap: int,
    tile_batch_size: int = 1,
    pid_prompt: Any = None,
    unique_id: str | None = None,
) -> torch.Tensor:
    latent_tensor = _extract_latent_tensor(latent)
    if latent_tensor.shape[1] != handle.latent_channels:
        raise ValueError(
            f"Backbone '{handle.backbone}' espera {handle.latent_channels} canais no latent, "
            f"mas recebeu {latent_tensor.shape[1]}."
        )

    compression = handle.latent_compression
    if tile_size % compression != 0:
        raise ValueError(f"tile_size precisa ser multiplo de {compression} para o backbone '{handle.backbone}'.")
    if tile_overlap % compression != 0:
        raise ValueError(f"tile_overlap precisa ser multiplo de {compression} para o backbone '{handle.backbone}'.")
    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size precisa ser maior que zero.")

    tile_latent = tile_size // compression
    overlap_latent = tile_overlap // compression
    tile_jobs = _iter_tile_jobs(latent_tensor, compression, handle.pid_scale, tile_latent, overlap_latent)

    baseline_h = int(latent_tensor.shape[-2]) * compression
    baseline_w = int(latent_tensor.shape[-1]) * compression
    output_h = baseline_h * handle.pid_scale
    output_w = baseline_w * handle.pid_scale
    overlap_out = tile_overlap * handle.pid_scale

    batch_size = int(latent_tensor.shape[0])
    output = torch.zeros((batch_size, output_h, output_w, 3), dtype=torch.float32)
    weight_sum = torch.zeros((batch_size, output_h, output_w, 1), dtype=torch.float32)

    prompt_value = _normalize_pid_prompt(pid_prompt) if pid_prompt is not None else _encode_prompt_once(handle, prompt)
    tile_count = len(tile_jobs)
    effective_steps = int(pid_inference_steps) if pid_inference_steps is not None else 4
    progress = _DecodeProgress(total=max(1, tile_count * max(1, effective_steps)), node_id=unique_id)
    model = _get_model(handle)
    precision = getattr(model, "precision", torch.float32)
    full_noise = _make_decode_noise(batch_size, output_h, output_w, handle.device, seed, dtype=precision)

    index = 0
    while index < tile_count:
        first_job = tile_jobs[index]
        first_h = first_job.end_y - first_job.start_y
        first_w = first_job.end_x - first_job.start_x
        job_batch = [first_job]
        index += 1

        while index < tile_count and len(job_batch) < tile_batch_size:
            candidate = tile_jobs[index]
            if (candidate.end_y - candidate.start_y, candidate.end_x - candidate.start_x) != (first_h, first_w):
                break
            job_batch.append(candidate)
            index += 1

        latent_batch = torch.cat(
            [
                latent_tensor[:, :, job.start_y : job.end_y, job.start_x : job.end_x].contiguous()
                for job in job_batch
            ],
            dim=0,
        )
        noise_batch = torch.cat(
            [
                full_noise[
                    :,
                    :,
                    job.out_y : job.out_y + (job.end_y - job.start_y) * compression * handle.pid_scale,
                    job.out_x : job.out_x + (job.end_x - job.start_x) * compression * handle.pid_scale,
                ].contiguous()
                for job in job_batch
            ],
            dim=0,
        )
        tile_samples = _decode_samples(
            handle=handle,
            latent_tensor=latent_batch,
            pid_prompt=_repeat_pid_prompt(prompt_value, len(job_batch), batch_size),
            cfg_scale=cfg_scale,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            progress=progress,
            preview_enabled=False,
            noise=noise_batch,
            progress_advance=len(job_batch),
        )
        tile_images = _samples_to_comfy_image(tile_samples)

        for offset, job in enumerate(job_batch):
            tile_image = tile_images[offset * batch_size : (offset + 1) * batch_size]
            tile_h = int(tile_image.shape[1])
            tile_w = int(tile_image.shape[2])
            weight = _tile_weight_mask(
                height=tile_h,
                width=tile_w,
                overlap=min(overlap_out, tile_h // 2, tile_w // 2),
                top_edge=job.start_y == 0,
                bottom_edge=job.end_y == int(latent_tensor.shape[-2]),
                left_edge=job.start_x == 0,
                right_edge=job.end_x == int(latent_tensor.shape[-1]),
                device=tile_image.device,
            )
            output[:, job.out_y : job.out_y + tile_h, job.out_x : job.out_x + tile_w, :] += tile_image * weight
            weight_sum[:, job.out_y : job.out_y + tile_h, job.out_x : job.out_x + tile_w, :] += weight

        composed = output / weight_sum.clamp_min(1e-6)
        preview = None
        try:
            arr = (composed[0].clamp(0, 1).cpu().numpy() * 255.0).astype("uint8")
            preview_image = Image.fromarray(arr)
            from comfy.cli_args import args

            preview_size = int(getattr(args, "preview_size", 512))
            if preview_size > 0:
                preview_image.thumbnail((preview_size, preview_size))
            preview = ("JPEG", preview_image, preview_size)
        except Exception:
            preview = None
        progress.update(advance=0, preview=preview)

    return output / weight_sum.clamp_min(1e-6)


def pid_ksampler(
    handle: PiDHandle,
    latent: Any,
    prompt: str,
    pid_inference_steps: int,
    seed: int,
    degrade_sigma: float,
    use_tiled: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 64,
    tile_batch_size: int = 1,
    pid_prompt: Any = None,
    unique_id: str | None = None,
) -> torch.Tensor:
    if use_tiled:
        return decode_latent_tiled(
            handle=handle,
            latent=latent,
            prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            pid_prompt=pid_prompt,
            unique_id=unique_id,
        )

    return decode_latent(
        handle=handle,
        latent=latent,
        prompt=prompt,
        cfg_scale=1.0,
        pid_inference_steps=pid_inference_steps,
        seed=seed,
        degrade_sigma=degrade_sigma,
        pid_prompt=pid_prompt,
        unique_id=unique_id,
    )
