from __future__ import annotations

import gc
import os
import sys
import time
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
HF_PID_TEXT_ENCODER_REPO_ID = "Comfy-Org/Lumina_Image_2.0_Repackaged"
HF_PID_TEXT_ENCODER_FILE = "split_files/text_encoders/gemma_2_2b_fp16.safetensors"
HF_PID_TEXT_ENCODER_TOKENIZER_REPO_ID = "Efficient-Large-Model/gemma-2-2b-it"
HF_PID_TEXT_ENCODER_TOKENIZER_FILE = "tokenizer.model"

SUPPORTED_BACKBONES = ("flux", "sd3", "flux2")
SUPPORTED_VARIANTS = ("2k", "2kto4k")
SUPPORTED_LATENT_INTERPOLATIONS = ("nearest-exact", "bilinear", "bicubic", "area")
LATENT_CHANNELS = {"flux": 16, "sd3": 16, "flux2": 128}
LATENT_COMPRESSION = {"flux": 8, "sd3": 8, "flux2": 16}
PID_TEXT_EMBED_DIM = 2304
PID_TEXT_TOKEN_COUNT = 300
MAX_PROMPT_CACHE_ITEMS = 8
AUTOENCODE_TILE_OVERLAP = {"flux": 128, "sd3": 128, "flux2": 128}
# Tiles muito pequenos tendem a colapsar a cor no PiD; decodificamos com contexto maior
# e recortamos o centro para manter a saida pedida sem o desvio verde.
MIN_TILED_DECODE_SIZE = 512

_HANDLE_CACHE: dict[tuple[str, str], "PiDHandle"] = {}
_PROMPT_CACHE: "OrderedDict[tuple[str, str, str], PiDPrompt]" = OrderedDict()
_ACTIVE_RUNTIME: "_PiDLoadedRuntime | None" = None
_VAE_ONLY_CACHE: dict[tuple[str, str], Any] = {}


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


@dataclass(frozen=True)
class _LightPiDConfig:
    backbone: str
    input_caption_key: str
    text_encoder_name: str
    model_max_length: int
    chi_prompt_str: str
    prediction_type: str
    student_timestep: float
    student_sample_steps: int
    student_sample_type: str
    student_t_list: tuple[float, ...] | None
    state_ch: int
    tokenizer_config: Any


@dataclass(frozen=True)
class _LightFlowMatching:
    timescale: float


class _LightPiDModel:
    def __init__(
        self,
        *,
        net: Any,
        config: _LightPiDConfig,
        precision: torch.dtype,
        autocast_dtype: torch.dtype | None,
        fm_timescale: float,
        text_encoder: Any = None,
        vae_encoder: Any = None,
    ) -> None:
        self.net = net
        self.config = config
        self.precision = precision
        self.autocast_dtype = autocast_dtype
        self.fm_trainer = _LightFlowMatching(timescale=float(fm_timescale))
        self.tokenizer = None
        self.text_encoder = text_encoder
        self.vae_encoder = vae_encoder
        self._num_chi_tokens = 0
        if self.text_encoder is not None:
            self._num_chi_tokens = None

    def eval(self):
        return self

    def _ensure_text_encoder_loaded(self) -> None:
        if self.tokenizer is not None and self.text_encoder is not None:
            return

        try:
            text_encoder = _load_native_text_encoder(self.config.text_encoder_name)
            self.tokenizer = getattr(text_encoder, "tokenizer", None)
            self.text_encoder = text_encoder
            self._num_chi_tokens = 0
            return
        except Exception:
            pass

        from pid._src.models.pixeldit_model import _load_text_encoder

        tokenizer, text_encoder = _load_text_encoder(self.config.text_encoder_name, device="cuda")
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self._num_chi_tokens = len(tokenizer.encode(self.config.chi_prompt_str)) if self.config.chi_prompt_str else 0

    @torch.no_grad()
    def _encode_text_raw(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_text_encoder_loaded()

        if _is_comfy_clip_text_encoder(self.text_encoder):
            import comfy.model_management

            comfy.model_management.load_models_gpu([self.text_encoder.patcher], force_full_load=True)
            caption_embs_all = []
            attention_masks_all = []
            for caption in captions:
                encoded = self.text_encoder.encode_from_tokens(
                    self.text_encoder.tokenize(caption),
                    return_dict=True,
                )
                caption_embs_all.append(encoded["cond"])
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones(
                        encoded["cond"].shape[:2],
                        dtype=torch.int64,
                        device=encoded["cond"].device,
                    )
                attention_masks_all.append(attention_mask.to(dtype=torch.int64))
            return torch.cat(caption_embs_all, dim=0), torch.cat(attention_masks_all, dim=0)

        if self.config.chi_prompt_str:
            prompts_all = [self.config.chi_prompt_str + cap for cap in captions]
            max_length_all = self._num_chi_tokens + self.config.model_max_length - 2
        else:
            prompts_all = captions
            max_length_all = self.config.model_max_length

        caption_token = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0]
        select_index = [0] + list(range(-self.config.model_max_length + 1, 0))
        caption_embs = caption_embs[:, select_index]
        emb_masks = caption_token.attention_mask[:, select_index]
        return caption_embs, emb_masks

    def _ensure_vae_encoder_loaded(self) -> None:
        if self.vae_encoder is not None:
            return
        self.vae_encoder = _instantiate_vae_encoder(self.config.backbone, self.config.tokenizer_config)
        if self.config.state_ch > 0 and getattr(self.vae_encoder, "latent_ch", None) != self.config.state_ch:
            raise ValueError(
                f"latent_ch {getattr(self.vae_encoder, 'latent_ch', None)} != state_ch {self.config.state_ch}"
            )

    @torch.no_grad()
    def encode_lq_latent(self, lq_image: torch.Tensor) -> torch.Tensor:
        self._ensure_vae_encoder_loaded()
        if lq_image.ndim == 4:
            lq_image = lq_image.unsqueeze(2)
        latent = self.vae_encoder.encode(lq_image)
        if latent.ndim == 5:
            latent = latent[:, :, 0, :, :]
        return latent

    def _net_output_to_x0(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        if prediction_type == "x0":
            return net_output.to(x_t.dtype)
        if prediction_type == "velocity":
            original_dtype = x_t.dtype
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s)
            return (x_t.double() - t_shaped * net_output.double()).to(original_dtype)
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _net_output_to_velocity(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        if prediction_type == "velocity":
            return net_output
        if prediction_type == "x0":
            original_dtype = x_t.dtype
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s).clamp(min=5e-2)
            return ((x_t.double() - net_output.double()) / t_shaped).to(original_dtype)
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _velocity_to_x0(self, x_t: torch.Tensor, net_output: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._net_output_to_x0(x_t, net_output, t, self.config.prediction_type)

    def _get_t_list(self, device, num_steps: int | None = None) -> torch.Tensor:
        target_steps = num_steps if num_steps is not None else self.config.student_sample_steps
        if self.config.student_t_list is not None:
            full_t = torch.tensor(self.config.student_t_list, device=device, dtype=torch.float32)
            if target_steps != self.config.student_sample_steps:
                indices = torch.linspace(0, len(full_t) - 1, target_steps + 1, device=device).round().long()
                t_list = full_t[indices]
            else:
                t_list = full_t
        else:
            t_list = torch.linspace(
                self.config.student_timestep,
                0.0,
                target_steps + 1,
                device=device,
                dtype=torch.float32,
            )
        if abs(t_list[-1].item()) >= 1e-6:
            raise ValueError("t_list must end at 0")
        return t_list


class _NativePiDModel(_LightPiDModel):
    def __init__(
        self,
        *,
        patcher: Any,
        base_model: Any,
        config: _LightPiDConfig,
        text_encoder: Any = None,
        vae_encoder: Any = None,
    ) -> None:
        inference_dtype = base_model.get_dtype_inference()
        super().__init__(
            net=base_model.diffusion_model,
            config=config,
            precision=inference_dtype,
            autocast_dtype=inference_dtype if inference_dtype in (torch.float16, torch.bfloat16) else None,
            fm_timescale=1000.0,
            text_encoder=text_encoder,
            vae_encoder=vae_encoder,
        )
        self.patcher = patcher
        self.base_model = base_model

    def ensure_model_loaded(self) -> None:
        import comfy.model_management

        comfy.model_management.load_models_gpu([self.patcher], force_full_load=True)
        self.patcher.pre_run()
        self.net = self.base_model.diffusion_model

    def release_native_model(self) -> None:
        import comfy.model_management

        comfy.model_management.unload_model_and_clones(self.patcher)

    def predict_x0(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        caption_embs: torch.Tensor,
        attention_mask: torch.Tensor | None,
        lq_latent: torch.Tensor,
        degrade_sigma: torch.Tensor,
    ) -> torch.Tensor:
        t_scaled = sigma * self.fm_trainer.timescale
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        with autocast_ctx:
            net_output = self.net(
                x,
                t_scaled,
                context=caption_embs,
                attention_mask=attention_mask,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma,
                transformer_options={},
            )
        return self._velocity_to_x0(x, net_output, sigma)


@dataclass
class _DecodeProgress:
    total: int
    node_id: str | None = None
    current: int = 0

    def __post_init__(self):
        self.total = max(1, int(self.total))
        self._started_at = time.perf_counter()
        self._last_cli_render_at = 0.0
        self._cli_finished = False
        self._cli_min_interval = 0.1
        self._stream = getattr(sys, "stderr", None)
        try:
            from comfy.utils import ProgressBar

            self._bar = ProgressBar(self.total, node_id=self.node_id)
            self._bar.update_absolute(0, self.total)
        except Exception:
            self._bar = None
        self._render_cli(force=True)

    def _render_cli(self, force: bool = False) -> None:
        if self._cli_finished:
            return
        stream = self._stream
        if stream is None or not hasattr(stream, "write"):
            return

        now = time.perf_counter()
        is_final = self.current >= self.total
        if not force and not is_final and (now - self._last_cli_render_at) < self._cli_min_interval:
            return

        elapsed = max(now - self._started_at, 1e-6)
        iterations_per_second = self.current / elapsed if self.current > 0 else 0.0
        remaining_steps = max(0, self.total - self.current)
        eta_seconds = (remaining_steps / iterations_per_second) if iterations_per_second > 1e-6 else None
        progress_ratio = min(1.0, self.current / max(1, self.total))
        filled = min(24, int(round(progress_ratio * 24)))
        bar = "#" * filled + "-" * (24 - filled)
        if eta_seconds is None:
            eta_text = "--:--"
        else:
            eta_total_seconds = max(0, int(round(eta_seconds)))
            eta_minutes, eta_secs = divmod(eta_total_seconds, 60)
            eta_hours, eta_minutes = divmod(eta_minutes, 60)
            eta_text = (
                f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
                if eta_hours > 0
                else f"{eta_minutes:02d}:{eta_secs:02d}"
            )
        line = (
            f"\rPiD KSampler [{bar}] "
            f"{self.current}/{self.total} steps "
            f"({progress_ratio * 100:5.1f}%) "
            f"{iterations_per_second:5.2f} it/s "
            f"ETA {eta_text}"
        )
        stream.write(line)
        if hasattr(stream, "flush"):
            stream.flush()
        self._last_cli_render_at = now

        if is_final:
            stream.write("\n")
            if hasattr(stream, "flush"):
                stream.flush()
            self._cli_finished = True

    def _send_legacy_preview(self, preview) -> None:
        if preview is None:
            return
        try:
            from protocol import BinaryEventTypes
            from server import PromptServer

            server_instance = getattr(PromptServer, "instance", None)
            if server_instance is None:
                return
            client_id = getattr(server_instance, "client_id", None)
            server_instance.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview, client_id)
        except Exception:
            pass

    def update(self, advance: int = 1, preview=None, emit_bar: bool = True) -> None:
        self.current = min(self.total, self.current + int(advance))
        self._render_cli(force=emit_bar or self.current >= self.total)
        if emit_bar and self._bar is not None:
            self._bar.update_absolute(self.current, self.total, preview)
        if emit_bar and preview is not None:
            self._send_legacy_preview(preview)


@dataclass(frozen=True)
class _TileDecodeJob:
    start_y: int
    start_x: int
    end_y: int
    end_x: int
    out_y: int
    out_x: int


@dataclass(frozen=True)
class _ExpandedTileDecodeJob:
    target_start_y: int
    target_start_x: int
    target_end_y: int
    target_end_x: int
    decode_start_y: int
    decode_start_x: int
    decode_end_y: int
    decode_end_x: int
    out_y: int
    out_x: int
    crop_y: int
    crop_x: int


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


def _native_chi_prompt() -> str:
    try:
        from comfy.text_encoders import pixeldit as comfy_pixeldit

        return getattr(comfy_pixeldit, "_PIXELDIT_CHI_PROMPT")
    except Exception:
        return (
            'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions '
            "suitable for image generation. Evaluate the level of detail in the user prompt:\n"
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n"
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n"
            "Here are examples of how to transform or refine prompts:\n"
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n"
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\n"
            "User Prompt: "
        )


def _native_pid_config(backbone: str) -> _LightPiDConfig:
    return _LightPiDConfig(
        backbone=backbone,
        input_caption_key="caption",
        text_encoder_name="gemma-2-2b-it",
        model_max_length=300,
        chi_prompt_str=_native_chi_prompt(),
        prediction_type="velocity",
        student_timestep=1.0,
        student_sample_steps=4,
        student_sample_type="sde",
        student_t_list=(0.999, 0.866, 0.634, 0.342, 0.0),
        state_ch=LATENT_CHANNELS[backbone],
        tokenizer_config=None,
    )


def _instantiate_vae_encoder(backbone: str, tokenizer_config: Any = None) -> Any:
    if tokenizer_config is not None:
        from pid._ext.imaginaire.lazy_config import instantiate as lazy_instantiate

        return lazy_instantiate(tokenizer_config)

    if backbone == "flux":
        from pid._src.tokenizers.flux_vae import FluxVAEInterface

        return FluxVAEInterface(vae_pth=(UPSTREAM_ROOT / "checkpoints" / "ae.safetensors").as_posix())
    if backbone == "sd3":
        from pid._src.tokenizers.flux_vae import SD3VAEInterface

        return SD3VAEInterface(
            vae_pth=(UPSTREAM_ROOT / "checkpoints" / "sd3_vae" / "vae" / "diffusion_pytorch_model.safetensors").as_posix()
        )
    if backbone == "flux2":
        from pid._src.tokenizers.flux2_vae import Flux2VAEInterface

        return Flux2VAEInterface(vae_pth=(UPSTREAM_ROOT / "checkpoints" / "flux2_ae.safetensors").as_posix())
    raise ValueError(f"Unsupported PiD backbone: {backbone!r}")


def _patch_native_pid_attention_dtype_mismatch() -> None:
    try:
        import comfy.ldm.pixeldit.model as comfy_pixeldit_model
        import comfy.ldm.pixeldit.modules as comfy_pixeldit_modules
    except Exception:
        return

    joint_attn_cls = getattr(comfy_pixeldit_model, "MMDiTJointAttention", None)
    if joint_attn_cls is not None and not getattr(joint_attn_cls, "_pidnode_dtype_patch", False):
        def patched_joint_forward(self, x, y, pos_img, pos_txt=None, attn_mask=None, transformer_options={}):
            batch_size, image_tokens, _ = x.shape
            _, text_tokens, _ = y.shape
            num_heads = self.num_heads
            head_dim = self.head_dim

            qkv_x = self.qkv_x(x).reshape(batch_size, image_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            qx, kx, vx = qkv_x.unbind(0)
            qx = self.q_norm_x(qx)
            kx = self.k_norm_x(kx)

            qkv_y = self.qkv_y(y).reshape(batch_size, text_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            qy, ky, vy = qkv_y.unbind(0)
            qy = self.q_norm_y(qy)
            ky = self.k_norm_y(ky)

            qx, kx = comfy_pixeldit_model.apply_rope(qx, kx, pos_img[None, None])
            if pos_txt is not None:
                qy, ky = comfy_pixeldit_model.apply_rope(qy, ky, pos_txt[None, None])

            q_joint = torch.cat([qy, qx], dim=2)
            k_joint = torch.cat([ky, kx], dim=2)
            v_joint = torch.cat([vy, vx], dim=2)
            target_dtype = v_joint.dtype
            if q_joint.dtype != target_dtype:
                q_joint = q_joint.to(dtype=target_dtype)
            if k_joint.dtype != target_dtype:
                k_joint = k_joint.to(dtype=target_dtype)

            out_joint = comfy_pixeldit_model.optimized_attention(
                q_joint,
                k_joint,
                v_joint,
                num_heads,
                mask=attn_mask,
                skip_reshape=True,
                skip_output_reshape=True,
                transformer_options=transformer_options,
            )

            out_y = out_joint[:, :, :text_tokens, :].transpose(1, 2).reshape(batch_size, text_tokens, num_heads * head_dim)
            out_x = out_joint[:, :, text_tokens:, :].transpose(1, 2).reshape(batch_size, image_tokens, num_heads * head_dim)
            return self.proj_x(out_x), self.proj_y(out_y)

        joint_attn_cls.forward = patched_joint_forward
        joint_attn_cls._pidnode_dtype_patch = True

    rotary_attn_cls = getattr(comfy_pixeldit_modules, "RotaryAttention", None)
    if rotary_attn_cls is not None and not getattr(rotary_attn_cls, "_pidnode_dtype_patch", False):
        def patched_rotary_forward(self, x, pos, mask=None, transformer_options={}):
            batch_size, token_count, channels = x.shape
            num_heads = self.num_heads
            head_dim = self.head_dim
            qkv = self.qkv(x).reshape(batch_size, token_count, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = comfy_pixeldit_modules.apply_rope(self.q_norm(q), self.k_norm(k), pos[None, None])
            target_dtype = v.dtype
            if q.dtype != target_dtype:
                q = q.to(dtype=target_dtype)
            if k.dtype != target_dtype:
                k = k.to(dtype=target_dtype)
            out = comfy_pixeldit_modules.optimized_attention(
                q,
                k,
                v,
                num_heads,
                mask=mask,
                skip_reshape=True,
                transformer_options=transformer_options,
            )
            return self.proj(out)

        rotary_attn_cls.forward = patched_rotary_forward
        rotary_attn_cls._pidnode_dtype_patch = True


def _native_lq_latent_process_in(backbone: str, latent: torch.Tensor) -> torch.Tensor:
    import comfy.latent_formats

    if backbone == "flux":
        return comfy.latent_formats.Flux().process_in(latent)
    if backbone == "sd3":
        return comfy.latent_formats.SD3().process_in(latent)
    if backbone == "flux2":
        return comfy.latent_formats.Flux2().process_in(latent)
    raise ValueError(f"Unsupported PiD backbone: {backbone!r}")


def _is_comfy_clip_text_encoder(text_encoder: Any) -> bool:
    if text_encoder is None:
        return False
    cls = text_encoder.__class__
    return getattr(cls, "__module__", "") == "comfy.sd" and getattr(cls, "__name__", "") == "CLIP"


def _native_text_encoder_dir(name: str) -> Path:
    if name != "gemma-2-2b-it":
        raise ValueError(f"Unsupported native PiD text encoder: {name!r}")
    return UPSTREAM_ROOT / "checkpoints" / "text_encoders" / name


def _find_native_text_encoder_files(name: str) -> tuple[list[Path], Path]:
    text_encoder_dir = _native_text_encoder_dir(name)
    shard_paths = sorted(text_encoder_dir.rglob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"Nenhum shard safetensors encontrado para o text encoder PiD em {text_encoder_dir}.")

    tokenizer_candidates = sorted(text_encoder_dir.rglob("tokenizer.model"))
    if not tokenizer_candidates:
        raise FileNotFoundError(f"tokenizer.model nao encontrado para o text encoder PiD em {text_encoder_dir}.")
    return shard_paths, tokenizer_candidates[0]


def _ensure_native_text_encoder_assets(name: str) -> Path:
    text_encoder_dir = _native_text_encoder_dir(name)
    text_encoder_dir.mkdir(parents=True, exist_ok=True)

    try:
        _find_native_text_encoder_files(name)
        return text_encoder_dir
    except FileNotFoundError:
        pass

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_PID_TEXT_ENCODER_REPO_ID,
        local_dir=str(text_encoder_dir),
        allow_patterns=[HF_PID_TEXT_ENCODER_FILE],
    )
    snapshot_download(
        repo_id=HF_PID_TEXT_ENCODER_TOKENIZER_REPO_ID,
        local_dir=str(text_encoder_dir),
        allow_patterns=[HF_PID_TEXT_ENCODER_TOKENIZER_FILE],
    )

    _find_native_text_encoder_files(name)
    return text_encoder_dir


def _load_native_text_encoder(name: str) -> Any:
    import comfy.sd
    import comfy.utils
    _ensure_native_text_encoder_assets(name)
    shard_paths, tokenizer_model = _find_native_text_encoder_files(name)

    state_dict = {}
    for shard_path in shard_paths:
        state_dict.update(comfy.utils.load_torch_file(str(shard_path), safe_load=True))

    state_dict["spiece_model"] = tokenizer_model.read_bytes()

    return comfy.sd.load_text_encoder_state_dicts(
        [state_dict],
        clip_type=comfy.sd.CLIPType.PIXELDIT,
        model_options={},
    )


def _validate_pid_prompt_tensors(
    caption_embs: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    prompt_source: str,
) -> None:
    if caption_embs.ndim != 3:
        raise ValueError(f"caption_embs invalido para PiD: {tuple(caption_embs.shape)}.")
    if int(caption_embs.shape[1]) != PID_TEXT_TOKEN_COUNT or int(caption_embs.shape[2]) != PID_TEXT_EMBED_DIM:
        raise ValueError(
            f"{prompt_source} gerou embeddings {tuple(caption_embs.shape)}; o PiD espera [B,{PID_TEXT_TOKEN_COUNT},{PID_TEXT_EMBED_DIM}]. "
            "Use o text encoder Gemma do PiD ou um CLIP carregado com type='pixeldit'."
        )
    if attention_mask is not None and tuple(attention_mask.shape) != tuple(caption_embs.shape[:2]):
        raise ValueError(
            f"attention_mask invalida para PiD: esperado {tuple(caption_embs.shape[:2])}, recebeu {tuple(attention_mask.shape)}."
        )


def _encode_prompt_with_external_clip(clip: Any, prompt: str) -> PiDPrompt:
    if not _is_comfy_clip_text_encoder(clip):
        raise ValueError("O input clip do PiD precisa ser um objeto CLIP do ComfyUI.")

    import comfy.model_management

    comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
    with torch.inference_mode():
        encoded = clip.encode_from_tokens(
            clip.tokenize(prompt),
            return_dict=True,
        )
    caption_embs = encoded["cond"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.int64)
    _validate_pid_prompt_tensors(caption_embs, attention_mask, prompt_source="O CLIP externo")
    return PiDPrompt(
        caption_embs=caption_embs.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        attention_mask=attention_mask.detach().to(device="cpu", dtype=torch.int64).contiguous() if attention_mask is not None else None,
        prompt=prompt,
    )


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


def _prepare_handle(backbone: str, ckpt_type: str) -> PiDHandle:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported PiD backbone: {backbone!r}")
    if ckpt_type not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported PiD variant: {ckpt_type!r}")

    cache_key = (backbone, ckpt_type)
    cached_handle = _HANDLE_CACHE.get(cache_key)
    if cached_handle is not None:
        return cached_handle

    _ensure_upstream_path()
    _ensure_assets(backbone, ckpt_type)

    from pid._src.inference.checkpoint_registry import get_pid_checkpoint

    pid_checkpoint = get_pid_checkpoint(backbone, ckpt_type)
    checkpoint_path = (UPSTREAM_ROOT / pid_checkpoint.checkpoint_path).resolve()
    handle = PiDHandle(
        backbone=backbone,
        ckpt_type=ckpt_type,
        pid_scale=4,
        latent_channels=LATENT_CHANNELS[backbone],
        latent_compression=LATENT_COMPRESSION[backbone],
        input_caption_key="caption",
        checkpoint_path=str(checkpoint_path),
        device="cuda",
    )
    _HANDLE_CACHE[cache_key] = handle
    return handle


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
    text_encoder = getattr(model, "text_encoder", None)
    if _is_comfy_clip_text_encoder(text_encoder):
        try:
            import comfy.model_management

            comfy.model_management.unload_model_and_clones(text_encoder.patcher)
        except Exception:
            pass
    else:
        _set_module_device(text_encoder, "cpu")
    _set_module_device(getattr(model, "vae_encoder", None), "cpu")
    _empty_cuda_cache()


def _get_execution_dtype(model: Any) -> torch.dtype:
    return getattr(model, "autocast_dtype", None) or getattr(model, "precision", torch.float32)


def _set_runtime_net_device(model: Any, device: str, precision: torch.dtype | None = None) -> None:
    net = getattr(model, "net", None)
    if net is None:
        return

    try:
        if precision is not None:
            model.net = net.to(device=device, dtype=precision)
        else:
            model.net = net.to(device=device)
    except TypeError:
        model.net = net.to(device=device)

    _set_module_eval(model.net)
    _disable_module_grads(model.net)
    if device == "cpu":
        _empty_cuda_cache()


def _prepare_model_for_inference(model: Any) -> None:
    if not hasattr(model, "patcher"):
        precision = _get_execution_dtype(model)
        _set_runtime_net_device(model, "cuda", precision=precision)
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


def _to_light_model(model: Any, backbone: str) -> _LightPiDModel:
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("PiD runtime invalido: modelo carregado sem config.")

    light_config = _LightPiDConfig(
        backbone=backbone,
        input_caption_key=str(config.input_caption_key),
        text_encoder_name=str(config.text_encoder_name),
        model_max_length=int(config.model_max_length),
        chi_prompt_str="\n".join(config.chi_prompt) if getattr(config, "chi_prompt", None) else "",
        prediction_type=str(getattr(config, "prediction_type", "velocity")),
        student_timestep=float(getattr(config, "student_timestep", 1.0)),
        student_sample_steps=int(getattr(config, "student_sample_steps", 1)),
        student_sample_type=str(getattr(config, "student_sample_type", "sde")),
        student_t_list=tuple(float(v) for v in config.student_t_list) if getattr(config, "student_t_list", None) else None,
        state_ch=int(getattr(config, "state_ch", 0)),
        tokenizer_config=getattr(config, "tokenizer", None),
    )

    net = getattr(model, "net", None)
    if net is None:
        raise RuntimeError("PiD runtime invalido: modelo carregado sem net.")

    text_encoder = getattr(model, "text_encoder", None)
    vae_encoder = getattr(model, "vae_encoder", None)

    try:
        model.net = None
    except Exception:
        pass
    try:
        model.text_encoder = None
    except Exception:
        pass
    try:
        model.vae_encoder = None
    except Exception:
        pass

    light_model = _LightPiDModel(
        net=net,
        config=light_config,
        precision=getattr(model, "precision", torch.float32),
        autocast_dtype=getattr(model, "autocast_dtype", None),
        fm_timescale=float(getattr(getattr(model, "fm_trainer", None), "timescale", 1000.0)),
        text_encoder=text_encoder,
        vae_encoder=vae_encoder,
    )
    del model
    gc.collect()
    return light_model


def _load_native_model(backbone: str, checkpoint_path: Path) -> _NativePiDModel:
    import comfy.sd

    _patch_native_pid_attention_dtype_mismatch()
    patcher = comfy.sd.load_diffusion_model(str(checkpoint_path), model_options={})
    base_model = patcher.model
    if base_model is None or getattr(base_model, "diffusion_model", None) is None:
        raise RuntimeError(f"Falha ao carregar runtime nativo PiD para {backbone}.")
    return _NativePiDModel(
        patcher=patcher,
        base_model=base_model,
        config=_native_pid_config(backbone),
    )


def _release_active_runtime() -> None:
    global _ACTIVE_RUNTIME

    runtime = _ACTIVE_RUNTIME
    if runtime is None:
        return

    model = runtime.model
    if hasattr(model, "release_native_model"):
        try:
            model.release_native_model()
        except Exception:
            pass
    for module_name in ("net", "text_encoder", "vae_encoder"):
        _set_module_device(getattr(model, module_name, None), "cpu")

    _ACTIVE_RUNTIME = None
    _clear_prompt_cache(runtime.cache_key)
    del model
    gc.collect()
    _empty_cuda_cache()


def _load_runtime(backbone: str, ckpt_type: str) -> tuple[PiDHandle, Any]:
    global _ACTIVE_RUNTIME

    if not torch.cuda.is_available():
        raise RuntimeError("PiD precisa de CUDA para carregar e inferir no ComfyUI.")

    cache_key = (backbone, ckpt_type)
    handle = _prepare_handle(backbone, ckpt_type)
    if _ACTIVE_RUNTIME is not None and _ACTIVE_RUNTIME.cache_key == cache_key:
        return handle, _ACTIVE_RUNTIME.model

    _release_active_runtime()
    checkpoint_path = Path(handle.checkpoint_path)
    _ensure_upstream_path()
    from pid._src.inference.checkpoint_registry import get_pid_checkpoint
    pid_checkpoint = get_pid_checkpoint(backbone, ckpt_type)
    from pid._src.utils.model_loader import load_model_from_checkpoint

    with _pushd(UPSTREAM_ROOT):
        model, _config = load_model_from_checkpoint(
            experiment_name=pid_checkpoint.experiment,
            checkpoint_path=str(checkpoint_path),
            config_file=UPSTREAM_CONFIG,
            enable_fsdp=False,
            experiment_opts=_tokenizer_overrides(backbone),
            strict=False,
        )
    model = _to_light_model(model, backbone)
    _prepare_model_for_inference(model)
    _ACTIVE_RUNTIME = _PiDLoadedRuntime(cache_key=cache_key, model=model)
    return handle, model


def load_pid_model(backbone: str, ckpt_type: str) -> PiDHandle:
    return _prepare_handle(backbone, ckpt_type)


def _get_model(handle: PiDHandle) -> Any:
    _handle, model = _load_runtime(handle.backbone, handle.ckpt_type)
    return model


def _get_encode_model(handle: PiDHandle) -> Any:
    cache_key = (handle.backbone, handle.ckpt_type)
    if _ACTIVE_RUNTIME is not None and _ACTIVE_RUNTIME.cache_key == cache_key:
        return _ACTIVE_RUNTIME.model

    model = _VAE_ONLY_CACHE.get(cache_key)
    if model is not None:
        return model

    _prepare_handle(handle.backbone, handle.ckpt_type)
    _ensure_upstream_path()
    config = _native_pid_config(handle.backbone)
    vae_encoder = _instantiate_vae_encoder(handle.backbone, config.tokenizer_config)
    _set_module_eval(vae_encoder)
    _disable_module_grads(vae_encoder)
    model = _LightPiDModel(
        net=None,
        config=config,
        precision=torch.float32,
        autocast_dtype=None,
        fm_timescale=1000.0,
        text_encoder=None,
        vae_encoder=vae_encoder,
    )
    _VAE_ONLY_CACHE[cache_key] = model
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


def _resolve_pid_prompt(handle: PiDHandle, prompt: str, pid_prompt: Any = None, clip: Any = None) -> PiDPrompt:
    prompt_text = str(prompt or "")
    if pid_prompt is None:
        return _encode_prompt_once(handle, prompt_text, clip=clip)

    normalized = _normalize_pid_prompt(pid_prompt)
    if prompt_text.strip() and prompt_text != normalized.prompt:
        return _encode_prompt_once(handle, prompt_text, clip=clip)
    return normalized


def _encode_prompt_once(handle: PiDHandle, prompt: str, clip: Any = None) -> PiDPrompt:
    cache_key = (handle.backbone, handle.ckpt_type, prompt) if clip is None else ("external", str(id(clip)), prompt)
    cached = _PROMPT_CACHE.get(cache_key)
    if cached is not None:
        _PROMPT_CACHE.move_to_end(cache_key)
        return cached

    if clip is not None:
        prompt_value = _encode_prompt_with_external_clip(clip, prompt)
        _PROMPT_CACHE[cache_key] = prompt_value
        _PROMPT_CACHE.move_to_end(cache_key)
        while len(_PROMPT_CACHE) > MAX_PROMPT_CACHE_ITEMS:
            _PROMPT_CACHE.popitem(last=False)
        return prompt_value

    model = _get_model(handle)
    if hasattr(model, "_ensure_text_encoder_loaded"):
        model._ensure_text_encoder_loaded()
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError("O modelo PiD carregado nao possui text_encoder.")
    if not _is_comfy_clip_text_encoder(text_encoder):
        _set_module_device(text_encoder, handle.device)
    with torch.inference_mode():
        caption_embs, attention_mask = model._encode_text_raw([prompt])
    _validate_pid_prompt_tensors(caption_embs, attention_mask, prompt_source="O text encoder do PiD")
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


def encode_prompt(handle: PiDHandle, prompt: str, clip: Any = None) -> dict[str, torch.Tensor | str]:
    encoded = _encode_prompt_once(handle, prompt, clip=clip)
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


def _nearest_multiple(value: int, alignment: int) -> int:
    if alignment <= 1:
        return max(1, int(value))

    lower = max(alignment, (int(value) // alignment) * alignment)
    upper = max(alignment, ((int(value) + alignment - 1) // alignment) * alignment)
    if abs(int(value) - lower) <= abs(upper - int(value)):
        return lower
    return upper


def _pick_aligned_image_size(height: int, width: int, alignment: int) -> tuple[int, int]:
    if alignment <= 1:
        return max(1, int(height)), max(1, int(width))

    return _nearest_multiple(height, alignment), _nearest_multiple(width, alignment)


def _center_crop_or_pad_image_tensor(
    image_tensor: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    height = int(image_tensor.shape[1])
    width = int(image_tensor.shape[2])
    if (target_height, target_width) == (height, width):
        return image_tensor

    crop_top = max(0, (height - target_height) // 2)
    crop_left = max(0, (width - target_width) // 2)
    crop_bottom = crop_top + min(height, target_height)
    crop_right = crop_left + min(width, target_width)
    cropped = image_tensor[:, crop_top:crop_bottom, crop_left:crop_right, :].contiguous()

    pad_h = max(0, target_height - int(cropped.shape[1]))
    pad_w = max(0, target_width - int(cropped.shape[2]))
    if pad_h == 0 and pad_w == 0:
        return cropped

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    chw_image = cropped.permute(0, 3, 1, 2).contiguous()
    padded = F.pad(chw_image, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return padded.permute(0, 2, 3, 1).contiguous()


def _autocorrect_encode_image_tensor(handle: PiDHandle, image_tensor: torch.Tensor) -> torch.Tensor:
    height = int(image_tensor.shape[1])
    width = int(image_tensor.shape[2])
    alignment = max(1, int(handle.latent_compression * handle.pid_scale))
    target_height, target_width = _pick_aligned_image_size(height, width, alignment)

    if (target_height, target_width) == (height, width):
        return image_tensor

    return _center_crop_or_pad_image_tensor(image_tensor, target_height, target_width).clamp(0.0, 1.0)


def _get_preview_size() -> int:
    try:
        from comfy.cli_args import args

        return int(getattr(args, "preview_size", 512))
    except Exception:
        return 512


def _get_preview_dimensions(height: int, width: int) -> tuple[int, int]:
    preview_size = _get_preview_size()
    scale = min(preview_size / max(height, 1), preview_size / max(width, 1), 1.0) if preview_size > 0 else 1.0
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _make_preview_tuple(image: torch.Tensor):
    preview_tensor = image[:1].float().clamp(0, 1)
    target_height, target_width = _get_preview_dimensions(int(preview_tensor.shape[-2]), int(preview_tensor.shape[-1]))
    if (target_height, target_width) != tuple(preview_tensor.shape[-2:]):
            preview_tensor = F.interpolate(
                preview_tensor,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

    arr = (preview_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    return ("JPEG", Image.fromarray(arr), _get_preview_size())


def _make_tiled_composed_preview(output: torch.Tensor, weight_sum: torch.Tensor):
    composed = output[:1].permute(0, 3, 1, 2).float()
    weights = weight_sum[:1].permute(0, 3, 1, 2).float().clamp_min(1e-6)
    target_height, target_width = _get_preview_dimensions(int(composed.shape[-2]), int(composed.shape[-1]))
    if (target_height, target_width) != tuple(composed.shape[-2:]):
            composed = F.interpolate(
                composed,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            weights = F.interpolate(
                weights,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).clamp_min(1e-6)

    return _make_preview_tuple((composed / weights).clamp(0, 1))


def _write_tile_to_preview_canvas(
    canvas: torch.Tensor,
    tile_image: torch.Tensor,
    out_y: int,
    out_x: int,
    output_h: int,
    output_w: int,
) -> None:
    preview_h = int(canvas.shape[1])
    preview_w = int(canvas.shape[2])
    tile_h = int(tile_image.shape[1])
    tile_w = int(tile_image.shape[2])
    start_y = min(preview_h - 1, max(0, int(round(out_y * preview_h / max(output_h, 1)))))
    start_x = min(preview_w - 1, max(0, int(round(out_x * preview_w / max(output_w, 1)))))
    end_y = min(preview_h, max(start_y + 1, int(round((out_y + tile_h) * preview_h / max(output_h, 1)))))
    end_x = min(preview_w, max(start_x + 1, int(round((out_x + tile_w) * preview_w / max(output_w, 1)))))

    resized = F.interpolate(
        tile_image.permute(0, 3, 1, 2).float(),
        size=(end_y - start_y, end_x - start_x),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1).contiguous()
    canvas[:, start_y:end_y, start_x:end_x, :] = resized


def encode_image_to_latent(
    handle: PiDHandle,
    image: Any,
    encode_tile_size: int | None = None,
) -> dict[str, torch.Tensor]:
    model = _get_encode_model(handle)
    if hasattr(model, "_ensure_vae_encoder_loaded"):
        model._ensure_vae_encoder_loaded()
    _set_runtime_net_device(model, "cpu")
    text_encoder = getattr(model, "text_encoder", None)
    if _is_comfy_clip_text_encoder(text_encoder):
        try:
            import comfy.model_management

            comfy.model_management.unload_model_and_clones(text_encoder.patcher)
        except Exception:
            pass
    else:
        _set_module_device(text_encoder, "cpu")
    _empty_cuda_cache()

    image_tensor = _autocorrect_encode_image_tensor(handle, _extract_image_tensor(image))
    vae_encoder = getattr(model, "vae_encoder", None)
    vae_dtype = getattr(vae_encoder, "dtype", torch.float32)
    if vae_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        vae_dtype = torch.float32

    chw_image = image_tensor.movedim(-1, 1)
    _set_module_device(vae_encoder, handle.device)
    height = int(chw_image.shape[-2])
    width = int(chw_image.shape[-1])
    if encode_tile_size is None:
        vae_input = chw_image.to(device=handle.device, dtype=vae_dtype, non_blocking=True)
        vae_input.mul_(2.0).add_(-1.0).clamp_(-1.0, 1.0)
        with torch.inference_mode():
            latent = model.encode_lq_latent(vae_input)
        del vae_input
    else:
        tile_size = max(handle.latent_compression, int(encode_tile_size))
        tile_size = max(handle.latent_compression, (tile_size // handle.latent_compression) * handle.latent_compression)
        overlap = int(AUTOENCODE_TILE_OVERLAP.get(handle.backbone, 128))
        overlap = max(0, min(tile_size - handle.latent_compression, (overlap // handle.latent_compression) * handle.latent_compression))
        latent = _encode_image_to_latent_tiled(
            handle=handle,
            model=model,
            chw_image=chw_image,
            vae_dtype=vae_dtype,
            tile_size=tile_size,
            tile_overlap=overlap,
        )
    _offload_aux_modules(model)

    if latent.ndim != 4:
        raise ValueError(f"Encoder do PiD retornou latent invalido: {tuple(latent.shape)}.")
    if latent.shape[1] != handle.latent_channels:
        raise ValueError(
            f"Encoder do PiD para '{handle.backbone}' retornou {latent.shape[1]} canais, "
            f"mas o modelo espera {handle.latent_channels}."
        )

    return {"samples": latent.float().cpu()}


def _latent_tile_weight_mask(
    height: int,
    width: int,
    overlap: int,
    top_edge: bool,
    bottom_edge: bool,
    left_edge: bool,
    right_edge: bool,
    device: torch.device,
) -> torch.Tensor:
    return _tile_weight_mask(
        height=height,
        width=width,
        overlap=overlap,
        top_edge=top_edge,
        bottom_edge=bottom_edge,
        left_edge=left_edge,
        right_edge=right_edge,
        device=device,
    ).permute(0, 3, 1, 2).contiguous()


def _encode_image_to_latent_tiled(
    handle: PiDHandle,
    model: Any,
    chw_image: torch.Tensor,
    vae_dtype: torch.dtype,
    tile_size: int,
    tile_overlap: int,
) -> torch.Tensor:
    batch_size = int(chw_image.shape[0])
    height = int(chw_image.shape[-2])
    width = int(chw_image.shape[-1])
    compression = int(handle.latent_compression)
    latent_h = height // compression
    latent_w = width // compression
    overlap_latent = tile_overlap // compression
    starts_y = _compute_tile_starts(height, tile_size, tile_overlap)
    starts_x = _compute_tile_starts(width, tile_size, tile_overlap)

    output = torch.zeros((batch_size, handle.latent_channels, latent_h, latent_w), dtype=torch.float32, device="cpu")
    weight_sum = torch.zeros((1, 1, latent_h, latent_w), dtype=torch.float32, device="cpu")
    weight_cache: dict[tuple[int, int, int, bool, bool, bool, bool], torch.Tensor] = {}

    for start_y in starts_y:
        for start_x in starts_x:
            end_y = min(start_y + tile_size, height)
            end_x = min(start_x + tile_size, width)
            pixel_tile = chw_image[:, :, start_y:end_y, start_x:end_x]
            vae_input = pixel_tile.to(device=handle.device, dtype=vae_dtype, non_blocking=True)
            vae_input.mul_(2.0).add_(-1.0).clamp_(-1.0, 1.0)

            with torch.inference_mode():
                latent_tile = model.encode_lq_latent(vae_input).float().cpu()
            del vae_input

            latent_start_y = start_y // compression
            latent_start_x = start_x // compression
            tile_latent_h = int(latent_tile.shape[-2])
            tile_latent_w = int(latent_tile.shape[-1])
            weight_key = (
                tile_latent_h,
                tile_latent_w,
                min(overlap_latent, tile_latent_h // 2, tile_latent_w // 2),
                start_y == 0,
                end_y == height,
                start_x == 0,
                end_x == width,
            )
            weight = weight_cache.get(weight_key)
            if weight is None:
                weight = _latent_tile_weight_mask(
                    height=tile_latent_h,
                    width=tile_latent_w,
                    overlap=weight_key[2],
                    top_edge=weight_key[3],
                    bottom_edge=weight_key[4],
                    left_edge=weight_key[5],
                    right_edge=weight_key[6],
                    device=torch.device("cpu"),
                )
                weight_cache[weight_key] = weight

            output[
                :,
                :,
                latent_start_y : latent_start_y + tile_latent_h,
                latent_start_x : latent_start_x + tile_latent_w,
            ] += latent_tile * weight
            weight_sum[
                :,
                :,
                latent_start_y : latent_start_y + tile_latent_h,
                latent_start_x : latent_start_x + tile_latent_w,
            ] += weight
            del latent_tile
            _empty_cuda_cache()

    return output / weight_sum.clamp_min(1e-6)


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


def _expand_tile_axis(start: int, end: int, total: int, min_size: int) -> tuple[int, int]:
    current = end - start
    target = min(total, max(current, min_size))
    extra = target - current
    if extra <= 0:
        return start, end

    before = min(start, extra // 2)
    after = min(total - end, extra - before)
    missing = extra - before - after
    if missing > 0:
        grow_before = min(start - before, missing)
        before += grow_before
        missing -= grow_before
    if missing > 0:
        grow_after = min(total - end - after, missing)
        after += grow_after

    return start - before, end + after


def _expand_tile_job(
    job: _TileDecodeJob,
    total_h: int,
    total_w: int,
    compression: int,
    pid_scale: int,
) -> _ExpandedTileDecodeJob:
    min_decode_latent = max(1, MIN_TILED_DECODE_SIZE // compression)
    decode_start_y, decode_end_y = _expand_tile_axis(job.start_y, job.end_y, total_h, min_decode_latent)
    decode_start_x, decode_end_x = _expand_tile_axis(job.start_x, job.end_x, total_w, min_decode_latent)

    crop_y = (job.start_y - decode_start_y) * compression * pid_scale
    crop_x = (job.start_x - decode_start_x) * compression * pid_scale

    return _ExpandedTileDecodeJob(
        target_start_y=job.start_y,
        target_start_x=job.start_x,
        target_end_y=job.end_y,
        target_end_x=job.end_x,
        decode_start_y=decode_start_y,
        decode_start_x=decode_start_x,
        decode_end_y=decode_end_y,
        decode_end_x=decode_end_x,
        out_y=job.out_y,
        out_x=job.out_x,
        crop_y=crop_y,
        crop_x=crop_x,
    )


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
    step_preview_callback: Any = None,
) -> torch.Tensor:
    if latent_tensor.shape[1] != handle.latent_channels:
        raise ValueError(
            f"Backbone '{handle.backbone}' espera {handle.latent_channels} canais no latent, "
            f"mas recebeu {latent_tensor.shape[1]}."
        )

    model = _get_model(handle)
    if hasattr(model, "ensure_model_loaded"):
        model.ensure_model_loaded()
    device = handle.device
    baseline_h = int(latent_tensor.shape[-2]) * handle.latent_compression
    baseline_w = int(latent_tensor.shape[-1]) * handle.latent_compression
    output_h = baseline_h * handle.pid_scale
    output_w = baseline_w * handle.pid_scale
    batch_size = int(latent_tensor.shape[0])

    precision = _get_execution_dtype(model)
    _set_runtime_net_device(model, device, precision=precision)
    caption_embs = pid_prompt.caption_embs
    attention_mask = pid_prompt.attention_mask
    _validate_pid_prompt_tensors(caption_embs, attention_mask, prompt_source="O condicionamento PiD")
    if caption_embs.shape[0] == 1 and batch_size > 1:
        caption_embs = caption_embs.expand(batch_size, -1, -1)
    elif caption_embs.shape[0] != batch_size:
        raise ValueError(
            f"caption_embs precisa ter batch 1 ou {batch_size}, mas recebeu {caption_embs.shape[0]}."
        )

    caption_embs = caption_embs.to(device=device, dtype=precision)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device, dtype=torch.int64)
    latent_state = latent_tensor.to(device=device, dtype=precision)
    if hasattr(model, "base_model"):
        latent_state = _native_lq_latent_process_in(handle.backbone, latent_state)
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
        return _make_preview_tuple(image)

    with torch.inference_mode():
        effective_steps = int(pid_inference_steps) if pid_inference_steps is not None else int(model.config.student_sample_steps)
        student_sample_type = getattr(model.config, "student_sample_type", "sde")
        prediction_type = getattr(model.config, "prediction_type", "velocity")
        student_timestep = float(getattr(model.config, "student_timestep", 1.0))
        if effective_steps == 1:
            t_student = torch.full((batch_size,), student_timestep, device=device, dtype=torch.float32)
            if hasattr(model, "predict_x0"):
                x0_student = model.predict_x0(
                    noise,
                    t_student,
                    caption_embs,
                    attention_mask,
                    latent_state,
                    sigma_tensor,
                )
            else:
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
                preview_tuple = _preview_tuple_from_x0(x0_student)
                if step_preview_callback is not None and preview_tuple is not None:
                    step_preview_callback((x0_student.float().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1))
                else:
                    progress.update(
                        advance=progress_advance,
                        preview=preview_tuple,
                        emit_bar=preview_enabled,
                    )
        else:
            t_list = model._get_t_list(device=torch.device(device), num_steps=effective_steps)
            x = noise
            timescale = model.fm_trainer.timescale
            with autocast_ctx:
                for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
                    t_cur_batch = t_cur.expand(batch_size)
                    if hasattr(model, "predict_x0"):
                        x0_pred = model.predict_x0(
                            x,
                            t_cur_batch,
                            caption_embs,
                            attention_mask,
                            latent_state,
                            sigma_tensor,
                        )
                        t_shape = [batch_size] + [1] * (x.ndim - 1)
                        v_pred = ((x.double() - x0_pred.double()) / t_cur_batch.double().view(*t_shape).clamp(min=5e-2)).to(x.dtype)
                    else:
                        t_cur_scaled = t_cur_batch * timescale
                        v_pred = net(
                            x,
                            t_cur_scaled,
                            caption_embs,
                            lq_video_or_image=None,
                            lq_latent=latent_state,
                            degrade_sigma=sigma_tensor,
                        )
                        x0_pred = model._velocity_to_x0(x, v_pred, t_cur_batch)

                    if t_next.item() > 0:
                        if student_sample_type == "ode":
                            v_for_step = model._net_output_to_velocity(x, v_pred, t_cur_batch, prediction_type)
                            dt = t_next - t_cur
                            x = x + dt * v_for_step
                            preview_x0 = x0_pred
                        else:
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
                        x = x0_pred
                        preview_x0 = x

                    if progress is not None:
                        preview_tuple = _preview_tuple_from_x0(preview_x0)
                        if step_preview_callback is not None and preview_tuple is not None:
                            step_preview_callback((preview_x0.float().clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1))
                        else:
                            progress.update(
                                advance=progress_advance,
                                preview=preview_tuple,
                                emit_bar=preview_enabled,
                            )
            x0_student = x
    return x0_student.clamp(-1, 1).unsqueeze(2)


def _samples_to_comfy_image(samples: torch.Tensor) -> torch.Tensor:
    image = samples.squeeze(2).float().clamp(-1, 1)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return image.permute(0, 2, 3, 1).cpu()


def _samples_to_image_tensor(samples: torch.Tensor) -> torch.Tensor:
    image = samples.squeeze(2).float().clamp(-1, 1)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return image.permute(0, 2, 3, 1).contiguous()


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


def _group_tile_jobs_by_decode_shape(tile_jobs: list[_ExpandedTileDecodeJob]) -> list[list[_ExpandedTileDecodeJob]]:
    grouped: dict[tuple[int, int], list[_ExpandedTileDecodeJob]] = {}
    order: list[tuple[int, int]] = []
    for job in tile_jobs:
        key = (job.decode_end_y - job.decode_start_y, job.decode_end_x - job.decode_start_x)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(job)
    return [grouped[key] for key in order]


def decode_latent(
    handle: PiDHandle,
    latent: Any,
    prompt: str,
    cfg_scale: float,
    pid_inference_steps: int,
    seed: int,
    degrade_sigma: float,
    pid_prompt: Any = None,
    clip: Any = None,
    unique_id: str | None = None,
) -> torch.Tensor:
    prompt_value = _resolve_pid_prompt(handle, prompt, pid_prompt=pid_prompt, clip=clip)
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
    clip: Any = None,
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
    expanded_jobs = [
        _expand_tile_job(
            job=job,
            total_h=int(latent_tensor.shape[-2]),
            total_w=int(latent_tensor.shape[-1]),
            compression=compression,
            pid_scale=handle.pid_scale,
        )
        for job in tile_jobs
    ]

    baseline_h = int(latent_tensor.shape[-2]) * compression
    baseline_w = int(latent_tensor.shape[-1]) * compression
    output_h = baseline_h * handle.pid_scale
    output_w = baseline_w * handle.pid_scale
    overlap_out = tile_overlap * handle.pid_scale

    batch_size = int(latent_tensor.shape[0])
    compose_device = torch.device("cpu")
    output = torch.zeros((batch_size, output_h, output_w, 3), dtype=torch.float32, device=compose_device)
    weight_sum = torch.zeros((batch_size, output_h, output_w, 1), dtype=torch.float32, device=compose_device)

    prompt_value = _resolve_pid_prompt(handle, prompt, pid_prompt=pid_prompt, clip=clip)
    tile_count = len(expanded_jobs)
    effective_steps = int(pid_inference_steps) if pid_inference_steps is not None else 4
    progress = _DecodeProgress(total=max(1, tile_count * max(1, effective_steps)), node_id=unique_id)
    model = _get_model(handle)
    precision = _get_execution_dtype(model)
    full_noise = _make_decode_noise(batch_size, output_h, output_w, "cpu", seed, dtype=torch.float32)
    preview_h, preview_w = _get_preview_dimensions(output_h, output_w)
    preview_canvas = torch.zeros((batch_size, preview_h, preview_w, 3), dtype=torch.float32, device=compose_device)

    weight_cache: dict[tuple[int, int, int, bool, bool, bool, bool], torch.Tensor] = {}
    grouped_jobs = _group_tile_jobs_by_decode_shape(expanded_jobs)

    for job_group in grouped_jobs:
        group_index = 0
        while group_index < len(job_group):
            job_batch = job_group[group_index : group_index + tile_batch_size]
            group_index += len(job_batch)

            latent_batch = torch.cat(
                [
                    latent_tensor[:, :, job.decode_start_y : job.decode_end_y, job.decode_start_x : job.decode_end_x].contiguous()
                    for job in job_batch
                ],
                dim=0,
            )
            noise_batch = torch.cat(
                [
                    full_noise[
                        :,
                        :,
                        job.decode_start_y * compression * handle.pid_scale : job.decode_end_y * compression * handle.pid_scale,
                        job.decode_start_x * compression * handle.pid_scale : job.decode_end_x * compression * handle.pid_scale,
                    ].contiguous()
                    for job in job_batch
                ],
                dim=0,
            )

            def _step_preview_callback(step_images: torch.Tensor) -> None:
                preview_frame = preview_canvas.clone()
                step_images_cpu = step_images.to(device=compose_device)
                if step_images_cpu.ndim != 4:
                    return
                if int(step_images_cpu.shape[-1]) != 3 and int(step_images_cpu.shape[1]) == 3:
                    step_images_cpu = step_images_cpu.permute(0, 2, 3, 1).contiguous()
                for offset, job in enumerate(job_batch):
                    tile_image = step_images_cpu[offset * batch_size : (offset + 1) * batch_size]
                    target_h = (job.target_end_y - job.target_start_y) * compression * handle.pid_scale
                    target_w = (job.target_end_x - job.target_start_x) * compression * handle.pid_scale
                    tile_image = tile_image[:, job.crop_y : job.crop_y + target_h, job.crop_x : job.crop_x + target_w, :]
                    _write_tile_to_preview_canvas(
                        preview_frame,
                        tile_image,
                        job.out_y,
                        job.out_x,
                        output_h,
                        output_w,
                    )
                progress.update(advance=len(job_batch), preview=_make_preview_tuple(preview_frame.permute(0, 3, 1, 2)), emit_bar=True)

            tile_samples = _decode_samples(
                handle=handle,
                latent_tensor=latent_batch,
                pid_prompt=_repeat_pid_prompt(prompt_value, len(job_batch), batch_size),
                cfg_scale=cfg_scale,
                pid_inference_steps=pid_inference_steps,
                seed=seed,
                degrade_sigma=degrade_sigma,
                progress=progress,
                preview_enabled=True,
                noise=noise_batch,
                progress_advance=len(job_batch),
                step_preview_callback=_step_preview_callback,
            )
            tile_images = _samples_to_image_tensor(tile_samples).to(device=compose_device)

            for offset, job in enumerate(job_batch):
                tile_image = tile_images[offset * batch_size : (offset + 1) * batch_size]
                target_h = (job.target_end_y - job.target_start_y) * compression * handle.pid_scale
                target_w = (job.target_end_x - job.target_start_x) * compression * handle.pid_scale
                tile_image = tile_image[:, job.crop_y : job.crop_y + target_h, job.crop_x : job.crop_x + target_w, :]
                tile_h = int(tile_image.shape[1])
                tile_w = int(tile_image.shape[2])
                weight_key = (
                    tile_h,
                    tile_w,
                    min(overlap_out, tile_h // 2, tile_w // 2),
                    job.target_start_y == 0,
                    job.target_end_y == int(latent_tensor.shape[-2]),
                    job.target_start_x == 0,
                    job.target_end_x == int(latent_tensor.shape[-1]),
                )
                weight = weight_cache.get(weight_key)
                if weight is None:
                    weight = _tile_weight_mask(
                        height=tile_h,
                        width=tile_w,
                        overlap=weight_key[2],
                        top_edge=weight_key[3],
                        bottom_edge=weight_key[4],
                        left_edge=weight_key[5],
                        right_edge=weight_key[6],
                        device=compose_device,
                    )
                    weight_cache[weight_key] = weight
                output[:, job.out_y : job.out_y + tile_h, job.out_x : job.out_x + tile_w, :] += tile_image * weight
                weight_sum[:, job.out_y : job.out_y + tile_h, job.out_x : job.out_x + tile_w, :] += weight
                _write_tile_to_preview_canvas(
                    preview_canvas,
                    tile_image,
                    job.out_y,
                    job.out_x,
                    output_h,
                    output_w,
                )

    final_preview = None
    try:
        final_preview = _make_tiled_composed_preview(output, weight_sum)
    except Exception:
        final_preview = None
    if final_preview is not None:
        progress.update(advance=0, preview=final_preview, emit_bar=True)

    return (output / weight_sum.clamp_min(1e-6)).cpu()


def pid_ksampler(
    handle: PiDHandle,
    latent: Any,
    prompt: str,
    pid_inference_steps: int,
    seed: int,
    degrade_sigma: float,
    keep_model_loaded_on_gpu: bool = True,
    use_tiled: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 64,
    tile_batch_size: int = 1,
    pid_prompt: Any = None,
    clip: Any = None,
    unique_id: str | None = None,
) -> torch.Tensor:
    try:
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
                clip=clip,
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
            clip=clip,
            unique_id=unique_id,
        )
    finally:
        if not keep_model_loaded_on_gpu:
            model = _get_model(handle)
            _set_runtime_net_device(model, "cpu")
