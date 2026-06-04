from __future__ import annotations

from .pid_runtime import (
    SUPPORTED_BACKBONES,
    SUPPORTED_VARIANTS,
    decode_latent,
    decode_latent_tiled,
    encode_image_to_latent,
    encode_prompt,
    load_pid_model,
    normalize_checkpoint_variant,
    pid_ksampler,
    match_colors,
    resolve_reference_image,
)

COLOR_MATCH_OPTIONS = ["disabled", "reinhard_rgb", "wavelet"]


def _format_image_resolution(image) -> str:
    height = int(image.shape[1])
    width = int(image.shape[2])
    return f"{height} x {width}"


def _with_resolution_ui(image):
    return {"ui": {"text": (_format_image_resolution(image),)}, "result": (image,)}


def _apply_color_match(image, latent, image_ref, color_match: str):
    reference = resolve_reference_image(latent, image_ref)
    if color_match != "disabled" and reference is not None:
        return match_colors(image, reference, method=color_match)
    return image


class PiDLoadModel:
    CATEGORY = "PiD"
    FUNCTION = "load_model"
    RETURN_TYPES = ("PID_MODEL",)
    RETURN_NAMES = ("pid_model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backbone": (list(SUPPORTED_BACKBONES),),
                "checkpoint_variant": (["auto", *list(SUPPORTED_VARIANTS)], {"default": "auto"}),
            }
        }

    def load_model(self, backbone: str, checkpoint_variant: str):
        return (load_pid_model(backbone, normalize_checkpoint_variant(backbone, checkpoint_variant)),)


class PiDDecodeLatent:
    CATEGORY = "PiD"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("PID_MODEL",),
                "latent": ("LATENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "pid_inference_steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lq_conditioning_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "source_denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "source_detail_noise_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "sde_noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_match": (COLOR_MATCH_OPTIONS, {"default": "disabled"}),
            },
            "optional": {
                "pid_prompt": ("PID_PROMPT",),
                "clip": ("CLIP",),
                "image_ref": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def decode(
        self,
        pid_model,
        latent,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        pid_inference_steps: int,
        seed: int,
        degrade_sigma: float,
        lq_conditioning_boost: float,
        source_denoise_strength: float,
        source_detail_noise_boost: float,
        color_match: str = "disabled",
        scheduler: str = "original",
        sde_noise_strength: float = 1.0,
        pid_prompt=None,
        clip=None,
        unique_id=None,
        image_ref=None,
    ):
        image = decode_latent(
            handle=pid_model,
            latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            lq_conditioning_boost=lq_conditioning_boost,
            source_denoise_strength=source_denoise_strength,
            source_detail_noise_boost=source_detail_noise_boost,
            source_image=image_ref,
            pid_prompt=pid_prompt,
            clip=clip,
            unique_id=unique_id,
            sampler="sde",
            scheduler=scheduler,
            sde_noise_strength=sde_noise_strength,
        )
        image = _apply_color_match(image, latent, image_ref, color_match)
        return _with_resolution_ui(image)


class PiDEncodeImage:
    CATEGORY = "PiD"
    FUNCTION = "encode"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("PID_MODEL",),
                "image": ("IMAGE",),
                "encode_tile_size": (["auto", "disabled", "512", "1024"], {"default": "auto"}),
            }
        }

    def encode(self, pid_model, image, encode_tile_size):
        if encode_tile_size == "auto":
            tile_size = None
        elif encode_tile_size == "disabled":
            tile_size = 0
        else:
            tile_size = int(encode_tile_size)
        return (encode_image_to_latent(pid_model, image, encode_tile_size=tile_size),)


class PiDEncodePrompt:
    CATEGORY = "PiD"
    FUNCTION = "encode"
    RETURN_TYPES = ("PID_PROMPT",)
    RETURN_NAMES = ("pid_prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("PID_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    def encode(self, pid_model, prompt: str, clip=None):
        return (encode_prompt(pid_model, prompt, clip=clip),)


class PiDKSampler:
    CATEGORY = "PiD"
    FUNCTION = "sample"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("PID_MODEL",),
                "latent": ("LATENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "pid_inference_steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lq_conditioning_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "source_denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "source_detail_noise_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "keep_model_loaded_on_gpu": ("BOOLEAN", {"default": True}),
                "use_tiled": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "tile_batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "seam_refine": ("BOOLEAN", {"default": False}),
                "seam_refine_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sde_noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tiled_sde_noise_boost": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 2.0, "step": 0.05}),
                "color_match": (COLOR_MATCH_OPTIONS, {"default": "disabled"}),
            },
            "optional": {
                "pid_prompt": ("PID_PROMPT",),
                "clip": ("CLIP",),
                "image_ref": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def sample(
        self,
        pid_model,
        latent,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        pid_inference_steps: int,
        seed: int,
        degrade_sigma: float,
        lq_conditioning_boost: float,
        source_denoise_strength: float,
        source_detail_noise_boost: float,
        keep_model_loaded_on_gpu: bool,
        use_tiled: bool,
        tile_size: int,
        tile_overlap: int,
        tile_batch_size: int,
        seam_refine: bool = False,
        seam_refine_strength: float = 0.25,
        color_match: str = "disabled",
        scheduler: str = "original",
        sde_noise_strength: float = 1.0,
        tiled_sde_noise_boost: float = 1.15,
        pid_prompt=None,
        clip=None,
        unique_id=None,
        image_ref=None,
    ):
        image = pid_ksampler(
            handle=pid_model,
            latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            lq_conditioning_boost=lq_conditioning_boost,
            source_denoise_strength=source_denoise_strength,
            source_detail_noise_boost=source_detail_noise_boost,
            source_image=image_ref,
            keep_model_loaded_on_gpu=keep_model_loaded_on_gpu,
            use_tiled=use_tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            seam_refine=seam_refine,
            seam_refine_strength=seam_refine_strength,
            pid_prompt=pid_prompt,
            clip=clip,
            unique_id=unique_id,
            sampler="sde",
            scheduler=scheduler,
            sde_noise_strength=sde_noise_strength,
            tiled_sde_noise_boost=tiled_sde_noise_boost,
        )
        image = _apply_color_match(image, latent, image_ref, color_match)
        return _with_resolution_ui(image)


class PiDDecodeLatentTiled:
    CATEGORY = "PiD"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("PID_MODEL",),
                "latent": ("LATENT",),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "tile_batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "seam_refine": ("BOOLEAN", {"default": False}),
                "seam_refine_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "pid_inference_steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lq_conditioning_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "source_denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "source_detail_noise_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "sde_noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tiled_sde_noise_boost": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 2.0, "step": 0.05}),
                "color_match": (COLOR_MATCH_OPTIONS, {"default": "disabled"}),
            },
            "optional": {
                "pid_prompt": ("PID_PROMPT",),
                "clip": ("CLIP",),
                "image_ref": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def decode(
        self,
        pid_model,
        latent,
        tile_size: int,
        tile_overlap: int,
        tile_batch_size: int,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        pid_inference_steps: int,
        seed: int,
        degrade_sigma: float,
        lq_conditioning_boost: float,
        source_denoise_strength: float,
        source_detail_noise_boost: float,
        seam_refine: bool = False,
        seam_refine_strength: float = 0.25,
        color_match: str = "disabled",
        scheduler: str = "original",
        sde_noise_strength: float = 1.0,
        tiled_sde_noise_boost: float = 1.15,
        pid_prompt=None,
        clip=None,
        unique_id=None,
        image_ref=None,
    ):
        image = decode_latent_tiled(
            handle=pid_model,
            latent=latent,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            seam_refine=seam_refine,
            seam_refine_strength=seam_refine_strength,
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            lq_conditioning_boost=lq_conditioning_boost,
            source_denoise_strength=source_denoise_strength,
            source_detail_noise_boost=source_detail_noise_boost,
            source_image=image_ref,
            pid_prompt=pid_prompt,
            clip=clip,
            unique_id=unique_id,
            sampler="sde",
            scheduler=scheduler,
            sde_noise_strength=sde_noise_strength,
            tiled_sde_noise_boost=tiled_sde_noise_boost,
        )
        image = _apply_color_match(image, latent, image_ref, color_match)
        return _with_resolution_ui(image)


NODE_CLASS_MAPPINGS = {
    "PiDDecodeLatentTiled": PiDDecodeLatentTiled,
    "PiDEncodeImage": PiDEncodeImage,
    "PiDEncodePrompt": PiDEncodePrompt,
    "PiDKSampler": PiDKSampler,
    "PiDLoadModel": PiDLoadModel,
    "PiDDecodeLatent": PiDDecodeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDDecodeLatentTiled": "PiD Decode Latent Tiled",
    "PiDEncodeImage": "PiD Encode Image",
    "PiDEncodePrompt": "PiD Encode Prompt",
    "PiDKSampler": "PiD KSampler",
    "PiDLoadModel": "PiD Load Model",
    "PiDDecodeLatent": "PiD Decode Latent",
}
