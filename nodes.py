from __future__ import annotations

from .pid_runtime import (
    SUPPORTED_BACKBONES,
    SUPPORTED_VARIANTS,
    decode_latent,
    decode_latent_tiled,
    encode_image_to_latent,
    encode_prompt,
    load_pid_model,
    pid_ksampler,
)


def _format_image_resolution(image) -> str:
    height = int(image.shape[1])
    width = int(image.shape[2])
    return f"{height} x {width}"


def _with_resolution_ui(image):
    return {"ui": {"text": (_format_image_resolution(image),)}, "result": (image,)}


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
                "checkpoint_variant": (list(SUPPORTED_VARIANTS),),
            }
        }

    def load_model(self, backbone: str, checkpoint_variant: str):
        return (load_pid_model(backbone, checkpoint_variant),)


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
                "pid_inference_steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "pid_prompt": ("PID_PROMPT",),
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
        pid_inference_steps: int,
        seed: int,
        degrade_sigma: float,
        pid_prompt=None,
        unique_id=None,
    ):
        image = decode_latent(
            handle=pid_model,
            latent=latent,
            prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            pid_prompt=pid_prompt,
            unique_id=unique_id,
        )
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
            }
        }

    def encode(self, pid_model, image):
        return (encode_image_to_latent(pid_model, image),)


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
            }
        }

    def encode(self, pid_model, prompt: str):
        return (encode_prompt(pid_model, prompt),)


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
                "pid_inference_steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "keep_model_loaded_on_gpu": ("BOOLEAN", {"default": True}),
                "use_tiled": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "tile_batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "pid_prompt": ("PID_PROMPT",),
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
        pid_inference_steps: int,
        seed: int,
        degrade_sigma: float,
        keep_model_loaded_on_gpu: bool,
        use_tiled: bool,
        tile_size: int,
        tile_overlap: int,
        tile_batch_size: int,
        pid_prompt=None,
        unique_id=None,
    ):
        image = pid_ksampler(
            handle=pid_model,
            latent=latent,
            prompt=prompt,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            keep_model_loaded_on_gpu=keep_model_loaded_on_gpu,
            use_tiled=use_tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            pid_prompt=pid_prompt,
            unique_id=unique_id,
        )
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
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "tile_batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "pid_inference_steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "pid_prompt": ("PID_PROMPT",),
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
        pid_inference_steps: int,
        seed: int,
        degrade_sigma: float,
        pid_prompt=None,
        unique_id=None,
    ):
        image = decode_latent_tiled(
            handle=pid_model,
            latent=latent,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=pid_inference_steps,
            seed=seed,
            degrade_sigma=degrade_sigma,
            pid_prompt=pid_prompt,
            unique_id=unique_id,
        )
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
