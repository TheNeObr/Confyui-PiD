# Minimal official PiD checkpoint registry used by this ComfyUI node.

from dataclasses import dataclass


@dataclass(frozen=True)
class PIDCheckpoint:
    experiment: str
    checkpoint_path: str
    pid_scale: int


_CKPT_ROOT = "checkpoints"

VALID_CKPT_TYPES = ("2k", "2kto4k")


PID_CHECKPOINT_REGISTRY: dict[tuple[str, str], PIDCheckpoint] = {
    ("flux", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux2", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("sd3", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_sd3_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("zimage", "2k"): PIDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("flux2", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux2_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("sd3", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_sd3_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("sdxl", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_sdxl_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
    ("qwenimage", "2kto4k"): PIDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_qwenimage_distill_4step",
        checkpoint_path=f"{_CKPT_ROOT}/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
        pid_scale=4,
    ),
}

PID_CHECKPOINT_REGISTRY[("zimage-turbo", "2k")] = PID_CHECKPOINT_REGISTRY[("flux", "2k")]
PID_CHECKPOINT_REGISTRY[("zimage", "2kto4k")] = PID_CHECKPOINT_REGISTRY[("flux", "2kto4k")]
PID_CHECKPOINT_REGISTRY[("zimage-turbo", "2kto4k")] = PID_CHECKPOINT_REGISTRY[("flux", "2kto4k")]
PID_CHECKPOINT_REGISTRY[("qwenimage-2512", "2kto4k")] = PID_CHECKPOINT_REGISTRY[("qwenimage", "2kto4k")]
for _klein in ("flux2-klein-4b", "flux2-klein-9b"):
    PID_CHECKPOINT_REGISTRY[(_klein, "2k")] = PID_CHECKPOINT_REGISTRY[("flux2", "2k")]
    PID_CHECKPOINT_REGISTRY[(_klein, "2kto4k")] = PID_CHECKPOINT_REGISTRY[("flux2", "2kto4k")]


def get_pid_checkpoint(backbone: str, ckpt_type: str = "2k") -> PIDCheckpoint:
    """Return the registered official PID checkpoint for `(backbone, ckpt_type)`.

    `ckpt_type` defaults to `"2k"` so existing call sites keep their pre-2kto4k
    behavior. Raises KeyError with the list of valid keys when the pair is
    unknown.
    """
    if ckpt_type not in VALID_CKPT_TYPES:
        raise KeyError(f"Unknown ckpt_type {ckpt_type!r}. Valid: {VALID_CKPT_TYPES}")
    try:
        return PID_CHECKPOINT_REGISTRY[(backbone, ckpt_type)]
    except KeyError as exc:
        valid = ", ".join(sorted(f"{b}+{t}" for b, t in PID_CHECKPOINT_REGISTRY))
        raise KeyError(f"Unknown (backbone, ckpt_type)=({backbone!r}, {ckpt_type!r}). Valid: {valid}") from exc


__all__ = [
    "PIDCheckpoint",
    "PID_CHECKPOINT_REGISTRY",
    "VALID_CKPT_TYPES",
    "get_pid_checkpoint",
]
