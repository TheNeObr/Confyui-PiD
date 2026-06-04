import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_package_module(module_name: str, file_name: str):
    package_name = "pidnode_custom"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        ROOT / file_name,
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pid_runtime = _load_package_module("pid_runtime", "pid_runtime.py")
nodes = _load_package_module("nodes", "nodes.py")


class DummyModel:
    def __init__(self, encode_latent_channels=16, encode_compression=8):
        self.config = types.SimpleNamespace(input_caption_key="caption")
        self.last_caption_embs = None
        self.last_attention_mask = None
        self.last_lq_latent = None
        self.last_degrade_sigma = None
        self.call_count = 0
        self.precision = torch.float32
        self.autocast_dtype = None
        self.fm_trainer = types.SimpleNamespace(timescale=1000.0)
        self.net = mock.Mock(side_effect=lambda x, *args, **kwargs: torch.zeros_like(x))
        self.text_encoder = mock.Mock()
        self.vae_encoder = mock.Mock()
        self.encode_latent_channels = int(encode_latent_channels)
        self.encode_compression = int(encode_compression)

    def eval(self):
        return self

    def _encode_text_raw(self, captions):
        batch = len(captions)
        return (
            torch.full(
                (batch, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM),
                2.0,
                dtype=torch.float32,
            ),
            torch.ones((batch, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
        )

    def _get_t_list(self, device, num_steps=None, scheduler=None):
        steps = num_steps or 4
        return torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

    def predict_x0(self, x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma):
        self.call_count += 1
        self.last_caption_embs = caption_embs
        self.last_attention_mask = attention_mask
        self.last_lq_latent = lq_latent
        self.last_degrade_sigma = degrade_sigma
        return torch.ones_like(x)

    def _student_sample_loop(self, noise, t_list, caption_embs, lq_video_or_image, lq_latent, degrade_sigma_tensor, generator=None):
        self.call_count += 1
        self.last_caption_embs = caption_embs
        self.last_lq_latent = lq_latent
        self.last_degrade_sigma = degrade_sigma_tensor
        return torch.ones_like(noise)

    def _velocity_to_x0(self, x, v_pred, t):
        self.call_count += 1
        self.last_caption_embs = getattr(self, "last_caption_embs", None)
        return torch.ones_like(x)

    def _net_output_to_velocity(self, x, v_pred, t_cur_batch, prediction_type):
        return torch.zeros_like(x)

    def encode_lq_latent(self, image):
        self.last_image = image
        batch, _, height, width = image.shape
        return torch.zeros(
            (batch, self.encode_latent_channels, height // self.encode_compression, width // self.encode_compression),
            dtype=torch.float32,
            device=image.device,
        )


class DummyComfyClipTextEncoder:
    def __init__(self):
        self.patcher = object()
        self.seen_prompts = []

    def tokenize(self, text):
        self.seen_prompts.append(text)
        return {"text": text}

    def encode_from_tokens(self, tokens, return_dict=False):
        prompt_len = float(len(tokens["text"]))
        return {
            "cond": torch.full(
                (1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM),
                prompt_len,
                dtype=torch.float32,
            ),
            "attention_mask": torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
        }


class DummyNativeBaseModel:
    def __init__(self):
        self.diffusion_model = mock.Mock()
        self.manual_cast_dtype = torch.bfloat16

    def get_dtype_inference(self):
        return torch.bfloat16

    def apply_model(self, *args, **kwargs):
        raise AssertionError("apply_model must not be used in the native PiD distill path")


class _FakeLinearOut:
    def __init__(self, out):
        self.out = out

    def __call__(self, _x):
        return self.out.clone()


class _FakeNormFloat32:
    def __call__(self, x):
        return x.float()


class PromptAwareModel(DummyModel):
    def _encode_text_raw(self, captions):
        values = [float(sum(ord(ch) for ch in caption) % 17) / 8.0 - 1.0 for caption in captions]
        caption_embs = torch.stack(
            [
                torch.full((pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), value, dtype=torch.float32)
                for value in values
            ],
            dim=0,
        )
        attention_mask = torch.ones((len(captions), pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64)
        return caption_embs, attention_mask

    def predict_x0(self, x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma):
        self.call_count += 1
        self.last_caption_embs = caption_embs
        self.last_attention_mask = attention_mask
        self.last_lq_latent = lq_latent
        self.last_degrade_sigma = degrade_sigma
        value = caption_embs.mean(dim=(1, 2), keepdim=True).view(-1, 1, 1, 1).to(dtype=x.dtype, device=x.device)
        return torch.ones_like(x) * value.clamp(-1.0, 1.0)


class PiDRuntimeTests(unittest.TestCase):
    def setUp(self):
        pid_runtime._ACTIVE_RUNTIME = None
        pid_runtime._HANDLE_CACHE.clear()
        pid_runtime._VAE_ONLY_CACHE.clear()
        pid_runtime._PROMPT_CACHE.clear()

    def _register_runtime(self, model, backbone="flux", ckpt_type="2k", latent_channels=16, latent_compression=8):
        handle = pid_runtime.PiDHandle(
            backbone=backbone,
            ckpt_type=ckpt_type,
            pid_scale=4,
            latent_channels=latent_channels,
            latent_compression=latent_compression,
            input_caption_key="caption",
            checkpoint_path="dummy",
            device="cpu",
        )
        pid_runtime._HANDLE_CACHE[(backbone, ckpt_type)] = handle
        pid_runtime._ACTIVE_RUNTIME = pid_runtime._PiDLoadedRuntime(cache_key=(backbone, ckpt_type), model=model)
        return handle

    def _make_light_model(self, text_encoder=None):
        return pid_runtime._LightPiDModel(
            net=mock.Mock(),
            config=pid_runtime._native_pid_config("flux"),
            precision=torch.float32,
            autocast_dtype=None,
            fm_timescale=1000.0,
            text_encoder=text_encoder,
        )

    def test_decode_latent_converts_to_comfy_image(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        latent = {"samples": torch.zeros((1, 16, 2, 3), dtype=torch.float32)}

        image = pid_runtime.decode_latent(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=7,
            degrade_sigma=0.25,
        )

        self.assertEqual(tuple(image.shape), (1, 64, 96, 3))
        self.assertTrue(torch.allclose(image, torch.ones_like(image)))
        self.assertEqual(tuple(model.last_lq_latent.shape), (1, 16, 2, 3))
        self.assertEqual(tuple(model.last_caption_embs.shape), (1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM))
        self.assertTrue(torch.allclose(model.last_degrade_sigma, torch.tensor([0.25])))

    def test_decode_latent_validates_channel_count(self):
        handle = self._register_runtime(DummyModel(), backbone="sd3")
        latent = {"samples": torch.zeros((1, 8, 2, 2), dtype=torch.float32)}

        with self.assertRaises(ValueError):
            pid_runtime.decode_latent(
                handle=handle,
                latent=latent,
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=0,
                degrade_sigma=0.0,
            )

    def test_encode_image_to_latent_uses_pid_encoder(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        latent = pid_runtime.encode_image_to_latent(
            handle=handle,
            image=torch.ones((1, 64, 96, 3), dtype=torch.float32),
        )

        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 8, 12))
        self.assertEqual(tuple(model.last_image.shape), (1, 3, 64, 96))
        self.assertTrue(torch.allclose(model.last_image, torch.ones_like(model.last_image)))
        self.assertNotIn(pid_runtime.LATENT_IMAGE_GEOMETRY_KEY, latent)

    def test_encode_image_to_latent_offloads_net_before_vae_encode(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        with mock.patch.object(pid_runtime, "_set_runtime_net_device") as patched_device:
            pid_runtime.encode_image_to_latent(
                handle=handle,
                image=torch.ones((1, 64, 96, 3), dtype=torch.float32),
            )

        self.assertTrue(any(call.args[1] == "cpu" for call in patched_device.call_args_list))

    def test_encode_image_to_latent_tiles_large_inputs(self):
        model = DummyModel()
        model.encode_lq_latent = mock.Mock(side_effect=model.encode_lq_latent)
        handle = self._register_runtime(model, backbone="flux", latent_channels=16, latent_compression=8)

        latent = pid_runtime.encode_image_to_latent(
            handle=handle,
            image=torch.ones((1, 64, 96, 3), dtype=torch.float32),
            encode_tile_size=32,
        )

        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 8, 12))
        self.assertGreater(model.encode_lq_latent.call_count, 1)

    def test_encode_image_to_latent_autocorrects_flux_alignment(self):
        model = DummyModel()
        handle = self._register_runtime(model, backbone="flux", latent_channels=16, latent_compression=8)

        latent = pid_runtime.encode_image_to_latent(
            handle=handle,
            image=torch.ones((1, 70, 95, 3), dtype=torch.float32),
        )

        self.assertEqual(tuple(model.last_image.shape), (1, 3, 96, 96))
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 12, 12))
        self.assertEqual(
            latent[pid_runtime.LATENT_IMAGE_GEOMETRY_KEY],
            {
                "original_height": 70,
                "original_width": 95,
                "aligned_height": 96,
                "aligned_width": 96,
                "pad_top": 0,
                "pad_bottom": 26,
                "pad_left": 0,
                "pad_right": 1,
                "pid_scale": 4,
            },
        )

    def test_encode_image_to_latent_autocorrects_flux2_alignment(self):
        model = DummyModel(encode_latent_channels=128, encode_compression=16)
        handle = self._register_runtime(model, backbone="flux2", latent_channels=128, latent_compression=16)

        latent = pid_runtime.encode_image_to_latent(
            handle=handle,
            image=torch.ones((1, 1000, 1537, 3), dtype=torch.float32),
        )

        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 64, 100))

    def test_encode_image_to_latent_auto_tiles_large_flux2_inputs(self):
        model = DummyModel(encode_latent_channels=128, encode_compression=16)
        model.encode_lq_latent = mock.Mock(side_effect=model.encode_lq_latent)
        handle = self._register_runtime(model, backbone="flux2", latent_channels=128, latent_compression=16)

        latent = pid_runtime.encode_image_to_latent(
            handle=handle,
            image=torch.ones((1, 1000, 1537, 3), dtype=torch.float32),
        )

        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 64, 100))
        self.assertGreater(model.encode_lq_latent.call_count, 1)
        self.assertLessEqual(int(model.last_image.shape[-1]), 512)
        self.assertLessEqual(int(model.last_image.shape[-2]), 512)

    def test_autocorrect_encode_image_tensor_preserves_pixels_without_resizing(self):
        handle = self._register_runtime(DummyModel(), backbone="flux", latent_channels=16, latent_compression=8)
        image = torch.linspace(0.0, 1.0, steps=1 * 70 * 95 * 3, dtype=torch.float32).reshape(1, 70, 95, 3)

        corrected = pid_runtime._autocorrect_encode_image_tensor(handle, image)

        self.assertEqual(tuple(corrected.shape), (1, 96, 96, 3))
        self.assertTrue(torch.equal(corrected[:, :70, :95, :], image))
        self.assertTrue(torch.equal(corrected[:, 70:96, :95, :], image[:, 69:70, :, :].expand(-1, 26, -1, -1)))
        self.assertTrue(torch.equal(corrected[:, :, 95:96, :], corrected[:, :, 94:95, :]))

    def test_decode_latent_restores_original_output_geometry_after_encode_padding(self):
        model = DummyModel()
        handle = self._register_runtime(model, backbone="flux", latent_channels=16, latent_compression=8)
        latent = {
            "samples": torch.zeros((1, 16, 12, 12), dtype=torch.float32),
            pid_runtime.LATENT_IMAGE_GEOMETRY_KEY: {
                "original_height": 70,
                "original_width": 95,
                "aligned_height": 96,
                "aligned_width": 96,
                "pad_top": 0,
                "pad_bottom": 26,
                "pad_left": 0,
                "pad_right": 1,
                "pid_scale": 4,
            },
        }

        image = pid_runtime.decode_latent(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
        )

        self.assertEqual(tuple(image.shape), (1, 280, 380, 3))

    def test_decode_latent_restores_geometry_from_encode_reference_when_metadata_is_missing(self):
        model = DummyModel()
        handle = self._register_runtime(model, backbone="flux", latent_channels=16, latent_compression=8)
        latent = {
            "samples": torch.zeros((1, 16, 12, 12), dtype=torch.float32),
            pid_runtime.LATENT_REFERENCE_IMAGE_KEY: torch.ones((1, 70, 95, 3), dtype=torch.float32),
        }

        image = pid_runtime.decode_latent(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
        )

        self.assertEqual(tuple(image.shape), (1, 280, 380, 3))

    def test_decode_latent_tiled_restores_geometry_from_encode_reference_when_metadata_is_missing(self):
        model = DummyModel()
        handle = self._register_runtime(model, backbone="flux", latent_channels=16, latent_compression=8)
        latent = {
            "samples": torch.zeros((1, 16, 12, 12), dtype=torch.float32),
            pid_runtime.LATENT_REFERENCE_IMAGE_KEY: torch.ones((1, 70, 95, 3), dtype=torch.float32),
        }

        image = pid_runtime.decode_latent_tiled(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            tile_size=64,
            tile_overlap=32,
            tile_batch_size=1,
        )

        self.assertEqual(tuple(image.shape), (1, 280, 380, 3))

    def test_decode_latent_tiled_blends_tiles_into_full_image(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        image = pid_runtime.decode_latent_tiled(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=3,
            degrade_sigma=0.0,
            tile_size=64,
            tile_overlap=32,
            tile_batch_size=1,
        )

        self.assertEqual(tuple(image.shape), (1, 128, 128, 3))
        self.assertTrue(torch.allclose(image, torch.ones_like(image)))
        self.assertEqual(model.call_count, 4)

    def test_decode_latent_tiled_restores_original_output_geometry_after_encode_padding(self):
        model = DummyModel()
        handle = self._register_runtime(model, backbone="flux", latent_channels=16, latent_compression=8)

        image = pid_runtime.decode_latent_tiled(
            handle=handle,
            latent={
                "samples": torch.zeros((1, 16, 12, 12), dtype=torch.float32),
                pid_runtime.LATENT_IMAGE_GEOMETRY_KEY: {
                    "original_height": 70,
                    "original_width": 95,
                    "aligned_height": 96,
                    "aligned_width": 96,
                    "pad_top": 0,
                    "pad_bottom": 26,
                    "pad_left": 0,
                    "pad_right": 1,
                    "pid_scale": 4,
                },
            },
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=3,
            degrade_sigma=0.0,
            tile_size=64,
            tile_overlap=32,
            tile_batch_size=1,
        )

        self.assertEqual(tuple(image.shape), (1, 280, 380, 3))

    def test_decode_latent_tiled_preserves_custom_rectangular_resolution(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        image = pid_runtime.decode_latent_tiled(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 4, 6), dtype=torch.float32)},
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=3,
            degrade_sigma=0.0,
            tile_size=64,
            tile_overlap=32,
            tile_batch_size=1,
        )

        self.assertEqual(tuple(image.shape), (1, 128, 192, 3))
        self.assertEqual(tuple(model.last_lq_latent.shape[-2:]), (4, 6))

    def test_decode_latent_tiled_applies_sde_noise_boost(self):
        handle = self._register_runtime(DummyModel())
        pid_prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM)),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="cat",
        )

        with (
            mock.patch.object(pid_runtime, "_resolve_pid_prompt", return_value=pid_prompt),
            mock.patch.object(
                pid_runtime,
                "_decode_samples",
                return_value=torch.ones((1, 3, 1, 128, 128), dtype=torch.float32),
            ) as patched_decode,
        ):
            pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=3,
                degrade_sigma=0.0,
                tile_size=64,
                tile_overlap=32,
                tile_batch_size=1,
                sde_noise_strength=1.0,
                tiled_sde_noise_boost=1.35,
            )

        self.assertAlmostEqual(patched_decode.call_args.kwargs["sde_noise_strength"], 1.35)

    def test_decode_latent_tiled_zero_sde_boost_keeps_base_sde_strength(self):
        handle = self._register_runtime(DummyModel())
        pid_prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM)),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="cat",
        )

        with (
            mock.patch.object(pid_runtime, "_resolve_pid_prompt", return_value=pid_prompt),
            mock.patch.object(
                pid_runtime,
                "_decode_samples",
                return_value=torch.ones((1, 3, 1, 128, 128), dtype=torch.float32),
            ) as patched_decode,
        ):
            pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=3,
                degrade_sigma=0.0,
                tile_size=64,
                tile_overlap=32,
                tile_batch_size=1,
                sde_noise_strength=1.0,
                tiled_sde_noise_boost=0.0,
            )

        self.assertAlmostEqual(patched_decode.call_args.kwargs["sde_noise_strength"], 1.0)

    def test_decode_latent_tiled_refines_seams_with_shifted_second_pass(self):
        handle = self._register_runtime(DummyModel())
        pid_prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM)),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="cat",
        )
        primary = torch.zeros((1, 3, 1, 1024, 1024), dtype=torch.float32)
        refined = torch.ones((1, 3, 1, 1024, 1024), dtype=torch.float32)

        with (
            mock.patch.object(pid_runtime, "_resolve_pid_prompt", return_value=pid_prompt),
            mock.patch.object(pid_runtime, "_decode_samples", side_effect=[primary, refined]) as patched_decode,
        ):
            image = pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 16, 32, 32), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=3,
                degrade_sigma=0.0,
                tile_size=512,
                tile_overlap=64,
                tile_batch_size=1,
                seam_refine=True,
                seam_refine_strength=0.25,
            )

        self.assertEqual(patched_decode.call_count, 2)
        refine_kwargs = patched_decode.call_args_list[1].kwargs
        self.assertEqual(refine_kwargs["tile_grid_offset"], 224)
        self.assertEqual(refine_kwargs["seed"], 3)
        self.assertAlmostEqual(refine_kwargs["source_denoise_strength"], 0.25)
        self.assertAlmostEqual(refine_kwargs["source_detail_noise_boost"], 0.0)
        self.assertAlmostEqual(refine_kwargs["sde_noise_strength"], 0.0)
        self.assertAlmostEqual(image[0, 0, 0, 0].item(), 0.5)
        self.assertGreater(image[0, 300, 300, 0].item(), 0.5)

    def test_decode_latent_tiled_batches_same_size_tiles(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        image = pid_runtime.decode_latent_tiled(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=3,
            degrade_sigma=0.0,
            tile_size=64,
            tile_overlap=32,
            tile_batch_size=4,
        )

        self.assertEqual(tuple(image.shape), (1, 128, 128, 3))
        self.assertTrue(torch.allclose(image, torch.ones_like(image)))

    def test_decode_latent_tiled_requests_intermediate_previews(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        fake_progress = mock.Mock()

        def fake_decode_samples(**kwargs):
            step_preview_callback = kwargs.get("step_preview_callback")
            if step_preview_callback is not None:
                latent_batch = kwargs["latent_tensor"]
                batch = int(latent_batch.shape[0])
                height = int(latent_batch.shape[-2]) * handle.latent_compression * handle.pid_scale
                width = int(latent_batch.shape[-1]) * handle.latent_compression * handle.pid_scale
                step_preview_callback(torch.ones((batch, 3, height, width), dtype=torch.float32))
            return torch.ones((1, 3, 1, 128, 128), dtype=torch.float32)

        with (
            mock.patch.object(pid_runtime, "_decode_samples", side_effect=fake_decode_samples) as patched_decode,
            mock.patch.object(pid_runtime, "_DecodeProgress", return_value=fake_progress),
        ):
            pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=3,
                degrade_sigma=0.0,
                tile_size=64,
                tile_overlap=32,
                tile_batch_size=1,
            )

        self.assertTrue(patched_decode.called)
        self.assertTrue(all(call.kwargs["progress"] is fake_progress for call in patched_decode.call_args_list))
        self.assertTrue(all(call.kwargs["preview_enabled"] for call in patched_decode.call_args_list))
        self.assertTrue(all(call.kwargs["progress_advance"] > 0 for call in patched_decode.call_args_list))
        self.assertTrue(all(call.kwargs["step_preview_callback"] is not None for call in patched_decode.call_args_list))
        self.assertTrue(fake_progress.update.called)
        self.assertTrue(all(call.kwargs["advance"] > 0 for call in fake_progress.update.call_args_list[:-1]))
        self.assertTrue(all(call.kwargs["emit_bar"] for call in fake_progress.update.call_args_list))

    def test_decode_latent_tiled_updates_preview_for_each_composited_tile(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        fake_progress = mock.Mock()

        def fake_decode_samples(**kwargs):
            step_preview_callback = kwargs.get("step_preview_callback")
            latent_batch = kwargs["latent_tensor"]
            batch = int(latent_batch.shape[0])
            height = int(latent_batch.shape[-2]) * handle.latent_compression * handle.pid_scale
            width = int(latent_batch.shape[-1]) * handle.latent_compression * handle.pid_scale
            if step_preview_callback is not None:
                step_preview_callback(torch.ones((batch, 3, height, width), dtype=torch.float32))
            return torch.ones((batch, 3, 1, height, width), dtype=torch.float32)

        with (
            mock.patch.object(pid_runtime, "_decode_samples", side_effect=fake_decode_samples) as patched_decode,
            mock.patch.object(pid_runtime, "_DecodeProgress", return_value=fake_progress),
        ):
            pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=3,
                degrade_sigma=0.0,
                tile_size=64,
                tile_overlap=32,
                tile_batch_size=2,
            )

        self.assertTrue(patched_decode.called)
        self.assertGreater(len(fake_progress.update.call_args_list), len(patched_decode.call_args_list))
        self.assertEqual(fake_progress.update.call_args_list[-1].kwargs["advance"], 0)

    def test_decode_latent_tiled_step_preview_callback_accepts_bchw_images(self):
        handle = self._register_runtime(DummyModel())
        fake_progress = mock.Mock()

        def fake_decode_samples(**kwargs):
            step_preview_callback = kwargs.get("step_preview_callback")
            latent_batch = kwargs["latent_tensor"]
            batch = int(latent_batch.shape[0])
            height = int(latent_batch.shape[-2]) * handle.latent_compression * handle.pid_scale
            width = int(latent_batch.shape[-1]) * handle.latent_compression * handle.pid_scale
            if step_preview_callback is not None:
                step_preview_callback(torch.ones((batch, 3, height, width), dtype=torch.float32))
            return torch.ones((batch, 3, 1, height, width), dtype=torch.float32)

        with (
            mock.patch.object(pid_runtime, "_decode_samples", side_effect=fake_decode_samples),
            mock.patch.object(pid_runtime, "_DecodeProgress", return_value=fake_progress),
        ):
            image = pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=3,
                degrade_sigma=0.0,
                tile_size=64,
                tile_overlap=32,
                tile_batch_size=1,
            )

        self.assertEqual(tuple(image.shape), (1, 128, 128, 3))
        self.assertTrue(fake_progress.update.called)

    def test_decode_progress_renders_cli_steps_iterations_per_second_and_eta(self):
        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        fake_stderr = FakeTTY()
        fake_bar = mock.Mock()
        fake_utils = types.ModuleType("comfy.utils")
        fake_utils.ProgressBar = mock.Mock(return_value=fake_bar)

        with (
            mock.patch.object(pid_runtime.sys, "stderr", fake_stderr),
            mock.patch.object(pid_runtime.time, "perf_counter", side_effect=[10.0, 10.0, 11.0, 12.0]),
            mock.patch.dict(sys.modules, {"comfy.utils": fake_utils}),
        ):
            progress = pid_runtime._DecodeProgress(total=4)
            progress.update(advance=2, emit_bar=False)
            progress.update(advance=2, emit_bar=True)

        output = fake_stderr.getvalue()
        self.assertIn("0/4 steps", output)
        self.assertIn("2/4 steps", output)
        self.assertIn("4/4 steps", output)
        self.assertIn("it/s", output)
        self.assertIn("ETA", output)
        self.assertIn("00:01", output)
        self.assertIn("00:00", output)
        self.assertTrue(output.endswith("\n"))
        fake_bar.update_absolute.assert_has_calls([mock.call(0, 4), mock.call(4, 4, None)])

    def test_decode_progress_sends_legacy_preview_fallback(self):
        fake_bar = mock.Mock()
        fake_utils = types.ModuleType("comfy.utils")
        fake_utils.ProgressBar = mock.Mock(return_value=fake_bar)
        fake_protocol = types.ModuleType("protocol")
        fake_protocol.BinaryEventTypes = types.SimpleNamespace(UNENCODED_PREVIEW_IMAGE=2)
        fake_server_module = types.ModuleType("server")
        fake_server_module.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(client_id="cid", send_sync=mock.Mock()))

        with (
            mock.patch.dict(sys.modules, {"comfy.utils": fake_utils, "protocol": fake_protocol, "server": fake_server_module}),
            mock.patch.object(pid_runtime.time, "perf_counter", side_effect=[10.0, 10.0, 10.0, 10.0, 10.0]),
        ):
            progress = pid_runtime._DecodeProgress(total=4)
            preview = ("JPEG", object(), 512)
            progress.update(advance=1, preview=preview, emit_bar=True)

        fake_server_module.PromptServer.instance.send_sync.assert_called_once_with(2, preview, "cid")

    def test_webp_preview_tuple_and_capped_size(self):
        img = torch.ones((1, 3, 512, 512), dtype=torch.float32)
        preview_fmt, preview_img, preview_sz = pid_runtime._make_preview_tuple(img)
        self.assertEqual(preview_fmt, "WEBP")
        self.assertLessEqual(preview_sz, 256)
        self.assertLessEqual(preview_img.size[0], 256)
        self.assertLessEqual(preview_img.size[1], 256)

    def test_decode_samples_uses_deterministic_step_when_sde_noise_strength_is_zero(self):
        class DeterministicModel:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    student_sample_steps=2,
                    student_timestep=1.0,
                    prediction_type="velocity",
                )
                self.fm_trainer = types.SimpleNamespace(timescale=1000.0)
                self.autocast_dtype = None
                self.net = mock.Mock()
                self.net.eval = mock.Mock()
                self.last_x = None

            def predict_x0(self, x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma):
                self.last_x = x.detach().clone()
                return torch.zeros_like(x)

            def _get_t_list(self, device, num_steps=None, scheduler="original"):
                return torch.tensor([1.0, 0.5, 0.0], device=device, dtype=torch.float32)

        model = DeterministicModel()
        handle = self._register_runtime(model)
        latent = torch.zeros((1, handle.latent_channels, 1, 1), dtype=torch.float32)
        pid_prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )
        initial_noise = torch.ones((1, 3, handle.latent_compression * handle.pid_scale, handle.latent_compression * handle.pid_scale))

        samples = pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=latent,
            pid_prompt=pid_prompt,
            cfg_scale=1.0,
            pid_inference_steps=2,
            seed=0,
            degrade_sigma=0.0,
            noise=initial_noise.clone(),
            scheduler="original",
            sde_noise_strength=0.0,
        )

        expected_after_first_step = initial_noise + (0.5 - 1.0) * initial_noise
        self.assertTrue(torch.allclose(model.last_x, expected_after_first_step, atol=1e-5))
        self.assertTrue(torch.allclose(samples.squeeze(2), torch.zeros_like(samples.squeeze(2)), atol=1e-5))

    def test_decode_samples_blends_low_sde_strength_with_deterministic_step(self):
        class ConservativeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    student_sample_steps=2,
                    student_timestep=1.0,
                    prediction_type="velocity",
                )
                self.fm_trainer = types.SimpleNamespace(timescale=1000.0)
                self.autocast_dtype = None
                self.net = mock.Mock()
                self.net.eval = mock.Mock()
                self.last_x = None

            def predict_x0(self, x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma):
                self.last_x = x.detach().clone()
                return torch.zeros_like(x)

            def _get_t_list(self, device, num_steps=None, scheduler="original"):
                return torch.tensor([1.0, 0.5, 0.0], device=device, dtype=torch.float32)

        model = ConservativeModel()
        handle = self._register_runtime(model)
        latent = torch.zeros((1, handle.latent_channels, 1, 1), dtype=torch.float32)
        pid_prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )
        initial_noise = torch.ones((1, 3, handle.latent_compression * handle.pid_scale, handle.latent_compression * handle.pid_scale))

        pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=latent,
            pid_prompt=pid_prompt,
            cfg_scale=1.0,
            pid_inference_steps=2,
            seed=0,
            degrade_sigma=0.0,
            noise=initial_noise.clone(),
            noise_list=[torch.zeros_like(initial_noise)],
            scheduler="original",
            sde_noise_strength=0.25,
        )

        expected_after_first_step = torch.full_like(initial_noise, 0.375)
        self.assertTrue(torch.allclose(model.last_x, expected_after_first_step, atol=1e-5))

    def test_native_lq_latent_process_in_unknown_backbone_is_identity(self):
        fake_comfy = types.ModuleType("comfy")
        fake_comfy_latent = types.ModuleType("comfy.latent_formats")
        fake_comfy.latent_formats = fake_comfy_latent

        with mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.latent_formats": fake_comfy_latent}):
            test_tensor = torch.ones((1, 3, 32, 32))
            res = pid_runtime._native_lq_latent_process_in("unknown", test_tensor)
            self.assertTrue(torch.equal(res, test_tensor))

    def test_shifted_tile_starts_offset_the_interior_grid_and_keep_edges(self):
        starts = pid_runtime._compute_shifted_tile_starts(total=10, tile=4, overlap=1, offset=2)
        self.assertEqual(starts, [0, 2, 5, 6])

    def test_seam_refine_mask_targets_overlaps_and_intersections(self):
        mask = pid_runtime._make_seam_refine_mask(
            latent_h=4,
            latent_w=4,
            tile_latent=2,
            overlap_latent=1,
            compression=8,
            pid_scale=4,
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 1, 128, 128))
        self.assertEqual(mask[0, 0, 0, 0, 0].item(), 0.0)
        self.assertGreater(mask[0, 0, 0, 48, 48].item(), 0.0)

    def test_removed_backbones_are_rejected(self):
        for backbone in ("dinov2", "siglip"):
            with self.subTest(backbone=backbone):
                with self.assertRaises(ValueError):
                    pid_runtime._asset_patterns(backbone, "2k")
                with self.assertRaises(ValueError):
                    pid_runtime._instantiate_vae_encoder(backbone)

    def test_upstream_20260602_backbones_and_flux2_checkpoint_are_registered(self):
        self.assertIn("sdxl", pid_runtime.SUPPORTED_BACKBONES)
        self.assertIn("qwenimage", pid_runtime.SUPPORTED_BACKBONES)
        self.assertIn("qwenimage-2512", pid_runtime.SUPPORTED_BACKBONES)
        self.assertIn("zimage", pid_runtime.SUPPORTED_BACKBONES)
        self.assertIn("zimage-turbo", pid_runtime.SUPPORTED_BACKBONES)
        self.assertIn("flux2-klein-4b", pid_runtime.SUPPORTED_BACKBONES)
        self.assertIn("flux2-klein-9b", pid_runtime.SUPPORTED_BACKBONES)

        flux2_patterns = pid_runtime._asset_patterns("flux2", "2kto4k")
        self.assertIn("checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/*", flux2_patterns)
        self.assertNotIn("checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step/*", flux2_patterns)

        sdxl_patterns = pid_runtime._asset_patterns("sdxl", "2kto4k")
        self.assertIn("checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/*", sdxl_patterns)
        self.assertIn("checkpoints/sdxl_vae.safetensors", sdxl_patterns)

        qwen_patterns = pid_runtime._asset_patterns("qwenimage-2512", "2kto4k")
        self.assertIn("checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/*", qwen_patterns)
        self.assertIn("checkpoints/QwenImage_VAE_2d.pth", qwen_patterns)

        zimage_patterns = pid_runtime._asset_patterns("zimage-turbo", "2kto4k")
        self.assertIn("checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/*", zimage_patterns)
        self.assertIn("checkpoints/ae.safetensors", zimage_patterns)

        with self.assertRaises(ValueError):
            pid_runtime._asset_patterns("sdxl", "2k")
        with self.assertRaises(ValueError):
            pid_runtime._asset_patterns("qwenimage", "2k")

    def test_upstream_checkpoint_registry_excludes_dinov2_siglip_and_uses_flux2_2606(self):
        pid_runtime._ensure_upstream_path()
        from pid._src.inference.checkpoint_registry import get_pid_checkpoint

        flux2 = get_pid_checkpoint("flux2", "2kto4k")
        self.assertEqual(flux2.experiment, "PiD_res2kto4k_sr4x_official_flux2_distill_4step")
        self.assertIn("PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606", flux2.checkpoint_path)

        klein = get_pid_checkpoint("flux2-klein-9b", "2kto4k")
        self.assertEqual(klein, flux2)
        zimage = get_pid_checkpoint("zimage", "2kto4k")
        self.assertEqual(zimage.experiment, "PiD_res2kto4k_sr4x_official_flux_distill_4step")
        qwen = get_pid_checkpoint("qwenimage-2512", "2kto4k")
        self.assertEqual(qwen.experiment, "PiD_res2kto4k_sr4x_official_qwenimage_distill_4step")

        with self.assertRaises(KeyError):
            get_pid_checkpoint("dinov2", "2k")
        with self.assertRaises(KeyError):
            get_pid_checkpoint("siglip", "2k")

    def test_checkpoint_variant_auto_and_single_variant_backbones_are_normalized(self):
        self.assertEqual(pid_runtime.normalize_checkpoint_variant("sdxl", "auto"), "2kto4k")
        self.assertEqual(pid_runtime.normalize_checkpoint_variant("sdxl", "2k"), "2kto4k")
        self.assertEqual(pid_runtime.normalize_checkpoint_variant("qwenimage", "2k"), "2kto4k")
        self.assertEqual(pid_runtime.normalize_checkpoint_variant("flux", "auto"), "2k")
        with self.assertRaises(ValueError):
            pid_runtime.normalize_checkpoint_variant("flux", "bad")

    def test_group_tile_jobs_by_decode_shape_batches_non_consecutive_matches(self):
        jobs = [
            pid_runtime._ExpandedTileDecodeJob(0, 0, 4, 4, 0, 0, 4, 4, 2, 2, 0, 0),
            pid_runtime._ExpandedTileDecodeJob(0, 0, 2, 4, 0, 0, 2, 4, 2, 4, 0, 0),
            pid_runtime._ExpandedTileDecodeJob(4, 4, 8, 8, 4, 4, 8, 8, 6, 6, 0, 0),
        ]

        grouped = pid_runtime._group_tile_jobs_by_decode_shape(jobs)

        self.assertEqual([len(group) for group in grouped], [2, 1])
        self.assertEqual(grouped[0][0], jobs[0])
        self.assertEqual(grouped[0][1], jobs[2])
        self.assertEqual(grouped[1][0], jobs[1])

    def test_group_tile_jobs_by_decode_window_reuses_identical_expanded_context(self):
        jobs = [
            pid_runtime._ExpandedTileDecodeJob(0, 0, 4, 4, 0, 0, 8, 8, 0, 0, 0, 0),
            pid_runtime._ExpandedTileDecodeJob(0, 4, 4, 8, 0, 0, 8, 8, 0, 4, 0, 4),
            pid_runtime._ExpandedTileDecodeJob(4, 0, 8, 4, 0, 0, 8, 8, 4, 0, 4, 0),
        ]

        grouped = pid_runtime._group_tile_jobs_by_decode_window(jobs)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0], jobs)

    def test_match_tile_moments_to_reference_softens_tile_color_drift(self):
        tile = torch.ones((1, 3, 8, 8), dtype=torch.float32) * 0.8
        reference = torch.ones((1, 3, 8, 8), dtype=torch.float32) * 0.2

        matched = pid_runtime._match_tile_moments_to_reference(tile, reference, strength=0.5)

        self.assertEqual(tuple(matched.shape), tuple(tile.shape))
        self.assertEqual(matched.dtype, tile.dtype)
        self.assertLess(matched.mean().item(), tile.mean().item())
        self.assertGreater(matched.mean().item(), reference.mean().item())

    def test_anchor_low_frequency_to_reference_reduces_tiled_luma_drift(self):
        prediction = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        prediction[:, :, :, 16:] = 0.8
        reference = torch.zeros_like(prediction)

        anchored = pid_runtime._anchor_low_frequency_to_reference(prediction, reference, strength=0.5, max_size=8)

        self.assertEqual(tuple(anchored.shape), tuple(prediction.shape))
        self.assertEqual(anchored.dtype, prediction.dtype)
        self.assertLess(anchored[:, :, :, 16:].mean().item(), prediction[:, :, :, 16:].mean().item())

    def test_small_tiled_decode_expands_context_window(self):
        base_job = pid_runtime._TileDecodeJob(start_y=32, start_x=32, end_y=64, end_x=64, out_y=1024, out_x=1024)
        expanded = pid_runtime._expand_tile_job(
            job=base_job,
            total_h=128,
            total_w=128,
            compression=8,
            pid_scale=4,
        )

        self.assertLess(expanded.decode_start_y, base_job.start_y)
        self.assertLess(expanded.decode_start_x, base_job.start_x)
        self.assertGreater(expanded.decode_end_y, base_job.end_y)
        self.assertGreater(expanded.decode_end_x, base_job.end_x)
        self.assertEqual(expanded.crop_y, (base_job.start_y - expanded.decode_start_y) * 32)
        self.assertEqual(expanded.crop_x, (base_job.start_x - expanded.decode_start_x) * 32)

    def test_small_tiled_decode_expands_context_window_flux2(self):
        base_job = pid_runtime._TileDecodeJob(start_y=32, start_x=32, end_y=48, end_x=48, out_y=1024, out_x=1024)
        expanded = pid_runtime._expand_tile_job(
            job=base_job,
            total_h=128,
            total_w=128,
            compression=16,
            pid_scale=4,
            min_size=1024,
        )

        self.assertEqual(expanded.decode_end_y - expanded.decode_start_y, 64)
        self.assertEqual(expanded.decode_end_x - expanded.decode_start_x, 64)
        self.assertEqual(expanded.crop_y, (base_job.start_y - expanded.decode_start_y) * 64)
        self.assertEqual(expanded.crop_x, (base_job.start_x - expanded.decode_start_x) * 64)

    def test_flux2_tiled_sampler_auto_raises_small_output_tiles(self):
        model = DummyModel()
        handle = self._register_runtime(model, backbone="flux2", latent_channels=128, latent_compression=16)

        pid_runtime.decode_latent_tiled(
            handle=handle,
            latent={"samples": torch.zeros((1, 128, 48, 48), dtype=torch.float32)},
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=1,
            seed=3,
            degrade_sigma=0.0,
            tile_size=512,
            tile_overlap=64,
            tile_batch_size=1,
        )

        self.assertEqual(tuple(model.last_lq_latent.shape[-2:]), (16, 16))
        self.assertEqual(model.call_count, 16)

    def test_decode_latent_tiled_validates_tile_alignment(self):
        handle = self._register_runtime(DummyModel(), backbone="flux2", latent_channels=128, latent_compression=16)

        with self.assertRaises(ValueError):
            pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 128, 4, 4), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=1.0,
                pid_inference_steps=4,
                seed=0,
                degrade_sigma=0.0,
                tile_size=200,
                tile_overlap=64,
                tile_batch_size=1,
            )

    def test_pid_ksampler_dispatches_to_tiled_path(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        image = pid_runtime.pid_ksampler(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
            prompt="cat",
            negative_prompt="",
            cfg_scale=5.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=True,
            tile_size=64,
            tile_overlap=32,
            tile_batch_size=2,
        )

        self.assertEqual(tuple(image.shape), (1, 128, 128, 3))

    def test_pid_ksampler_can_offload_model_net_after_sampling(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        with mock.patch.object(pid_runtime, "_set_runtime_net_device") as patched_device:
            pid_runtime.pid_ksampler(
                handle=handle,
                latent={"samples": torch.zeros((1, 16, 2, 2), dtype=torch.float32)},
                prompt="cat",
                negative_prompt="",
                cfg_scale=5.0,
                pid_inference_steps=4,
                seed=0,
                degrade_sigma=0.0,
                keep_model_loaded_on_gpu=False,
                use_tiled=False,
            )

        self.assertTrue(any(call.args[1] == "cpu" for call in patched_device.call_args_list))

    def test_resize_latent_resizes_dict_samples(self):
        latent = {
            "samples": torch.zeros((1, 16, 8, 8), dtype=torch.float32),
            "noise_mask": torch.ones((1, 8, 8), dtype=torch.float32),
            pid_runtime.LATENT_IMAGE_GEOMETRY_KEY: {"original_height": 64, "original_width": 64, "pid_scale": 4},
        }

        resized = pid_runtime.resize_latent(latent, 0.5, "bicubic")

        self.assertEqual(tuple(resized["samples"].shape), (1, 16, 4, 4))
        self.assertEqual(tuple(resized["noise_mask"].shape), (1, 4, 4))
        self.assertNotIn(pid_runtime.LATENT_IMAGE_GEOMETRY_KEY, resized)

    def test_encode_prompt_reuses_cached_embeddings(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        first = pid_runtime.encode_prompt(handle, "cat")
        second = pid_runtime.encode_prompt(handle, "cat")

        self.assertEqual(first["prompt"], "cat")
        self.assertTrue(torch.equal(first["caption_embs"], second["caption_embs"]))
        self.assertEqual(model.text_encoder.to.call_count, 2)

    def test_pid_ksampler_prompt_changes_output(self):
        model = PromptAwareModel()
        handle = self._register_runtime(model)
        latent = {"samples": torch.zeros((1, 16, 2, 2), dtype=torch.float32)}

        image_cat = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
        )
        image_dog = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="dog",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
        )

        self.assertFalse(torch.allclose(image_cat, image_dog))

    def test_pid_ksampler_prompt_overrides_connected_pid_prompt_when_text_differs(self):
        model = PromptAwareModel()
        handle = self._register_runtime(model)
        latent = {"samples": torch.zeros((1, 16, 2, 2), dtype=torch.float32)}
        cached_cat = pid_runtime.encode_prompt(handle, "cat")

        image_cat = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
            pid_prompt=cached_cat,
        )
        image_dog = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="dog",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
            pid_prompt=cached_cat,
        )

        self.assertFalse(torch.allclose(image_cat, image_dog))

    def test_pid_ksampler_cfg_scale_changes_output_strength(self):
        model = PromptAwareModel()
        handle = self._register_runtime(model)
        latent = {"samples": torch.zeros((1, 16, 2, 2), dtype=torch.float32)}

        image_cfg_1 = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
        )
        image_cfg_5 = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=5.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
        )

        self.assertFalse(torch.allclose(image_cfg_1, image_cfg_5))

    def test_pid_ksampler_negative_prompt_changes_output(self):
        model = PromptAwareModel()
        handle = self._register_runtime(model)
        latent = {"samples": torch.zeros((1, 16, 2, 2), dtype=torch.float32)}

        image_default_negative = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="",
            cfg_scale=5.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
        )
        image_custom_negative = pid_runtime.pid_ksampler(
            handle=handle,
            latent=latent,
            prompt="cat",
            negative_prompt="blurry, low quality",
            cfg_scale=5.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=False,
        )

        self.assertFalse(torch.allclose(image_default_negative, image_custom_negative))

    def test_encode_prompt_uses_external_clip_when_provided(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        clip_text_encoder = DummyComfyClipTextEncoder()

        fake_comfy = types.ModuleType("comfy")
        fake_model_management = types.ModuleType("comfy.model_management")
        fake_model_management.load_models_gpu = mock.Mock()
        fake_comfy.model_management = fake_model_management

        with (
            mock.patch.object(pid_runtime, "_is_comfy_clip_text_encoder", return_value=True),
            mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.model_management": fake_model_management}),
        ):
            encoded = pid_runtime.encode_prompt(handle, "cat", clip=clip_text_encoder)

        self.assertEqual(clip_text_encoder.seen_prompts, ["cat"])
        self.assertEqual(tuple(encoded["caption_embs"].shape), (1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM))
        self.assertEqual(model.text_encoder.to.call_count, 0)

    def test_decode_latent_accepts_preencoded_prompt(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        pid_prompt = {
            "caption_embs": torch.full((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), 3.0),
            "attention_mask": torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            "prompt": "cat",
        }

        image = pid_runtime.decode_latent(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 2, 2), dtype=torch.float32)},
            prompt="cat",
            negative_prompt="",
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            pid_prompt=pid_prompt,
        )

        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertTrue(torch.all(model.last_caption_embs == 3.0))

    def test_get_t_list_linear_interpolation(self):
        model = self._make_light_model()
        model.config = types.SimpleNamespace(
            student_timestep=1.0,
            student_sample_steps=4,
            student_t_list=[0.999, 0.866, 0.634, 0.342, 0.0]
        )
        t_list = model._get_t_list(device="cpu", num_steps=8, scheduler="original")
        self.assertEqual(len(t_list), 9)
        for i in range(len(t_list) - 1):
            self.assertGreater(t_list[i].item(), t_list[i+1].item())
        self.assertAlmostEqual(t_list[0].item(), 0.999)
        self.assertAlmostEqual(t_list[-1].item(), 0.0)

    def test_get_t_list_ignores_legacy_scheduler_names(self):
        model = self._make_light_model()
        model.config = types.SimpleNamespace(
            student_timestep=1.0,
            student_sample_steps=4,
            student_t_list=None
        )
        expected = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        for scheduler in ("original", "uniform", "cosine", "quadratic"):
            t_list = model._get_t_list(device="cpu", num_steps=4, scheduler=scheduler)
            self.assertTrue(torch.allclose(t_list, expected))

    def test_light_model_prefers_native_text_encoder_loader(self):
        clip_text_encoder = DummyComfyClipTextEncoder()
        model = self._make_light_model()

        with mock.patch.object(pid_runtime, "_load_native_text_encoder", return_value=clip_text_encoder) as patched_native:
            model._ensure_text_encoder_loaded()

        patched_native.assert_called_once_with("gemma-2-2b-it")
        self.assertIs(model.text_encoder, clip_text_encoder)

    def test_ensure_native_text_encoder_assets_downloads_repackaged_gemma_to_local_cache(self):
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = mock.Mock()

        with (
            mock.patch.object(pid_runtime, "_find_native_text_encoder_files", side_effect=[FileNotFoundError(), ([Path("a.safetensors")], Path("tokenizer.model"))]),
            mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
        ):
            text_encoder_dir = pid_runtime._ensure_native_text_encoder_assets("gemma-2-2b-it")

        self.assertEqual(text_encoder_dir, pid_runtime.UPSTREAM_ROOT / "checkpoints" / "text_encoders" / "gemma-2-2b-it")
        self.assertEqual(fake_hf.snapshot_download.call_count, 2)
        first_call = fake_hf.snapshot_download.call_args_list[0]
        second_call = fake_hf.snapshot_download.call_args_list[1]
        self.assertEqual(first_call.kwargs["repo_id"], pid_runtime.HF_PID_TEXT_ENCODER_REPO_ID)
        self.assertEqual(first_call.kwargs["allow_patterns"], [pid_runtime.HF_PID_TEXT_ENCODER_FILE])
        self.assertEqual(second_call.kwargs["repo_id"], pid_runtime.HF_PID_TEXT_ENCODER_TOKENIZER_REPO_ID)
        self.assertEqual(second_call.kwargs["allow_patterns"], [pid_runtime.HF_PID_TEXT_ENCODER_TOKENIZER_FILE])

    def test_load_pid_model_prepares_handle_without_prefetching_text_encoder(self):
        sentinel = object()
        with (
            mock.patch.object(pid_runtime, "_prepare_handle", return_value=sentinel) as patched_prepare,
            mock.patch.object(pid_runtime, "_ensure_native_text_encoder_assets") as patched_assets,
        ):
            result = pid_runtime.load_pid_model("flux", "2k")

        patched_prepare.assert_called_once_with("flux", "2k")
        patched_assets.assert_not_called()
        self.assertIs(result, sentinel)

    def test_get_encode_model_builds_vae_only_runtime_without_loading_full_model(self):
        handle = self._register_runtime(DummyModel())
        pid_runtime._ACTIVE_RUNTIME = None

        fake_vae = mock.Mock()
        with (
            mock.patch.object(pid_runtime, "_prepare_handle", return_value=handle) as patched_prepare,
            mock.patch.object(pid_runtime, "_instantiate_vae_encoder", return_value=fake_vae) as patched_vae,
        ):
            model = pid_runtime._get_encode_model(handle)

        patched_prepare.assert_called_once_with(handle.backbone, handle.ckpt_type)
        patched_vae.assert_called_once()
        self.assertIs(model.vae_encoder, fake_vae)
        self.assertIsNone(model.text_encoder)
        self.assertIsNone(model.net)

    def test_load_native_model_keeps_native_inference_dtype(self):
        fake_base_model = DummyNativeBaseModel()
        fake_patcher = mock.Mock()
        fake_patcher.model = fake_base_model

        fake_comfy = types.ModuleType("comfy")
        fake_sd = types.ModuleType("comfy.sd")
        fake_sd.load_diffusion_model = mock.Mock(return_value=fake_patcher)
        fake_comfy.sd = fake_sd

        with (
            mock.patch.object(pid_runtime, "_patch_native_pid_attention_dtype_mismatch") as patched_attn,
            mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.sd": fake_sd}),
        ):
            model = pid_runtime._load_native_model("flux", Path("dummy"))

        patched_attn.assert_called_once()
        self.assertEqual(model.precision, torch.bfloat16)
        self.assertEqual(model.autocast_dtype, torch.bfloat16)
        self.assertEqual(fake_base_model.manual_cast_dtype, torch.bfloat16)

    def test_native_model_predict_x0_uses_diffusion_model_directly(self):
        fake_base_model = DummyNativeBaseModel()
        fake_base_model.diffusion_model = mock.Mock(return_value=torch.full((1, 3, 8, 8), 0.25, dtype=torch.float32))
        model = pid_runtime._NativePiDModel(
            patcher=mock.Mock(),
            base_model=fake_base_model,
            config=pid_runtime._native_pid_config("flux"),
        )
        model.autocast_dtype = None

        x = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
        caption_embs = torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32)
        attention_mask = torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64)
        lq_latent = torch.zeros((1, 16, 2, 2), dtype=torch.float32)
        degrade_sigma = torch.zeros((1,), dtype=torch.float32)

        x0 = model.predict_x0(x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma)

        call_args = fake_base_model.diffusion_model.call_args
        self.assertIsNotNone(call_args)
        self.assertTrue(torch.allclose(call_args.args[1], torch.tensor([500.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(call_args.kwargs["context"], caption_embs))
        self.assertTrue(torch.equal(call_args.kwargs["attention_mask"], attention_mask))
        self.assertTrue(torch.equal(call_args.kwargs["lq_latent"], lq_latent))
        self.assertTrue(torch.equal(call_args.kwargs["degrade_sigma"], degrade_sigma))
        self.assertEqual(tuple(x0.shape), (1, 3, 8, 8))

    def test_patch_native_pid_attention_dtype_mismatch_upcasts_value_to_query_dtype(self):
        captured = {}

        class FakeJointAttention:
            def __init__(self):
                self.num_heads = 1
                self.head_dim = 2
                self.qkv_x = _FakeLinearOut(torch.tensor([[[1, 2, 3, 4, 5, 6]]], dtype=torch.bfloat16))
                self.qkv_y = _FakeLinearOut(torch.tensor([[[7, 8, 9, 10, 11, 12]]], dtype=torch.bfloat16))
                self.q_norm_x = _FakeNormFloat32()
                self.k_norm_x = _FakeNormFloat32()
                self.q_norm_y = _FakeNormFloat32()
                self.k_norm_y = _FakeNormFloat32()
                self.proj_x = lambda x: x
                self.proj_y = lambda y: y

        fake_model_module = types.ModuleType("comfy.ldm.pixeldit.model")

        def fake_apply_rope(q, k, _pos):
            return q, k

        def fake_optimized_attention(q, k, v, *_args, **_kwargs):
            captured["q_dtype"] = q.dtype
            captured["k_dtype"] = k.dtype
            captured["v_dtype"] = v.dtype
            return torch.zeros_like(q)

        fake_model_module.MMDiTJointAttention = FakeJointAttention
        fake_model_module.apply_rope = fake_apply_rope
        fake_model_module.optimized_attention = fake_optimized_attention
        fake_modules_module = types.ModuleType("comfy.ldm.pixeldit.modules")
        fake_modules_module.RotaryAttention = None

        fake_comfy = types.ModuleType("comfy")
        fake_ldm = types.ModuleType("comfy.ldm")
        fake_pixeldit = types.ModuleType("comfy.ldm.pixeldit")
        fake_comfy.ldm = fake_ldm
        fake_ldm.pixeldit = fake_pixeldit
        fake_pixeldit.model = fake_model_module
        fake_pixeldit.modules = fake_modules_module

        with mock.patch.dict(
            sys.modules,
            {
                "comfy": fake_comfy,
                "comfy.ldm": fake_ldm,
                "comfy.ldm.pixeldit": fake_pixeldit,
                "comfy.ldm.pixeldit.model": fake_model_module,
                "comfy.ldm.pixeldit.modules": fake_modules_module,
            },
        ):
            pid_runtime._patch_native_pid_attention_dtype_mismatch()
            attn = fake_model_module.MMDiTJointAttention()
            attn.forward(
                torch.zeros((1, 1, 2), dtype=torch.bfloat16),
                torch.zeros((1, 1, 2), dtype=torch.bfloat16),
                torch.zeros((1, 1), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
            )

        self.assertEqual(captured["q_dtype"], torch.bfloat16)
        self.assertEqual(captured["k_dtype"], torch.bfloat16)
        self.assertEqual(captured["v_dtype"], torch.bfloat16)

    def test_patch_native_pid_rotary_attention_upcasts_value_to_query_dtype(self):
        captured = {}

        class FakeRotaryAttention:
            def __init__(self):
                self.num_heads = 1
                self.head_dim = 2
                self.qkv = _FakeLinearOut(torch.tensor([[[1, 2, 3, 4, 5, 6]]], dtype=torch.bfloat16))
                self.q_norm = _FakeNormFloat32()
                self.k_norm = _FakeNormFloat32()
                self.proj = lambda x: x

        fake_model_module = types.ModuleType("comfy.ldm.pixeldit.model")
        fake_modules_module = types.ModuleType("comfy.ldm.pixeldit.modules")

        def fake_apply_rope(q, k, _pos):
            return q, k

        def fake_optimized_attention(q, k, v, *_args, **_kwargs):
            captured["q_dtype"] = q.dtype
            captured["k_dtype"] = k.dtype
            captured["v_dtype"] = v.dtype
            return torch.zeros((1, 1, 1, 2), dtype=q.dtype)

        fake_model_module.MMDiTJointAttention = None
        fake_modules_module.RotaryAttention = FakeRotaryAttention
        fake_modules_module.apply_rope = fake_apply_rope
        fake_modules_module.optimized_attention = fake_optimized_attention

        fake_comfy = types.ModuleType("comfy")
        fake_ldm = types.ModuleType("comfy.ldm")
        fake_pixeldit = types.ModuleType("comfy.ldm.pixeldit")
        fake_comfy.ldm = fake_ldm
        fake_ldm.pixeldit = fake_pixeldit
        fake_pixeldit.model = fake_model_module
        fake_pixeldit.modules = fake_modules_module

        with mock.patch.dict(
            sys.modules,
            {
                "comfy": fake_comfy,
                "comfy.ldm": fake_ldm,
                "comfy.ldm.pixeldit": fake_pixeldit,
                "comfy.ldm.pixeldit.model": fake_model_module,
                "comfy.ldm.pixeldit.modules": fake_modules_module,
            },
        ):
            pid_runtime._patch_native_pid_attention_dtype_mismatch()
            attn = fake_modules_module.RotaryAttention()
            attn.forward(
                torch.zeros((1, 1, 2), dtype=torch.bfloat16),
                torch.zeros((1, 1), dtype=torch.float32),
            )

        self.assertEqual(captured["q_dtype"], torch.bfloat16)
        self.assertEqual(captured["k_dtype"], torch.bfloat16)
        self.assertEqual(captured["v_dtype"], torch.bfloat16)

    def test_decode_samples_processes_native_lq_latent_with_comfy_latent_format(self):
        model = DummyNativeBaseModel()
        model.diffusion_model = mock.Mock()
        model.precision = torch.float32
        model.autocast_dtype = None
        model.fm_trainer = types.SimpleNamespace(timescale=1000.0)
        model.config = pid_runtime._native_pid_config("flux")
        model.net = mock.Mock()
        model.base_model = object()
        seen = {}

        def predict_x0(x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma):
            seen["lq_latent"] = lq_latent
            return torch.ones_like(x)

        model.predict_x0 = predict_x0
        handle = self._register_runtime(model, backbone="flux")
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="cat",
        )

        fake_comfy = types.ModuleType("comfy")
        fake_latent_formats = types.ModuleType("comfy.latent_formats")

        class FakeFlux:
            def process_in(self, latent):
                return latent + 3.0

        fake_latent_formats.Flux = FakeFlux
        fake_latent_formats.SD3 = FakeFlux
        fake_latent_formats.Flux2 = FakeFlux
        fake_comfy.latent_formats = fake_latent_formats

        with mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.latent_formats": fake_latent_formats}):
            pid_runtime._decode_samples(
                handle=handle,
                latent_tensor=torch.zeros((1, 16, 2, 2), dtype=torch.float32),
                pid_prompt=prompt,
                cfg_scale=1.0,
                pid_inference_steps=1,
                seed=0,
                degrade_sigma=0.0,
            )

        self.assertTrue(torch.allclose(seen["lq_latent"], torch.full((1, 16, 2, 2), 3.0)))

    def test_decode_samples_applies_lq_conditioning_boost_to_internal_sigma(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )

        pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=torch.zeros((1, handle.latent_channels, 1, 1), dtype=torch.float32),
            pid_prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=1,
            seed=0,
            degrade_sigma=0.0,
            lq_conditioning_boost=0.5,
        )

        self.assertTrue(torch.allclose(model.last_degrade_sigma, torch.tensor([-0.5])))

    def test_decode_samples_source_denoise_zero_returns_resized_reference_without_prediction(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )
        source = torch.full((1, 8, 8, 3), 0.75, dtype=torch.float32)

        samples = pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            source_image=source,
            source_denoise_strength=0.0,
        )

        self.assertEqual(model.call_count, 0)
        self.assertTrue(torch.allclose(samples.squeeze(2), torch.full((1, 3, 32, 32), 0.5)))

    def test_decode_samples_source_denoise_preserves_padded_source_aspect_ratio(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )
        source = torch.tensor(
            [[[[0.0, 0.0, 0.0]] * 4, [[1.0, 1.0, 1.0]] * 4]],
            dtype=torch.float32,
        )
        geometry = {
            "original_height": 2,
            "original_width": 4,
            "aligned_height": 4,
            "aligned_width": 4,
            "pad_top": 0,
            "pad_bottom": 2,
            "pad_left": 0,
            "pad_right": 0,
            "pid_scale": 4,
        }

        samples = pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            source_image=source,
            source_geometry=geometry,
            source_denoise_strength=0.0,
        )

        padded = pid_runtime._pad_image_tensor(source, geometry)
        expected = torch.nn.functional.interpolate(
            padded.permute(0, 3, 1, 2),
            size=(32, 32),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).mul(2.0).sub(1.0)
        self.assertTrue(torch.allclose(samples.squeeze(2), expected.clamp(-1, 1)))

    def test_decode_samples_resizes_mismatched_reference_to_latent_geometry_before_padding(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )
        source = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        source[:, :, 1:, :] = 1.0
        geometry = {
            "original_height": 8,
            "original_width": 8,
            "aligned_height": 8,
            "aligned_width": 8,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
            "pid_scale": 4,
        }

        samples = pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            source_image=source,
            source_geometry=geometry,
            source_denoise_strength=0.0,
        )

        resized = pid_runtime._resize_image_tensor_to_size(source, 8, 8)
        expected = torch.nn.functional.interpolate(
            resized.permute(0, 3, 1, 2),
            size=(32, 32),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).mul(2.0).sub(1.0)
        self.assertTrue(torch.allclose(samples.squeeze(2), expected.clamp(-1, 1), atol=1e-5))

    def test_decode_samples_source_detail_noise_boost_increases_residual_sde_detail(self):
        class SourceDetailModel(DummyModel):
            def predict_x0(self, x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma):
                self.call_count += 1
                self.last_x = x.detach().clone()
                return torch.zeros_like(x)

        model = SourceDetailModel()
        handle = self._register_runtime(model)
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )
        initial_noise = torch.ones((1, 3, 32, 32), dtype=torch.float32)

        pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=2,
            seed=0,
            degrade_sigma=0.0,
            source_image=torch.full((1, 8, 8, 3), 0.5, dtype=torch.float32),
            source_denoise_strength=0.5,
            source_detail_noise_boost=2.0,
            sde_noise_strength=0.5,
            noise=initial_noise,
            noise_list=[initial_noise],
        )

        self.assertTrue(torch.allclose(model.last_x, torch.full_like(initial_noise, 0.375), atol=1e-5))

    def test_decode_samples_source_denoise_strength_anchors_final_output(self):
        class SourceAnchorModel(DummyModel):
            def predict_x0(self, x, sigma, caption_embs, attention_mask, lq_latent, degrade_sigma):
                self.call_count += 1
                return torch.ones_like(x)

        model = SourceAnchorModel()
        handle = self._register_runtime(model)
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )

        samples = pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=2,
            seed=0,
            degrade_sigma=0.0,
            source_image=torch.full((1, 8, 8, 3), 0.5, dtype=torch.float32),
            source_denoise_strength=0.2,
            sde_noise_strength=0.0,
            noise=torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        )

        self.assertTrue(torch.allclose(samples.squeeze(2), torch.full((1, 3, 32, 32), 0.2), atol=1e-5))

    def test_decode_samples_tiled_sde_casts_float32_prediction_to_bfloat16_state(self):
        model = DummyModel()
        model.precision = torch.bfloat16
        handle = self._register_runtime(model)
        prompt = pid_runtime.PiDPrompt(
            caption_embs=torch.zeros((1, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM), dtype=torch.float32),
            attention_mask=torch.ones((1, pid_runtime.PID_TEXT_TOKEN_COUNT), dtype=torch.int64),
            prompt="",
        )

        samples = pid_runtime._decode_samples(
            handle=handle,
            latent_tensor=torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_prompt=prompt,
            cfg_scale=1.0,
            pid_inference_steps=2,
            seed=0,
            degrade_sigma=0.0,
            sde_noise_strength=0.5,
            use_tiled=True,
            tile_size=32,
            tile_overlap=0,
        )

        self.assertEqual(samples.dtype, torch.float32)
        self.assertTrue(torch.isfinite(samples).all())

    def test_light_model_encodes_text_with_comfy_clip_text_encoder(self):
        clip_text_encoder = DummyComfyClipTextEncoder()
        model = self._make_light_model(text_encoder=clip_text_encoder)
        model.tokenizer = object()
        fake_comfy = types.ModuleType("comfy")
        fake_model_management = types.ModuleType("comfy.model_management")
        fake_model_management.load_models_gpu = mock.Mock()
        fake_comfy.model_management = fake_model_management

        with (
            mock.patch.object(pid_runtime, "_is_comfy_clip_text_encoder", return_value=True),
            mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.model_management": fake_model_management}),
        ):
            caption_embs, attention_mask = model._encode_text_raw(["cat", "mouse"])

        fake_model_management.load_models_gpu.assert_called_once_with([clip_text_encoder.patcher], force_full_load=True)
        self.assertEqual(clip_text_encoder.seen_prompts, ["cat", "mouse"])
        self.assertEqual(tuple(caption_embs.shape), (2, pid_runtime.PID_TEXT_TOKEN_COUNT, pid_runtime.PID_TEXT_EMBED_DIM))
        self.assertEqual(tuple(attention_mask.shape), (2, pid_runtime.PID_TEXT_TOKEN_COUNT))
        self.assertTrue(torch.all(caption_embs[0] == 3.0))
        self.assertTrue(torch.all(caption_embs[1] == 5.0))


class PiDNodeTests(unittest.TestCase):
    def test_loader_node_delegates_to_runtime(self):
        node = nodes.PiDLoadModel()
        sentinel = object()
        with mock.patch.object(nodes, "load_pid_model", return_value=sentinel) as patched:
            result = node.load_model("flux", "2k")

        patched.assert_called_once_with("flux", "2k")
        self.assertEqual(result, (sentinel,))

    def test_loader_node_normalizes_single_variant_backbone(self):
        node = nodes.PiDLoadModel()
        sentinel = object()
        with mock.patch.object(nodes, "load_pid_model", return_value=sentinel) as patched:
            result = node.load_model("sdxl", "2k")

        patched.assert_called_once_with("sdxl", "2kto4k")
        self.assertEqual(result, (sentinel,))

    def test_decode_node_delegates_to_runtime(self):
        node = nodes.PiDDecodeLatent()
        sentinel = torch.zeros((1, 8, 8, 3))
        with mock.patch.object(nodes, "decode_latent", return_value=sentinel) as patched:
            result = node.decode("model", {"samples": torch.zeros((1, 16, 2, 2))}, "cat", "bad", 5.0, 4, 1, 0.0, 0.0, 1.0, 1.0)

        patched.assert_called_once_with(
            handle="model",
            latent=mock.ANY,
            prompt="cat",
            negative_prompt="bad",
            cfg_scale=5.0,
            pid_inference_steps=4,
            seed=1,
            degrade_sigma=0.0,
            lq_conditioning_boost=0.0,
            source_denoise_strength=1.0,
            source_detail_noise_boost=1.0,
            source_image=None,
            pid_prompt=None,
            clip=None,
            unique_id=None,
            sampler="sde",
            scheduler="original",
            sde_noise_strength=1.0,
        )
        self.assertEqual(result["result"], (sentinel,))
        self.assertEqual(result["ui"]["text"], ("8 x 8",))

    def test_encode_image_node_delegates_to_runtime(self):
        node = nodes.PiDEncodeImage()
        sentinel = {"samples": torch.zeros((1, 16, 8, 8))}
        with mock.patch.object(nodes, "encode_image_to_latent", return_value=sentinel) as patched:
            result = node.encode("model", torch.zeros((1, 64, 64, 3)), "512")

        patched.assert_called_once_with("model", mock.ANY, encode_tile_size=512)
        self.assertEqual(result, (sentinel,))

    def test_encode_image_node_uses_auto_tiled_encode_by_default(self):
        inputs = nodes.PiDEncodeImage.INPUT_TYPES()
        self.assertEqual(inputs["required"]["encode_tile_size"][1]["default"], "auto")

        node = nodes.PiDEncodeImage()
        sentinel = {"samples": torch.zeros((1, 16, 8, 8))}
        with mock.patch.object(nodes, "encode_image_to_latent", return_value=sentinel) as patched:
            result = node.encode("model", torch.zeros((1, 64, 64, 3)), "auto")

        patched.assert_called_once_with("model", mock.ANY, encode_tile_size=None)
        self.assertEqual(result, (sentinel,))

    def test_encode_image_node_can_force_full_encode(self):
        node = nodes.PiDEncodeImage()
        sentinel = {"samples": torch.zeros((1, 16, 8, 8))}
        with mock.patch.object(nodes, "encode_image_to_latent", return_value=sentinel) as patched:
            result = node.encode("model", torch.zeros((1, 64, 64, 3)), "disabled")

        patched.assert_called_once_with("model", mock.ANY, encode_tile_size=0)
        self.assertEqual(result, (sentinel,))

    def test_encode_prompt_node_delegates_to_runtime(self):
        node = nodes.PiDEncodePrompt()
        sentinel = {"caption_embs": torch.zeros((1, 4, 8)), "attention_mask": torch.ones((1, 4), dtype=torch.int64)}
        with mock.patch.object(nodes, "encode_prompt", return_value=sentinel) as patched:
            result = node.encode("model", "cat")

        patched.assert_called_once_with("model", "cat", clip=None)
        self.assertEqual(result, (sentinel,))

    def test_decode_tiled_node_delegates_to_runtime(self):
        node = nodes.PiDDecodeLatentTiled()
        sentinel = torch.zeros((1, 64, 64, 3))
        with mock.patch.object(nodes, "decode_latent_tiled", return_value=sentinel) as patched:
            result = node.decode("model", {"samples": torch.zeros((1, 16, 2, 2))}, 256, 64, 2, "cat", "bad", 5.0, 4, 1, 0.0, 0.0, 1.0, 1.0)

        patched.assert_called_once_with(
            handle="model",
            latent=mock.ANY,
            tile_size=256,
            tile_overlap=64,
            tile_batch_size=2,
            prompt="cat",
            negative_prompt="bad",
            cfg_scale=5.0,
            pid_inference_steps=4,
            seed=1,
            degrade_sigma=0.0,
            lq_conditioning_boost=0.0,
            source_denoise_strength=1.0,
            source_detail_noise_boost=1.0,
            source_image=None,
            pid_prompt=None,
            clip=None,
            unique_id=None,
            sampler="sde",
            scheduler="original",
            sde_noise_strength=1.0,
            tiled_sde_noise_boost=1.0,
            seam_refine=False,
            seam_refine_strength=0.25,
        )
        self.assertEqual(result["result"], (sentinel,))
        self.assertEqual(result["ui"]["text"], ("64 x 64",))

    def test_pid_ksampler_node_delegates_to_runtime(self):
        node = nodes.PiDKSampler()
        sentinel = torch.zeros((1, 64, 64, 3))
        with mock.patch.object(nodes, "pid_ksampler", return_value=sentinel) as patched:
            result = node.sample("model", {"samples": torch.zeros((1, 16, 2, 2))}, "cat", "bad", 5.0, 4, 1, 0.0, 0.0, 1.0, 1.0, True, True, 256, 64, 2)

        patched.assert_called_once_with(
            handle="model",
            latent=mock.ANY,
            prompt="cat",
            negative_prompt="bad",
            cfg_scale=5.0,
            pid_inference_steps=4,
            seed=1,
            degrade_sigma=0.0,
            lq_conditioning_boost=0.0,
            source_denoise_strength=1.0,
            source_detail_noise_boost=1.0,
            source_image=None,
            keep_model_loaded_on_gpu=True,
            use_tiled=True,
            tile_size=256,
            tile_overlap=64,
            tile_batch_size=2,
            pid_prompt=None,
            clip=None,
            unique_id=None,
            sampler="sde",
            scheduler="original",
            sde_noise_strength=1.0,
            tiled_sde_noise_boost=1.0,
            seam_refine=False,
            seam_refine_strength=0.25,
        )
        self.assertEqual(result["result"], (sentinel,))
        self.assertEqual(result["ui"]["text"], ("64 x 64",))

    def test_match_colors_adjusts_mean_and_std(self):
        target = torch.ones((1, 16, 16, 3), dtype=torch.float32) * 0.2
        reference = torch.ones((1, 16, 16, 3), dtype=torch.float32) * 0.8

        matched = pid_runtime.match_colors(target, reference)

        self.assertTrue(torch.allclose(matched, reference, atol=1e-3))

    def test_match_colors_wavelet_accepts_reference_at_source_resolution(self):
        target = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 0.4
        reference = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        reference[..., 0] = 0.8
        reference[..., 1] = 0.2
        reference[..., 2] = 0.2

        matched = pid_runtime.match_colors(target, reference, method="wavelet")

        self.assertEqual(tuple(matched.shape), tuple(target.shape))
        self.assertGreater(matched[..., 0].mean().item(), matched[..., 2].mean().item())

    def test_match_colors_wavelet_preserves_target_luminance_structure(self):
        target = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 0.4
        reference = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 0.4
        reference[:, 8:24, 8:24, :] = torch.tensor([0.55, 0.35, 0.35], dtype=torch.float32)

        matched = pid_runtime.match_colors(target, reference, method="wavelet")
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        target_luma = (target * weights).sum(dim=-1)
        matched_luma = (matched * weights).sum(dim=-1)

        self.assertTrue(torch.allclose(matched_luma, target_luma, atol=1e-4))

    def test_node_schemas_hide_legacy_scheduler_options(self):
        for node_cls in (nodes.PiDDecodeLatent, nodes.PiDKSampler, nodes.PiDDecodeLatentTiled):
            required = node_cls.INPUT_TYPES()["required"]
            self.assertNotIn("scheduler", required)
            self.assertEqual(required["color_match"][0], nodes.COLOR_MATCH_OPTIONS)

    def test_resolve_reference_image_uses_latent_reference_when_input_is_disconnected(self):
        reference = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        latent = {
            "samples": torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_runtime.LATENT_REFERENCE_IMAGE_KEY: reference,
        }

        resolved = pid_runtime.resolve_reference_image(latent)

        self.assertTrue(torch.equal(resolved, reference))

    def test_resolve_reference_image_prefers_explicit_input(self):
        stored = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        explicit = torch.ones((1, 8, 8, 3), dtype=torch.float32)
        latent = {
            "samples": torch.zeros((1, 16, 1, 1), dtype=torch.float32),
            pid_runtime.LATENT_REFERENCE_IMAGE_KEY: stored,
        }

        resolved = pid_runtime.resolve_reference_image(latent, explicit)

        self.assertTrue(torch.equal(resolved, explicit))

    def test_match_colors_preserves_highlights(self):
        # target has one pixel at 0.5 (mid) and one at 0.95 (highlight)
        target = torch.tensor([[[[0.5, 0.5, 0.5], [0.95, 0.95, 0.95]]]], dtype=torch.float32)
        # reference with high mean shifts target 0.95 beyond 1.0 without protection
        reference = torch.tensor([[[[0.8, 0.8, 0.8], [0.99, 0.99, 0.99]]]], dtype=torch.float32)

        matched = pid_runtime.match_colors(target, reference)

        # Verify that the highlight pixel is protected from blowout and remains strictly < 1.0
        self.assertLess(matched[0, 0, 1, 0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
