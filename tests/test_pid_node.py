import importlib.util
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
        return torch.full((batch, 4, 8), 2.0, dtype=torch.float32), torch.ones((batch, 4), dtype=torch.int64)

    def _get_t_list(self, device, num_steps=None):
        steps = num_steps or 4
        return torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)

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


class PiDRuntimeTests(unittest.TestCase):
    def setUp(self):
        pid_runtime._ACTIVE_RUNTIME = None
        pid_runtime._HANDLE_CACHE.clear()
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

    def test_decode_latent_converts_to_comfy_image(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        latent = {"samples": torch.zeros((1, 16, 2, 3), dtype=torch.float32)}

        image = pid_runtime.decode_latent(
            handle=handle,
            latent=latent,
            prompt="cat",
            pid_inference_steps=4,
            seed=7,
            degrade_sigma=0.25,
        )

        self.assertEqual(tuple(image.shape), (1, 64, 96, 3))
        self.assertTrue(torch.allclose(image, torch.ones_like(image)))
        self.assertEqual(tuple(model.last_lq_latent.shape), (1, 16, 2, 3))
        self.assertEqual(tuple(model.last_caption_embs.shape), (1, 4, 8))
        self.assertTrue(torch.allclose(model.last_degrade_sigma, torch.tensor([0.25])))

    def test_decode_latent_validates_channel_count(self):
        handle = self._register_runtime(DummyModel(), backbone="sd3")
        latent = {"samples": torch.zeros((1, 8, 2, 2), dtype=torch.float32)}

        with self.assertRaises(ValueError):
            pid_runtime.decode_latent(
                handle=handle,
                latent=latent,
                prompt="cat",
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

    def test_encode_image_to_latent_autocorrects_flux_alignment(self):
        model = DummyModel()
        handle = self._register_runtime(model, backbone="flux", latent_channels=16, latent_compression=8)

        latent = pid_runtime.encode_image_to_latent(
            handle=handle,
            image=torch.ones((1, 70, 95, 3), dtype=torch.float32),
        )

        self.assertEqual(tuple(model.last_image.shape), (1, 3, 64, 96))
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 8, 12))

    def test_encode_image_to_latent_autocorrects_flux2_alignment(self):
        model = DummyModel(encode_latent_channels=128, encode_compression=16)
        handle = self._register_runtime(model, backbone="flux2", latent_channels=128, latent_compression=16)

        latent = pid_runtime.encode_image_to_latent(
            handle=handle,
            image=torch.ones((1, 1000, 1537, 3), dtype=torch.float32),
        )

        self.assertEqual(tuple(model.last_image.shape), (1, 3, 1024, 1536))
        self.assertEqual(tuple(latent["samples"].shape), (1, 128, 64, 96))

    def test_decode_latent_tiled_blends_tiles_into_full_image(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        image = pid_runtime.decode_latent_tiled(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
            tile_size=16,
            tile_overlap=8,
            tile_batch_size=1,
            prompt="cat",
            pid_inference_steps=4,
            seed=3,
            degrade_sigma=0.0,
        )

        self.assertEqual(tuple(image.shape), (1, 128, 128, 3))
        self.assertTrue(torch.allclose(image, torch.ones_like(image)))
        self.assertGreaterEqual(model.call_count, 9)

    def test_decode_latent_tiled_batches_same_size_tiles(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        image = pid_runtime.decode_latent_tiled(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
            tile_size=16,
            tile_overlap=8,
            tile_batch_size=4,
            prompt="cat",
            pid_inference_steps=4,
            seed=3,
            degrade_sigma=0.0,
        )

        self.assertEqual(tuple(image.shape), (1, 128, 128, 3))
        self.assertTrue(torch.allclose(image, torch.ones_like(image)))

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

    def test_decode_latent_tiled_validates_tile_alignment(self):
        handle = self._register_runtime(DummyModel(), backbone="flux2", latent_channels=128, latent_compression=16)

        with self.assertRaises(ValueError):
            pid_runtime.decode_latent_tiled(
                handle=handle,
                latent={"samples": torch.zeros((1, 128, 4, 4), dtype=torch.float32)},
                tile_size=200,
                tile_overlap=64,
                tile_batch_size=1,
                prompt="cat",
                pid_inference_steps=4,
                seed=0,
                degrade_sigma=0.0,
            )

    def test_pid_ksampler_dispatches_to_tiled_path(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        image = pid_runtime.pid_ksampler(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 4, 4), dtype=torch.float32)},
            prompt="cat",
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            keep_model_loaded_on_gpu=True,
            use_tiled=True,
            tile_size=16,
            tile_overlap=8,
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
        }

        resized = pid_runtime.resize_latent(latent, 0.5, "bicubic")

        self.assertEqual(tuple(resized["samples"].shape), (1, 16, 4, 4))
        self.assertEqual(tuple(resized["noise_mask"].shape), (1, 4, 4))

    def test_encode_prompt_reuses_cached_embeddings(self):
        model = DummyModel()
        handle = self._register_runtime(model)

        first = pid_runtime.encode_prompt(handle, "cat")
        second = pid_runtime.encode_prompt(handle, "cat")

        self.assertEqual(first["prompt"], "cat")
        self.assertTrue(torch.equal(first["caption_embs"], second["caption_embs"]))
        self.assertEqual(model.text_encoder.to.call_count, 2)

    def test_decode_latent_accepts_preencoded_prompt(self):
        model = DummyModel()
        handle = self._register_runtime(model)
        pid_prompt = {"caption_embs": torch.full((1, 4, 8), 3.0), "attention_mask": torch.ones((1, 4), dtype=torch.int64)}

        image = pid_runtime.decode_latent(
            handle=handle,
            latent={"samples": torch.zeros((1, 16, 2, 2), dtype=torch.float32)},
            prompt="ignored",
            pid_inference_steps=4,
            seed=0,
            degrade_sigma=0.0,
            pid_prompt=pid_prompt,
        )

        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertTrue(torch.all(model.last_caption_embs == 3.0))


class PiDNodeTests(unittest.TestCase):
    def test_loader_node_delegates_to_runtime(self):
        node = nodes.PiDLoadModel()
        sentinel = object()
        with mock.patch.object(nodes, "load_pid_model", return_value=sentinel) as patched:
            result = node.load_model("flux", "2k")

        patched.assert_called_once_with("flux", "2k")
        self.assertEqual(result, (sentinel,))

    def test_decode_node_delegates_to_runtime(self):
        node = nodes.PiDDecodeLatent()
        sentinel = torch.zeros((1, 8, 8, 3))
        with mock.patch.object(nodes, "decode_latent", return_value=sentinel) as patched:
            result = node.decode("model", {"samples": torch.zeros((1, 16, 2, 2))}, "cat", 4, 1, 0.0)

        patched.assert_called_once()
        self.assertEqual(result["result"], (sentinel,))
        self.assertEqual(result["ui"]["text"], ("8 x 8",))

    def test_encode_image_node_delegates_to_runtime(self):
        node = nodes.PiDEncodeImage()
        sentinel = {"samples": torch.zeros((1, 16, 8, 8))}
        with mock.patch.object(nodes, "encode_image_to_latent", return_value=sentinel) as patched:
            result = node.encode("model", torch.zeros((1, 64, 64, 3)))

        patched.assert_called_once()
        self.assertEqual(result, (sentinel,))

    def test_encode_prompt_node_delegates_to_runtime(self):
        node = nodes.PiDEncodePrompt()
        sentinel = {"caption_embs": torch.zeros((1, 4, 8)), "attention_mask": torch.ones((1, 4), dtype=torch.int64)}
        with mock.patch.object(nodes, "encode_prompt", return_value=sentinel) as patched:
            result = node.encode("model", "cat")

        patched.assert_called_once_with("model", "cat")
        self.assertEqual(result, (sentinel,))

    def test_decode_tiled_node_delegates_to_runtime(self):
        node = nodes.PiDDecodeLatentTiled()
        sentinel = torch.zeros((1, 64, 64, 3))
        with mock.patch.object(nodes, "decode_latent_tiled", return_value=sentinel) as patched:
            result = node.decode("model", {"samples": torch.zeros((1, 16, 2, 2))}, 256, 64, 2, "cat", 4, 1, 0.0)

        patched.assert_called_once()
        self.assertEqual(result["result"], (sentinel,))
        self.assertEqual(result["ui"]["text"], ("64 x 64",))

    def test_pid_ksampler_node_delegates_to_runtime(self):
        node = nodes.PiDKSampler()
        sentinel = torch.zeros((1, 64, 64, 3))
        with mock.patch.object(nodes, "pid_ksampler", return_value=sentinel) as patched:
            result = node.sample("model", {"samples": torch.zeros((1, 16, 2, 2))}, "cat", 4, 1, 0.0, True, True, 256, 64, 2)

        patched.assert_called_once()
        self.assertEqual(result["result"], (sentinel,))
        self.assertEqual(result["ui"]["text"], ("64 x 64",))


if __name__ == "__main__":
    unittest.main()
