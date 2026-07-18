# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import checkpoint as checkpoint_module  # noqa: E402
import model as model_module  # noqa: E402


class EfficientNetModelTests(unittest.TestCase):
    def test_offline_builder_returns_one_logit_efficientnet_b0(self):
        net = model_module.build_model(pretrained=False).eval()

        self.assertEqual(net.classifier[1].out_features, 1)
        with torch.no_grad():
            logits = net(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 1))

    def test_pretrained_constructor_error_is_not_silently_downgraded(self):
        with mock.patch(
            "torchvision.models.efficientnet_b0",
            side_effect=RuntimeError("weights download failed"),
        ) as constructor:
            with self.assertRaisesRegex(RuntimeError, "weights download failed"):
                model_module.build_model(pretrained=True)

        self.assertEqual(constructor.call_count, 1)


class DeviceSelectionTests(unittest.TestCase):
    def test_require_gpu_raises_when_directml_and_cuda_are_unavailable(self):
        with mock.patch.object(model_module, "probe_directml", return_value=False), \
             mock.patch.object(model_module.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                model_module.get_device(require_gpu=True, verbose=False)

    def test_cuda_is_selected_only_after_directml_probe_fails(self):
        with mock.patch.object(model_module, "probe_directml", return_value=False) as probe, \
             mock.patch.object(model_module.torch.cuda, "is_available", return_value=True):
            device = model_module.get_device(verbose=False)

        probe.assert_called_once_with(verbose=False)
        self.assertEqual(device, torch.device("cuda"))

    def test_directml_probe_performs_real_forward_and_backward(self):
        class TinyOneLogitModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Linear(3, 1)

            def forward(self, x):
                return self.head(self.pool(x).flatten(1))

        net = TinyOneLogitModel()
        fake_directml = types.SimpleNamespace(
            device_count=lambda: 1,
            device=lambda: torch.device("cpu"),
        )
        with mock.patch.dict(sys.modules, {"torch_directml": fake_directml}), \
             mock.patch.object(model_module, "build_model", return_value=net):
            available = model_module.probe_directml(verbose=False)

        self.assertTrue(available)
        self.assertTrue(all(p.grad is not None for p in net.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))


class PortableCheckpointTests(unittest.TestCase):
    def _save_checkpoint(self, path: Path, model: nn.Module) -> None:
        checkpoint_module.save_checkpoint(
            path,
            model,
            img_size=64,
            class_names=["benign", "malignant"],
            threshold=0.42,
            best_epoch=3,
            best_metric=0.81,
            best_metric_name="val_case_auc",
            split_seed=42,
            fold=0,
            history={"train_loss": [0.7, 0.5]},
            manifest_fp="manifest-sha256",
        )

    def test_checkpoint_has_cpu_state_and_round_trips_identical_cpu_logits(self):
        torch.manual_seed(7)
        original = model_module.build_model(pretrained=False).eval()
        sample = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = original(sample)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            self._save_checkpoint(path, original)
            raw = torch.load(path, map_location="cpu", weights_only=True)
            restored, metadata = checkpoint_module.load_for_inference(
                path, device=torch.device("cpu")
            )

            self.assertEqual(raw["arch"], "efficientnet_b0")
            self.assertEqual(raw["num_outputs"], 1)
            self.assertTrue(
                all(t.device.type == "cpu" for t in raw["model_state"].values())
            )
            self.assertEqual(metadata["img_size"], 64)
            with torch.no_grad():
                actual = restored(sample)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_loader_requests_weights_only_when_supported(self):
        model = model_module.build_model(pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            self._save_checkpoint(path, model)
            real_load = torch.load
            calls = []

            def recording_load(*args, **kwargs):
                calls.append(dict(kwargs))
                return real_load(*args, **kwargs)

            with mock.patch.object(
                checkpoint_module.torch, "load", side_effect=recording_load
            ):
                checkpoint_module.load_for_inference(path)

        self.assertIs(calls[0].get("weights_only"), True)

    def test_loader_supports_torch_versions_without_weights_only_argument(self):
        model = model_module.build_model(pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            self._save_checkpoint(path, model)
            real_load = torch.load

            def legacy_load(*args, **kwargs):
                if "weights_only" in kwargs:
                    raise TypeError("unexpected keyword argument 'weights_only'")
                return real_load(*args, **kwargs)

            with mock.patch.object(
                checkpoint_module.torch, "load", side_effect=legacy_load
            ):
                restored, _ = checkpoint_module.load_for_inference(path)

        self.assertIsInstance(restored, nn.Module)

    def test_loader_rejects_unsupported_reconstruction_metadata(self):
        model = model_module.build_model(pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            self._save_checkpoint(path, model)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["arch"] = "unknown_architecture"
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "unknown_architecture"):
                checkpoint_module.load_for_inference(path)


if __name__ == "__main__":
    unittest.main()
