import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from diffusionGS.data.base import BaseDataset as ObjectDataset, BaseDataModuleConfig
from diffusionGS.data.base_scene import BaseDataset as SceneDataset
from diffusionGS.utils.checkpoint import load_denoiser_checkpoint
from diffusionGS.utils.losses import LossComputer, MetricComputer
from diffusionGS.utils.misc import get_device, get_rank
from diffusionGS.utils.callbacks import EarlyEnvironmentSetter
from diffusionGS.models.denoiser.denoiser_utils import SplitData
from eval_scene_result import compute_metrics
from process_data import process_directory, process_torch_file


class RuntimeRegressionTests(unittest.TestCase):
    def test_empty_foreground_mask_has_finite_training_losses(self):
        computer = SimpleNamespace(
            lpips_loss_module=lambda x, y: torch.zeros(len(x), 1, 1, 1),
            ssim_loss_module=lambda x, y: torch.zeros(len(x)),
        )
        images = torch.zeros(1, 4, 3, 16, 16)
        for dtype in (torch.float32, torch.float16):
            with self.subTest(mask_dtype=dtype):
                masks = torch.zeros(1, 4, 1, 16, 16, dtype=dtype)
                losses = LossComputer.forward(computer, images, images, masks, masks, images,
                                              img_aligned_xyz=images, gt_img_aligned_xyz=images)
                self.assertTrue(all(torch.isfinite(loss).all() for loss in losses))
                self.assertEqual(losses[-1].item(), 0)

    def test_distributed_device_uses_local_rank_without_overwriting_global_rank(self):
        trainer = SimpleNamespace(world_size=8, num_devices=4, global_rank=5,
                                  strategy=SimpleNamespace(local_rank=1))
        with patch.dict(os.environ, {"RANK": "5", "LOCAL_RANK": "1"}):
            self.assertEqual(get_rank(), 5)
            self.assertEqual(get_device().index, 1)
            EarlyEnvironmentSetter().setup(trainer, None, "fit")
            self.assertEqual(os.environ["RANK"], "5")
            self.assertEqual(os.environ["WORLD_SIZE"], "8")
            self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "4")

    def test_invalid_batch_does_not_enter_debugger(self):
        cfg = SimpleNamespace(training=SimpleNamespace(num_input_views=4, num_target_views=4))
        batch = {"image": torch.rand(1, 4, 3, 16, 16), "uid": ["example"]}
        with patch("builtins.breakpoint") as debugger:
            with self.assertRaisesRegex(ValueError, "Batch field 'uid'"):
                SplitData(cfg)(batch)
            debugger.assert_not_called()

    def test_stage_two_accepts_training_and_raw_checkpoints(self):
        expected = torch.nn.Linear(3, 2).state_dict()
        formats = [
            expected,
            {"state_dict": {"shape_model." + k: v for k, v in expected.items()}},
            {"state_dict": {"module.shape_model." + k: v for k, v in expected.items()}},
            {"model": {"denoiser." + k: v for k, v in expected.items()}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage1.ckpt"
            for checkpoint in formats:
                with self.subTest(keys=list(checkpoint)):
                    torch.save(checkpoint, path)
                    actual = load_denoiser_checkpoint(path)
                    model = torch.nn.Linear(3, 2)
                    model.load_state_dict(actual, strict=True)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key])

    def test_static_scene_cameras_remain_finite(self):
        dataset = SceneDataset.__new__(SceneDataset)
        result = dataset.preprocess_poses(torch.eye(4).repeat(4, 1, 1))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, torch.eye(4).repeat(4, 1, 1))

    def test_bad_dataset_item_reports_error_without_recursive_retry(self):
        for cls in (ObjectDataset, SceneDataset):
            with self.subTest(dataset=cls.__module__):
                dataset = cls.__new__(cls)
                dataset.uids = ["missing"]
                dataset._get_data = Mock(side_effect=FileNotFoundError("missing.png"))
                with self.assertRaisesRegex(RuntimeError, "Could not load.*missing"):
                    dataset[0]
                dataset._get_data.assert_called_once_with(0)

    def test_object_image_root_does_not_require_trailing_slash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.json").write_text(json.dumps(["object"]))
            for index in range(5):
                folder = root / "object/campos_512_v4" / f"{index:05d}"
                folder.mkdir(parents=True)
                prefix = folder / f"{index:05d}"
                Image.new("RGBA", (16, 16), (100, 150, 200, 255)).save(prefix.with_suffix(".png"))
                prefix.with_suffix(".json").write_text(json.dumps({
                    "x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "origin": [0, 0, 3],
                }))
            cfg = BaseDataModuleConfig(local_dir=str(root), image_dir=str(root),
                                       gen_idxs=[0, 1, 2, 3], all_idxs=list(range(5)),
                                       sel_views=1, training_res=[16, 16], norm_camera=False)
            with patch("diffusionGS.data.base.read_dnormal", return_value=np.ones((16, 16, 1))):
                result = ObjectDataset(cfg, "train")[0]
            self.assertEqual(tuple(result["rgbs"].shape), (5, 3, 16, 16))

    def test_lpips_preserves_single_image_batch(self):
        metric = SimpleNamespace(lpips_loss_module=lambda x, y: torch.zeros(len(x), 1, 1, 1))
        value = MetricComputer.compute_lpips(metric, torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
        self.assertEqual(tuple(value.shape), (1,))

    def test_metrics_accept_different_view_counts_and_weight_each_image(self):
        class ImageMeanMetric(torch.nn.Module):
            def forward(self, target, predicted):
                means = target.mean(dim=(-3, -2, -1))
                return means, means, means

        with tempfile.TemporaryDirectory() as directory:
            for name, views, value in [("one", 1, 0.25), ("two", 2, 0.75)]:
                torch.save({"render_images": torch.zeros(views, 3, 16, 16),
                            "image": torch.full((views, 3, 16, 16), value)}, Path(directory) / f"{name}.pt")
            with patch("eval_scene_result.MetricComputer", ImageMeanMetric):
                result = compute_metrics(directory, chunk=1, device="cpu")
            self.assertEqual(result["num_images"], 3)
            self.assertAlmostEqual(result["psnr"], 7 / 12)

    def test_corrupt_preprocessing_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.torch"
            torch.save([{"key": "broken", "cameras": torch.zeros(1, 18),
                         "images": [torch.tensor([1, 2, 3], dtype=torch.uint8)]}], path)
            with self.assertLogs(level="ERROR"):
                success, _ = process_torch_file(str(path), str(Path(directory) / "processed"))
            self.assertFalse(success)
            self.assertFalse((Path(directory) / "processed/metadata/broken.json").exists())

    def test_empty_preprocessing_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No .torch files"):
                process_directory(directory, str(Path(directory) / "out"), num_processes=1)


if __name__ == "__main__":
    unittest.main()
