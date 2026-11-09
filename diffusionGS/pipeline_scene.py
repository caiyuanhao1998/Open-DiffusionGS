"""Single-image scene generation using the released RealEstate10K models."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageOps
import torch
from easydict import EasyDict
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from diffusionGS.models.denoiser.denoiser_scene import DGSDenoiser
from diffusionGS.models.diffusion import create_diffusion
from diffusionGS.models.gsrenderer.gs_core import GaussianModel
from diffusionGS.systems.utils import TransformInput


@dataclass
class ScenePipelineOutput:
    gaussians: GaussianModel
    render_images: torch.Tensor
    input_image: Image.Image


class DiffusionGSScenePipeline:
    def __init__(self, model, resolution, device, dtype):
        self.model = model
        self.resolution = resolution
        self.device = torch.device(device)
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path="CaiYuanhao/DiffusionGS",
                        resolution=256, device="cuda:0", torch_dtype=torch.float16,
                        cache_dir=None):
        if resolution not in (256, 512):
            raise ValueError("Scene checkpoints support resolutions 256 and 512.")
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Scene generation requires a CUDA GPU.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        filename = f"scene_ckpt_{resolution}.ckpt"
        source = Path(pretrained_model_name_or_path)
        if source.is_dir():
            checkpoint = source / filename
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing scene checkpoint: {checkpoint}")
        else:
            checkpoint = hf_hub_download(str(pretrained_model_name_or_path), filename,
                                         cache_dir=cache_dir)
        suffix = "_512" if resolution == 512 else ""
        config_path = Path(__file__).parent / "configs" / f"diffusionGS_scene_eval{suffix}.yaml"
        config = OmegaConf.to_container(OmegaConf.load(config_path).system.shape_model)
        # The evaluation YAML has a training checkpoint placeholder. Load once here.
        config["pretrained_model_name_or_path"] = ""
        with torch.cuda.device(device):
            model = DGSDenoiser(config)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state.get("model", state))
            if any(key.startswith("shape_model.") for key in state):
                state = {key[len("shape_model."):]: value for key, value in state.items()
                         if key.startswith("shape_model.")}
            model.load_state_dict(state, strict=True)
            del state
            model = model.to(device=device, dtype=torch_dtype).eval()
        return cls(model, resolution, device, torch_dtype)

    @torch.inference_mode()
    def __call__(self, image, c2ws, intrinsics, seed=62, num_inference_steps=30,
                 progress_callback: Optional[Callable] = None):
        """Generate one Gaussian scene from an image and four OpenCV c2w poses.

        Intrinsics are (fx, fy, cx, cy) in pixels at the selected resolution.
        Scene images retain their backgrounds and are center cropped to square.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (self.resolution, self.resolution), Image.Resampling.LANCZOS)
        c2ws = np.asarray(c2ws, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if c2ws.shape != (4, 4, 4) or intrinsics.shape != (4, 4):
            raise ValueError("Provide exactly four 4x4 camera-to-world poses and four intrinsics.")
        if not np.isfinite(c2ws).all() or not np.isfinite(intrinsics).all():
            raise ValueError("Camera parameters must be finite.")
        if (intrinsics[:, :2] <= 0).any():
            raise ValueError("Focal lengths must be positive.")
        if not np.allclose(c2ws[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("Camera matrices must have homogeneous last row [0, 0, 0, 1].")
        rotations = c2ws[:, :3, :3]
        if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)):
            raise ValueError("Camera rotations must be proper orthonormal matrices.")
        if int(seed) != seed or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 2**32 - 1.")
        if int(num_inference_steps) != num_inference_steps or not 2 <= num_inference_steps <= 100:
            raise ValueError("Inference steps must be an integer between 2 and 100.")

        with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(int(seed))
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
            pixels = pixels.to(self.device, dtype=self.dtype)[None, None] / 255
            noise = torch.randn(1, 3, 3, self.resolution, self.resolution,
                                device=self.device, dtype=self.dtype)
            cameras = torch.from_numpy(c2ws).to(self.device)[None]
            calibration = torch.from_numpy(intrinsics).to(self.device)[None]
            ray_o, ray_d = TransformInput(torch.cat([pixels, noise], dim=1), cameras, calibration)
            batch = EasyDict(image=pixels, image_noisy=noise, c2w=cameras,
                             fxfycxcy=calibration, ray_o=ray_o.to(self.dtype), ray_d=ray_d.to(self.dtype))
            diffusion = create_diffusion(str(int(num_inference_steps)), predict_xstart=True)
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                for step, output in enumerate(diffusion.p_sample_loop_progressive(
                        self.model, noise.shape, batch, clip_denoised=False,
                        progress=False, device=self.device)):
                    if progress_callback is not None:
                        progress_callback(step + 1, diffusion.num_timesteps)
            gaussians = output["denoiser_output_dict"]["pred_gaussians"][0]
            rendered = output["denoiser_output_dict"]["render_images"][0]
            if not torch.isfinite(rendered).all() or not torch.isfinite(gaussians.get_xyz).all():
                raise RuntimeError("Scene generation produced nonfinite values.")
            gaussians = gaussians.prune(opacity_thres=0.05)
            if len(gaussians.get_xyz) == 0:
                raise RuntimeError("No Gaussians survived opacity filtering. Try another image or seed.")
            return ScenePipelineOutput(gaussians, rendered.detach().float().cpu(), image)
