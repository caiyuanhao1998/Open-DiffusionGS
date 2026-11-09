"""Shared ZeroGPU backend for the Object and Scene Hugging Face Spaces."""
import os
from pathlib import Path
from types import SimpleNamespace

# rembg imports Numba. Its default TBB pool cannot safely fork from a
# Gradio worker thread when ZeroGPU starts the GPU process.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

# ZeroGPU must install its CUDA emulation before torch is imported.
import spaces
import torch
import gradio as gr
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from object_demo import ObjectDemo, build_demo as build_object_demo
from scene_demo import SceneDemo, build_demo as build_scene_demo
from diffusionGS.models.denoiser.denoiser import DGSDenoiser
from diffusionGS.models.diffusion import create_diffusion
from diffusionGS.pipeline_scene import DiffusionGSScenePipeline
from diffusionGS.pipline_obj import DiffusionGSPipeline

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("DIFFUSIONGS_OUTPUT_DIR", ROOT / "outputs")).resolve()
MODEL_REPO = "CaiYuanhao/DiffusionGS"
MODEL_REVISION = "6ad55569352ab32bd7ca52ba1c5f5cf89065b2d6"


def use_torch_attention(model):
    """Use PyTorch SDPA across the different GPUs allocated by ZeroGPU."""
    for module in model.modules():
        if hasattr(module, "use_flashatt_v2"):
            module.use_flashatt_v2 = False


def load_object_pipeline(model_dir):
    # Inference only needs the denoiser, diffusion schedule and camera template.
    # Avoid allocating training losses and downloading their VGG checkpoint.
    config = OmegaConf.load(model_dir / "obj_configs.yaml")
    model = DGSDenoiser(OmegaConf.to_container(config.system.shape_model))
    checkpoint = torch.load(model_dir / "obj_ckpt_512.ckpt", map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({key.removeprefix("shape_model."): value for key, value in state.items()
                           if key.startswith("shape_model.")}, strict=True)
    del checkpoint, state
    use_torch_attention(model)
    model = model.to(device="cuda:0", dtype=torch.float16).eval()
    system = SimpleNamespace(
        shape_model=model, device=torch.device("cuda:0"), dtype=torch.float16,
        diffusion_inference=create_diffusion(str(config.system.num_inference_steps), predict_xstart=True),
        cam_template=torch.load(model_dir / "camera_template.pt", map_location="cpu", weights_only=False),
    )
    return DiffusionGSPipeline("cuda:0", config, system)


torch.set_num_threads(1)
model_dir = Path(snapshot_download(
    MODEL_REPO, revision=MODEL_REVISION,
    allow_patterns=["obj_ckpt_512.ckpt", "obj_configs.yaml", "camera_template.pt",
                    "scene_ckpt_256.ckpt", "scene_ckpt_512.ckpt"],
))
object_service = ObjectDemo(output_dir=OUTPUT_ROOT / "object")
object_service.pipeline = load_object_pipeline(model_dir)
scene_pipelines = {}
for resolution in (256, 512):
    pipeline = DiffusionGSScenePipeline.from_pretrained(model_dir, resolution=resolution)
    use_torch_attention(pipeline.model)
    scene_pipelines[resolution] = pipeline
scene_service = SceneDemo(output_dir=OUTPUT_ROOT / "scene")

_generate_object = object_service.generate
_extract_mesh = object_service.extract_mesh
_generate_scene = scene_service.generate


@spaces.GPU(duration=40)
def generate_object(image, seed, foreground_ratio, force_remove_background):
    print("Starting object generation on ZeroGPU", flush=True)
    return _generate_object(image, seed, foreground_ratio, force_remove_background)


@spaces.GPU(duration=40)
def extract_mesh(result_directory):
    print("Starting mesh extraction on ZeroGPU", flush=True)
    return _extract_mesh(result_directory)


def scene_gpu_duration(image, preset, translation_scale, rotation_degrees, fov_degrees,
                       duration_seconds, resolution, steps, seed, fps):
    return (20 + (20 if int(resolution) == 512 else 0)
            + max(0, int(steps) - 30) // 2
            + max(0, int(duration_seconds) * int(fps) // 2 - 120) // 10)


@spaces.GPU(duration=scene_gpu_duration)
def generate_scene(image, preset, translation_scale, rotation_degrees, fov_degrees,
                   duration_seconds, resolution, steps, seed, fps):
    print("Starting scene generation on ZeroGPU", flush=True)
    scene_service.pipeline = scene_pipelines[int(resolution)]
    return _generate_scene(image, preset, translation_scale, rotation_degrees, fov_degrees,
                           duration_seconds, resolution, steps, seed, fps)


object_service.generate = generate_object
object_service.extract_mesh = extract_mesh
scene_service.generate = generate_scene
demo = build_object_demo(service=object_service)
with demo:
    gr.Navbar(main_page_name="Object Generation")
scene_demo = build_scene_demo(service=scene_service)
with demo.route("Scene Generation", "/scene"):
    gr.Navbar(main_page_name="Object Generation")
    scene_demo.render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
                max_file_size="20mb", allowed_paths=[str(OUTPUT_ROOT)], ssr_mode=False)
