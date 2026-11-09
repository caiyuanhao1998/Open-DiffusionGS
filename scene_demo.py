"""Gradio demo for single-image 3D scene generation and camera control."""
import argparse
import gc
import json
import math
from pathlib import Path
import tempfile
import threading
import time

import gradio as gr
from PIL import Image

from camera_trajectories import TRAJECTORY_PRESETS, build_trajectory, trajectory_figure


EXAMPLE_IMAGE = Path(__file__).parent / "extra_files/test_cases/scene_room.png"
EXAMPLE_DIRECTORY = Path(__file__).parent / "extra_files/scene_examples"
SCENE_EXAMPLES = [
    ("Living room", EXAMPLE_IMAGE),
    ("RealEstate10K open kitchen", EXAMPLE_DIRECTORY / "re10k_0bf152ef84195293.png"),
    ("RealEstate10K home office", EXAMPLE_DIRECTORY / "re10k_0f47577ab3441480.png"),
    ("RealEstate10K living room", EXAMPLE_DIRECTORY / "re10k_2c80f9eb0d3b2bb4.png"),
    ("RealEstate10K bedroom", EXAMPLE_DIRECTORY / "re10k_3c35b868a8ec3433.png"),
    ("RealEstate10K kitchen and dining room", EXAMPLE_DIRECTORY / "re10k_4a2d6753676df096.png"),
]


def video_frame_count(duration_seconds, fps):
    if (not math.isfinite(duration_seconds) or int(duration_seconds) != duration_seconds
            or not 1 <= duration_seconds <= 10):
        raise ValueError("Video duration must be a whole number between 1 and 10 seconds.")
    if not math.isfinite(fps) or int(fps) != fps or not 1 <= fps <= 60:
        raise ValueError("Video FPS must be an integer between 1 and 60.")
    return int(duration_seconds) * int(fps)


def preview_trajectory(preset, translation_scale, rotation_degrees, fov_degrees,
                       duration_seconds, resolution, fps=24):
    trajectory = build_trajectory(preset, translation_scale, rotation_degrees,
                                  fov_degrees, video_frame_count(duration_seconds, fps), int(resolution))
    return trajectory_figure(trajectory)


class SceneDemo:
    def __init__(self, model_dir="CaiYuanhao/DiffusionGS", cache_dir=None,
                 output_dir="outputs/scene_demo", device="cuda:0"):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir).resolve()
        self.device = device
        self.pipeline = None
        self.lock = threading.Lock()

    def generate(self, image, preset, translation_scale, rotation_degrees,
                 fov_degrees, duration_seconds, resolution, steps, seed, fps,
                 progress=gr.Progress()):
        if image is None:
            raise gr.Error("Upload a scene image first.")
        trajectory = build_trajectory(preset, translation_scale, rotation_degrees,
                                      fov_degrees, video_frame_count(duration_seconds, fps), int(resolution))

        import imageio.v2 as imageio
        import torch
        from diffusionGS.pipeline_scene import DiffusionGSScenePipeline
        from diffusionGS.models.gsrenderer.gs_core import render_opencv_cam

        with self.lock:
            if self.pipeline is None or self.pipeline.resolution != int(resolution):
                # Retain only one checkpoint on the GPU when switching resolution.
                self.pipeline = None
                gc.collect()
                torch.cuda.empty_cache()
                progress(0, desc=f"Loading {int(resolution)}px scene model")
                self.pipeline = DiffusionGSScenePipeline.from_pretrained(
                    self.model_dir, resolution=int(resolution), device=self.device,
                    cache_dir=self.cache_dir,
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="scene-", dir=self.output_dir))
            start = time.perf_counter()
            anchors = trajectory.keyframe_indices

            def update_progress(step, total):
                progress(0.7 * step / total, desc=f"Generating scene: {step}/{total}")

            output = self.pipeline(
                image, trajectory.c2ws[anchors], trajectory.intrinsics[anchors],
                seed=seed, num_inference_steps=steps,
                progress_callback=update_progress,
            )
            generation_seconds = time.perf_counter() - start
            output.input_image.save(directory / "input.png")
            output.gaussians.save_ply(str(directory / "scene.ply"))
            video_path = directory / "scene.mp4"
            # Render and encode one frame at a time instead of keeping a whole
            # 512px video in GPU memory. These are the exact previewed cameras.
            with torch.inference_mode(), torch.cuda.device(self.pipeline.device), imageio.get_writer(
                    str(video_path), fps=int(fps), codec="libx264",
                    pixelformat="yuv420p", macro_block_size=1) as writer:
                cameras = torch.from_numpy(trajectory.c2ws).to(self.pipeline.device)
                intrinsics = torch.from_numpy(trajectory.intrinsics).to(self.pipeline.device)
                for index in range(len(cameras)):
                    frame = render_opencv_cam(
                        output.gaussians, int(resolution), int(resolution),
                        cameras[index], intrinsics[index],
                    )["render"]
                    if not torch.isfinite(frame).all():
                        raise RuntimeError("Video rendering produced nonfinite pixels.")
                    frame = frame.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
                    writer.append_data(frame)
                    progress(0.7 + 0.3 * (index + 1) / len(cameras),
                             desc=f"Rendering video: {index + 1}/{len(cameras)}")

            camera_data = trajectory.to_dict()
            camera_data.update(seed=int(seed), inference_steps=int(steps), fps=int(fps),
                               duration_seconds=int(duration_seconds),
                               translation_scale=float(translation_scale),
                               rotation_degrees=float(rotation_degrees),
                               fov_degrees=float(fov_degrees))
            camera_path = directory / "cameras.json"
            camera_path.write_text(json.dumps(camera_data, indent=2))
            gallery = []
            for index, tensor in zip(anchors, output.render_images):
                pixels = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
                gallery.append((Image.fromarray(pixels),
                                f"{'Input camera' if index == 0 else 'Generated camera'} · frame {index}"))
            elapsed = time.perf_counter() - start
            count = len(output.gaussians.get_xyz)
            (directory / "result.json").write_text(json.dumps({
                "status": "passed", "generation_seconds": generation_seconds,
                "total_seconds": elapsed, "gaussian_count": count,
                "resolution": int(resolution), "frames": len(cameras),
                "fps": int(fps), "duration_seconds": int(duration_seconds),
                "seed": int(seed), "inference_steps": int(steps), "preset": preset,
            }, indent=2))
            status = (f"**Done.** {int(resolution)}×{int(resolution)} · {int(duration_seconds)}s · "
                      f"{len(cameras)} frames at {int(fps)} FPS · "
                      f"{count:,} Gaussians · generation {generation_seconds:.1f}s · total {elapsed:.1f}s")
            return (str(video_path), gallery, str(directory / "scene.ply"),
                    str(camera_path), trajectory_figure(trajectory), status)


def build_demo(model_dir="CaiYuanhao/DiffusionGS", cache_dir=None,
               output_dir="outputs/scene_demo", device="cuda:0"):
    service = SceneDemo(model_dir, cache_dir, output_dir, device)
    with gr.Blocks(title="DiffusionGS · Scene Generation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# DiffusionGS · Scene Generation\n"
                    "Turn a scene image into a 3D Gaussian scene and a camera-controlled video.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(value=str(EXAMPLE_IMAGE), type="pil", image_mode="RGB",
                                 label="1 · Scene image", height=280, elem_id="scene-input")
                gr.Markdown("Images are center cropped to square. Keep the scene background.")
                gr.Examples(
                    examples=[[str(path)] for _, path in SCENE_EXAMPLES],
                    inputs=[image], examples_per_page=len(SCENE_EXAMPLES),
                    label="Scene examples · click to select", cache_examples=False,
                    api_name=False, elem_id="scene-examples",
                )
                preset = gr.Dropdown(list(TRAJECTORY_PRESETS), value="Spiral forward",
                                     label="2 · Camera trajectory")
                translation = gr.Slider(0, 2, value=1, step=0.05, label="Translation scale")
                rotation = gr.Slider(0, 45, value=15, step=1, label="Rotation (degrees)")
                fov = gr.Slider(30, 90, value=60, step=1, label="Horizontal field of view (degrees)")
                gr.Markdown("The default spiral makes three small circles while moving forward. "
                            "Translation scale controls camera travel; rotation controls pan, tilt, "
                            "orbit and curved paths. Small motions usually give better scene coverage.")
                with gr.Accordion("Generation settings", open=True):
                    resolution = gr.Radio([256, 512], value=256, label="Resolution")
                    steps = gr.Slider(10, 100, value=30, step=1, label="Diffusion steps")
                    seed = gr.Number(value=62, precision=0, minimum=0,
                                     maximum=2**32 - 1, label="Seed")
                    duration = gr.Slider(1, 10, value=5, step=1, label="Video duration (seconds)")
                    fps = gr.Slider(8, 30, value=24, step=1, label="Video FPS")
                generate = gr.Button("Generate scene & video", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(value=preview_trajectory("Spiral forward", 1, 15, 60, 5, 256, 24),
                               label="Camera trajectory · drag to rotate / scroll to zoom")
                gr.Markdown("**Pink:** input camera · **Gold:** three generated keyframes · "
                            "**Green arrows:** travel direction. Camera pyramids and arrows show viewing direction. "
                            "The preview and video use the same poses.")
                preview = gr.Button("Preview trajectory")
                video = gr.Video(label="3 · Generated scene video", format="mp4")
                status = gr.Markdown("Choose an image and a camera path, then generate.")
                gallery = gr.Gallery(label="Generated keyframe views", columns=4, height=230)
                with gr.Row():
                    ply = gr.DownloadButton("Download point cloud (.ply)",
                                            variant="primary", interactive=False)
                    camera_file = gr.File(label="Download camera trajectory (.json)")
                gr.Markdown("The point cloud download becomes available after generation.")
        gr.Markdown("Trajectory visualization and camera-control interaction inspired by "
                    "[CameraCtrl](https://github.com/hehao13/CameraCtrl). "
                    "All scene generation uses the released DiffusionGS scene checkpoints.")
        camera_inputs = [preset, translation, rotation, fov, duration, resolution, fps]
        preview.click(preview_trajectory, camera_inputs, plot,
                      api_name="preview_trajectory", queue=False)
        for control in camera_inputs:
            control.change(preview_trajectory, camera_inputs, plot, queue=False, api_name=False)
        generation = generate.click(
            service.generate,
            [image, preset, translation, rotation, fov, duration, resolution, steps, seed, fps],
            [video, gallery, ply, camera_file, plot, status],
            api_name="generate_scene", concurrency_limit=1,
        )
        generation.success(lambda: gr.update(interactive=True), outputs=ply,
                           queue=False, api_name=False)
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="CaiYuanhao/DiffusionGS",
                        help="Hugging Face repo or local directory with scene_ckpt_256/512.ckpt")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face checkpoint cache")
    parser.add_argument("--output-dir", default="outputs/scene_demo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_demo(args.model_dir, args.cache_dir, args.output_dir, args.device)
    demo.launch(server_name=args.server_name, server_port=args.server_port,
                share=args.share, show_error=True,
                allowed_paths=[str(Path(args.output_dir).resolve())])


if __name__ == "__main__":
    main()
