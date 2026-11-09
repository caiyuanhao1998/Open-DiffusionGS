"""Gradio demo for single-image 3D object generation and optional mesh export."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
import time

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import plotly.graph_objects as go


EXAMPLE_DIRECTORY = Path(__file__).parent / "extra_files/test_cases"
OBJECT_EXAMPLES = [EXAMPLE_DIRECTORY / name for name in (
    "dp.png", "avocado.png", "doctor.png", "jige.png", "link.png", "monster.png", "popmart.png",
)]


def point_cloud_figure(xyz, colors, max_points=5000):
    indices = np.linspace(0, len(xyz) - 1, min(len(xyz), max_points), dtype=int)
    positions, rgb = xyz[indices], colors[indices]
    figure = go.Figure(go.Scatter3d(
        x=positions[:, 0].tolist(), y=positions[:, 1].tolist(), z=positions[:, 2].tolist(),
        mode="markers", marker=dict(size=2, color=[f"rgb({r},{g},{b})" for r, g, b in rgb]),
        hoverinfo="skip", showlegend=False,
    ))
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    radius = max(float(np.ptp(xyz, axis=0).max()), 0.1) * 0.65
    axes = [dict(title=label, range=[float(value - radius), float(value + radius)])
            for value, label in zip(center, ["X", "Y", "Z"])]
    figure.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0),
                         scene=dict(aspectmode="cube", xaxis=axes[0], yaxis=axes[1], zaxis=axes[2],
                                    camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0))))
    return figure


class ObjectDemo:
    def __init__(self, model_dir="CaiYuanhao/DiffusionGS", output_dir="outputs/object_demo",
                 device="cuda:0"):
        self.model_dir = model_dir
        self.output_dir = Path(output_dir).resolve()
        self.device = device
        self.pipeline = None
        self.lock = threading.Lock()

    def generate(self, image, seed, foreground_ratio, force_remove_background,
                 progress=gr.Progress()):
        if image is None:
            raise gr.Error("Upload an object image first.")
        if int(seed) != seed or not 0 <= seed <= 2**32 - 1:
            raise gr.Error("Seed must be an integer between 0 and 2**32 - 1.")
        if not 0 < foreground_ratio <= 1:
            raise gr.Error("Foreground ratio must be between 0 and 1.")
        image = ImageOps.exif_transpose(image).convert("RGBA")
        if image.getextrema()[3][1] == 0:
            raise gr.Error("The input image is fully transparent.")

        import torch
        from torchvision.utils import save_image
        from diffusionGS.pipline_obj import DiffusionGSPipeline
        from diffusionGS.models.gsrenderer.gs_core import SH2RGB

        if not torch.cuda.is_available() or torch.device(self.device).type != "cuda":
            raise gr.Error("Object generation requires a CUDA GPU.")
        with self.lock, torch.cuda.device(self.device):
            if self.pipeline is None:
                progress(0, desc="Loading object model")
                self.pipeline = DiffusionGSPipeline.from_pretrained(
                    self.model_dir, device=self.device, torch_dtype=torch.float16,
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="object-", dir=self.output_dir))
            image.save(directory / "input.png")
            start = time.perf_counter()

            def update_progress(step, total):
                progress(0.1 + 0.75 * step / total, desc=f"Generating object: {step}/{total}")

            output = self.pipeline(
                image, seed=int(seed), foreground_ratio=float(foreground_ratio),
                force_remove_background=force_remove_background,
                extract_mesh=False, progress_callback=update_progress,
            )
            xyz = output.gaussians.get_xyz.detach().float().cpu().numpy()
            if len(xyz) == 0 or not np.isfinite(xyz).all() or not torch.isfinite(output.render_images).all():
                raise RuntimeError("Generation produced empty or nonfinite outputs.")
            colors = (SH2RGB(output.gaussians._features_dc[:, 0]).detach().float()
                      .clamp(0, 1).mul(255).byte().cpu().numpy())
            progress(0.9, desc="Saving point cloud and rendered views")
            output.gaussians.save_ply(str(directory / "test.ply"))
            save_image(output.render_images, str(directory / "test.png"))
            views = []
            for i, tensor in enumerate(output.render_images):
                pixels = tensor.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
                views.append((Image.fromarray(pixels), f"View {i + 1}"))
            elapsed = time.perf_counter() - start
            (directory / "result.json").write_text(json.dumps({
                "status": "passed", "seed": int(seed), "foreground_ratio": float(foreground_ratio),
                "force_remove_background": bool(force_remove_background),
                "generation_seconds": elapsed, "gaussian_count": len(xyz),
                "resolution": 512, "mesh_exported": False,
            }, indent=2))
            progress(1, desc="Object ready")
            return (views, point_cloud_figure(xyz, colors), str(directory / "test.ply"),
                    str(directory / "test.png"), None, None, str(directory),
                    f"**Done.** {len(xyz):,} Gaussians · {elapsed:.1f}s. Mesh export is available below.",
                    "Mesh has not been extracted for this object.")

    def extract_mesh(self, result_directory, progress=gr.Progress()):
        if not result_directory:
            raise gr.Error("Generate an object first.")
        directory = Path(result_directory).resolve()
        if not directory.is_relative_to(self.output_dir) or not (directory / "test.ply").is_file():
            raise gr.Error("The generated object is unavailable. Please generate it again.")
        destination = directory / "test.obj"
        with self.lock:
            if not destination.is_file() or destination.stat().st_size == 0:
                import torch
                from diffusionGS.models.gsrenderer.gs_core import GaussianModel
                progress(0.1, desc="Loading generated Gaussians for mesh extraction")
                start = time.perf_counter()
                with torch.inference_mode(), torch.cuda.device(self.device):
                    gaussians = GaussianModel(sh_degree=0)
                    gaussians.load_ply(str(directory / "test.ply"))
                    gaussians.to(self.device)
                    progress(0.2, desc="Extracting mesh; this can take several minutes")
                    mesh = gaussians.extract_mesh()
                    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                        raise RuntimeError("Mesh extraction produced invalid geometry.")
                    mesh.export(destination)
                report = json.loads((directory / "result.json").read_text())
                report.update(mesh_exported=True, mesh_seconds=time.perf_counter() - start,
                              mesh_vertices=len(mesh.vertices), mesh_faces=len(mesh.faces))
                (directory / "result.json").write_text(json.dumps(report, indent=2))
            preview_path = directory / "mesh_preview.obj"
            if not preview_path.is_file() or preview_path.stat().st_mtime < destination.stat().st_mtime:
                import trimesh
                preview_mesh = trimesh.load(destination, force="mesh", process=False)
                # The web viewer is Y-up; retain the model's original Z-up OBJ
                # for download and rotate only the separate preview asset.
                preview_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
                preview_mesh.export(preview_path)
            progress(1, desc="Mesh ready")
            return str(preview_path), str(destination), "**Mesh ready.** Preview or download the OBJ below."


def build_demo(model_dir="CaiYuanhao/DiffusionGS", output_dir="outputs/object_demo", device="cuda:0"):
    service = ObjectDemo(model_dir, output_dir, device)
    with gr.Blocks(title="DiffusionGS · Object Generation", theme=gr.themes.Soft()) as demo:
        result_directory = gr.State()
        gr.Markdown("# DiffusionGS · Object Generation\n"
                    "Generate a 3D object from one image, then optionally extract its mesh.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(value=str(OBJECT_EXAMPLES[0]), type="pil", image_mode="RGBA",
                                 label="Object image", height=300, elem_id="object-input")
                gr.Examples([[str(path)] for path in OBJECT_EXAMPLES], inputs=[image],
                            examples_per_page=7, label="Object examples · click to select",
                            cache_examples=False, api_name=False, elem_id="object-examples")
                seed = gr.Number(value=62, precision=0, minimum=0, maximum=2**32 - 1, label="Seed")
                ratio = gr.Slider(0.5, 1, value=0.825, step=0.005, label="Foreground ratio")
                force = gr.Checkbox(value=False, label="Remove background even when transparency is provided")
                generate = gr.Button("Generate object", variant="primary")
                gr.Markdown("Background removal runs automatically for opaque images. "
                            "Transparent images can use their existing alpha mask.")
            with gr.Column(scale=2):
                views = gr.Gallery(label="Rendered object views", columns=4, height=270)
                cloud = gr.Plot(label="3D point cloud · drag to rotate / scroll to zoom")
                gr.Markdown("The preview shows up to 5,000 points; the PLY download contains the full object.")
                status = gr.Markdown("Choose an image and generate an object.")
                with gr.Row():
                    ply = gr.DownloadButton("Download point cloud (.ply)", interactive=False)
                    png = gr.DownloadButton("Download rendered views (.png)", interactive=False)
                mesh_button = gr.Button("Extract mesh", interactive=False)
                gr.Markdown("Mesh extraction is optional and can take several minutes. "
                            "It reuses the generated object without another diffusion pass.")
                mesh_status = gr.Markdown()
                mesh_preview = gr.Model3D(label="Mesh preview", height=350, interactive=False)
                obj = gr.DownloadButton("Download mesh (.obj)", interactive=False)
        generation = generate.click(
            service.generate, [image, seed, ratio, force],
            [views, cloud, ply, png, mesh_preview, obj, result_directory, status, mesh_status],
            api_name="generate_object", concurrency_id="object_gpu", concurrency_limit=1,
        )
        generation.success(
            lambda: (gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=True), gr.update(interactive=False)),
            outputs=[ply, png, mesh_button, obj], queue=False, api_name=False,
        )
        extraction = mesh_button.click(service.extract_mesh, [result_directory],
                                       [mesh_preview, obj, mesh_status], api_name="extract_mesh",
                                       concurrency_id="object_gpu", concurrency_limit=1)
        extraction.success(lambda: gr.update(interactive=True), outputs=obj, queue=False, api_name=False)
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="CaiYuanhao/DiffusionGS")
    parser.add_argument("--output-dir", default="outputs/object_demo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_demo(args.model_dir, args.output_dir, args.device)
    demo.launch(server_name=args.server_name, server_port=args.server_port,
                share=args.share, show_error=True, allowed_paths=[str(Path(args.output_dir).resolve())])


if __name__ == "__main__":
    main()
