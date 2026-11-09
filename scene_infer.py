"""Generate a scene, rendered views, and a ten-second round-trip video."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch
from torchvision.utils import save_image

from camera_trajectories import build_trajectory
from diffusionGS.pipeline_scene import DiffusionGSScenePipeline
from diffusionGS.models.gsrenderer.gs_core import render_generic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="CaiYuanhao/DiffusionGS")
    parser.add_argument("--image", default=str(Path(__file__).parent / "extra_files/test_cases/scene_room.png"))
    parser.add_argument("--output-dir", default="outputs/scene_infer")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = build_trajectory(num_frames=120, resolution=256)
    pipeline = DiffusionGSScenePipeline.from_pretrained(
        args.model_dir, resolution=256, device="cuda:0", torch_dtype=torch.float16,
    )
    keyframes = trajectory.keyframe_indices
    scene = pipeline(args.image, trajectory.c2ws[keyframes], trajectory.intrinsics[keyframes], seed=62)
    scene.gaussians.save_ply(str(output_dir / "scene.ply"))
    save_image(scene.render_images, str(output_dir / "scene.png"))

    frames = render_generic(scene.gaussians, torch.from_numpy(trajectory.c2ws).cuda(),
                            torch.from_numpy(trajectory.intrinsics).cuda(), h=256, w=256)
    with imageio.get_writer(str(output_dir / "scene.mp4"), fps=24, codec="libx264",
                           pixelformat="yuv420p", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
        for frame in frames[::-1]:
            writer.append_data(frame)

    cameras = trajectory.to_dict()
    playback = list(range(len(frames))) + list(range(len(frames) - 1, -1, -1))
    for key in ("c2ws", "intrinsics"):
        cameras[key] = [cameras[key][index] for index in playback]
    cameras.update(seed=62, inference_steps=30, fps=24, duration_seconds=10,
                   forward_duration_seconds=5, reverse_duration_seconds=5,
                   rendered_frames=len(frames), playback="forward_then_reverse",
                   playback_frame_indices=playback)
    (output_dir / "cameras.json").write_text(json.dumps(cameras, indent=2))
    print(f"Saved scene.ply, scene.png, scene.mp4 and cameras.json to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
