"""Generate object views, Gaussian points, and an optional mesh from one image."""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusionGS.pipline_obj import DiffusionGSPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="CaiYuanhao/DiffusionGS")
    parser.add_argument("--image", default=str(Path(__file__).parent / "extra_files/test_cases/dp.png"))
    parser.add_argument("--output-dir", default="outputs/object_infer")
    parser.add_argument("--no-mesh", action="store_true", help="Skip mesh extraction and OBJ export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = DiffusionGSPipeline.from_pretrained(
        args.model_dir, device="cuda:0", torch_dtype=torch.float16,
    )
    output = pipeline(args.image, seed=62, foreground_ratio=0.825, extract_mesh=not args.no_mesh)
    output.gaussians.save_ply(str(output_dir / "test.ply"))
    save_image(output.render_images, str(output_dir / "test.png"))
    if not args.no_mesh:
        output.mesh.export(output_dir / "test.obj")
    print(f"Saved object outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
