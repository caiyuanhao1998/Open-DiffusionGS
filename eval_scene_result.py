"""Compute image-weighted PSNR, SSIM and LPIPS for saved scene evaluations."""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from diffusionGS.utils.losses import MetricComputer


@torch.inference_mode()
def compute_metrics(psnr_path, chunk=8, device=None):
    if chunk < 1:
        raise ValueError("chunk must be positive")
    directory = Path(psnr_path)
    paths = sorted(directory.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt evaluation results found in {directory}")
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    metric_computer = MetricComputer().to(device).eval()
    totals = dict(psnr=0.0, ssim=0.0, lpips=0.0)
    count = 0
    for path in tqdm(paths, desc="Evaluation results"):
        result = torch.load(path, map_location="cpu", weights_only=True)
        predicted = result["render_images"].float()
        target = result["image"].float()
        if predicted.shape != target.shape or predicted.ndim < 4 or predicted.shape[-3] != 3:
            raise ValueError(f"Mismatched or invalid RGB image shapes in {path}")
        if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
            raise ValueError(f"Nonfinite image values in {path}")
        predicted = predicted.reshape(-1, *predicted.shape[-3:])
        target = target.reshape(-1, *target.shape[-3:])
        for start in range(0, len(target), chunk):
            pred_batch = predicted[start:start + chunk].to(device)
            gt_batch = target[start:start + chunk].to(device)
            values = metric_computer(gt_batch, pred_batch)
            for name, value in zip(totals, values):
                totals[name] += value.double().sum().item()
            count += len(gt_batch)
    if count == 0:
        raise ValueError(f"No images found in {directory}")
    summary = {name: total / count for name, total in totals.items()}
    summary["num_images"] = count
    (directory / "eval_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Directory containing saved .pt evaluation results")
    parser.add_argument("--chunk", type=int, default=8, help="Maximum images evaluated per batch")
    parser.add_argument("--device", default=None, help="Default: CUDA when available, otherwise CPU")
    args = parser.parse_args()
    compute_metrics(args.path, args.chunk, args.device)
