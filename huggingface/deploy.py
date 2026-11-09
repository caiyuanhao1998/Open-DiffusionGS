"""Publish the two Spaces from the shared demo source and a prebuilt CUDA wheel."""
import argparse
from pathlib import Path
import subprocess

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]


def tracked_files(directory):
    return subprocess.check_output(
        ["git", "ls-files", "-z", directory], cwd=ROOT,
    ).decode().strip("\0").split("\0")


def deploy(wheel):
    api = HfApi()
    object_files = [(path, ROOT / path) for directory in ("diffusionGS", "extra_files")
                    for path in tracked_files(directory) if path]
    object_files += [(name, ROOT / name) for name in
                     ("object_demo.py", "scene_demo.py", "camera_trajectories.py")]
    object_files += [(name, ROOT / "huggingface/object" / name) for name in
                     ("README.md", "space_app.py", "requirements.txt", "packages.txt")]
    object_files += [(f"wheels/{wheel.name}", wheel),
                     ("wheels/RASTERIZER_LICENSE.md", ROOT / "submodules/diff-gaussian-rasterization/LICENSE.md")]
    screenshots = [(f"img/demos/{name}.png", ROOT / f"img/demos/{name}.png")
                   for name in ("object_demo", "scene_demo")]
    for repo, files in (
        ("CaiYuanhao/DiffusionGS-Object", object_files + screenshots),
        ("CaiYuanhao/DiffusionGS-Scene", screenshots + [(name, ROOT / "huggingface/scene" / name)
                                                     for name in ("README.md", "index.html")]),
    ):
        commit = api.create_commit(
            repo_id=repo, repo_type="space", commit_message="Deploy DiffusionGS online generation demos",
            operations=[CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(source))
                        for dest, source in files],
        )
        print(repo, commit.oid, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args()
    deploy(args.wheel.resolve(strict=True))
