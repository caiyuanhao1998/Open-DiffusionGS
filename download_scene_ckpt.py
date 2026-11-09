"""Download a released scene checkpoint and print its local path."""
import argparse

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=(256, 512), default=256)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    path = hf_hub_download("CaiYuanhao/DiffusionGS", f"scene_ckpt_{args.resolution}.ckpt",
                           cache_dir=args.cache_dir)
    print(path)


if __name__ == "__main__":
    main()
