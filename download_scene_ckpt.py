from huggingface_hub import hf_hub_download
import os



os.makedirs('scene_ckpts',exist_ok=True)
breakpoint()
ckpt_path = hf_hub_download(repo_id='CaiYuanhao/DiffusionGS', filename="scene_ckpt_256.ckpt", repo_type="model",cache_dir='scene_ckpts')