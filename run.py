from diffusionGS.pipline_obj import DiffusionGSPipeline
import torch
from torchvision.utils import save_image
## export U2NET_HOME=/mnt/bn/yixun3d/u2net_cache
pipeline = DiffusionGSPipeline.from_pretrained("CaiYuanhao/DiffusionGS", device="cuda:0", torch_dtype=torch.float16)
gs_output = pipeline("extra_files/test_cases/dp.png",seed=62, foreground_ratio = 0.825)#for the case that with pose
##export gaussians
gs_output.gaussians.save_ply("debug/test.ply")
##export image
save_image(gs_output.render_images, "debug/test.png")
