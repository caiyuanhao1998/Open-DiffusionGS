from diffusionGS.models.gsrenderer.gs_core import (
    GaussianModel,
    render_opencv_cam,
    render_turntable,
    RGB2SH,
    deferred_gaussian_render,
    deferred_gaussian_render_scene,
    imageseq2video,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
'''
    调用时输入的是 Gaussian Model 的参数
    随后 init Gaussian model, 用 set_data 传参
    最后调用 render_opencv_cam 渲染
    返回的是渲染出来的 multi-view images
'''
class Renderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.scaling_modifier = None #onfig.model.gaussians.get("scaling_modifier", None)
        self.gaussians_model = GaussianModel(
            config.gaussians_sh_degree, self.scaling_modifier
        )
        print(f"in Renderer, scaling_modifier: {self.scaling_modifier}")
        print(
            f"in Renderer, gaussians_model.scaling_modifier: {self.gaussians_model.scaling_modifier}"
        )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        deferred=True,
    ):
        """
        高斯常规的一些 attribute
        xyz: [b, n_gaussians, 3]
        features: [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: [b, n_gaussians, 3]
        rotation: [b, n_gaussians, 4]
        opacity: [b, n_gaussians, 1]

        height: int
        width: int
        C2W: [b, v, 4, 4]
        fxfycxcy: [b, v, 4]

        output: [b, v, 3, height, width]
        """

        if deferred:
            renderings = deferred_gaussian_render(
                xyz,
                features,
                scaling,
                rotation,
                opacity,
                height,
                width,
                C2W,
                fxfycxcy,
                self.scaling_modifier,
                self.config.use_gssplat,
            )
        else:
            b, v = C2W.size(0), C2W.size(1)
            renderings = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device
            )

            for i in range(b):
                pc = self.gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                for j in range(v):
                    renderings[i, j] = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )["render"]
        return renderings



class SceneRenderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.sh_degree = config.model.gaussians.sh_degree
        self.scaling_modifier = None #config.model.gaussians.get("scaling_modifier", None)
        self.gaussians_model = GaussianModel(
            config.gaussians_sh_degree, self.scaling_modifier
        )
        print(f"in Renderer, scaling_modifier: {self.scaling_modifier}")
        print(f"in Renderer, gaussians_model.scaling_modifier: {self.gaussians_model.scaling_modifier}")

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        defer_render=False,
    ):
        """
        xyz: [b, n_gaussians, 3]
        features: [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: [b, n_gaussians, 3]
        rotation: [b, n_gaussians, 4]
        opacity: [b, n_gaussians, 1]

        height: int
        width: int
        C2W: [b, v, 4, 4]
        fxfycxcy: [b, v, 4]

        output: [b, v, 3, height, width]
        """

        if defer_render == True:
            # if False:
            renderings, depth, alpha = deferred_gaussian_render_scene(
                xyz, features, scaling, rotation, opacity, height, width, C2W, fxfycxcy, self.scaling_modifier
            )
        else:
            b, v = C2W.size(0), C2W.size(1)
            renderings = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device
            )

            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )

            for i in range(b):
                pc = self.gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                for j in range(v):
                    # renderings[i, j] = render_opencv_cam(
                    #     pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    # )["render"]
                    buffers = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )
                    renderings[i, j] = buffers["render"]
                    if "depth" in buffers and buffers["depth"] is not None: 
                        depth[i, j] = buffers["depth"]
                    if "alpha" in buffers and buffers["alpha"] is not None:
                        alpha[i, j] = buffers["alpha"]

        # return renderings
        return renderings #edict(render=renderings, depth=depth, alpha=alpha)