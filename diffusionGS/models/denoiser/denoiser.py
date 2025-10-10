from dataclasses import dataclass
import math
import copy
import torch
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from easydict import EasyDict as edict
import diffusionGS
from diffusionGS.models.transformers.utils_transformer import DiTBlock,_init_weights
from diffusionGS.models.gsrenderer.renderer import Renderer
from diffusionGS.utils.checkpoint import checkpoint
from diffusionGS.utils.base import BaseModule
from diffusionGS.utils.typing import *
from diffusionGS.utils.ops import generate_dense_grid_points
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    '''
        staticmethod 装饰器可以声明静态方法
        无需实例化 TimestepEmbedder 类, 就可以直接调用 timestep_embedding 方法
    '''
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    


class GaussiansUpsampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        xyz : torch.tensor of shape (n_gaussians, 3)
        features : torch.tensor of shape (n_gaussians, (sh_degree + 1) ** 2, 3)
        scaling : torch.tensor of shape (n_gaussians, 3)
        rotation : torch.tensor of shape (n_gaussians, 4)
        opacity : torch.tensor of shape (n_gaussians, 1)
        """
        self.layernorm = nn.LayerNorm(config.width, bias=False) # d - dimension - channel

        u = 1 #config.model.gaussians.upsampler.upsample_factor
        if u > 1:
            raise NotImplementedError("GaussiansUpsampler only supports u=1")
        else:
            self.linear = nn.Linear(
                config.width,
                3 + (config.gaussians_sh_degree + 1) ** 2 * 3 + 3 + 4 + 1,        # 直接转成 per-pixel Gaussian map
                bias=False,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.width, 2 * config.width, bias=True)
        )

    def to_gs(self, gaussians):     # 把 dimenssion 分掉，分出对应的维度
        """
        gaussians: [b, n_gaussians, d]
        n_gaussians - 高斯的数量
        d - 每个高斯的 attribute 的数量
        """
        xyz, features, scaling, rotation, opacity = gaussians.split(
            [3, (self.config.gaussians_sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=2
        )
        features = features.reshape(
            features.size(0),
            features.size(1),
            (self.config.gaussians_sh_degree + 1) ** 2,
            3,
        )
        scaling = (scaling - 2.3).clamp(max=-1.20)
        opacity = opacity - 2.0
        return xyz, features, scaling, rotation, opacity

    def forward(self, gaussians, t_embedding):
        """
        gaussians: [b, n_gaussians, d]
        t_embedding: [b, d]
        output: [b, n_gaussians, dd]
        """
        u = 1 #self.config.model.gaussians.upsampler.upsample_factor
        if u > 1:
            raise NotImplementedError("GaussiansUpsampler only supports u=1")
        
        shift, scale = self.adaLN_modulation(t_embedding).chunk(2, dim=1)
        gaussians = modulate(self.layernorm(gaussians), shift, scale)
        gaussians = self.linear(gaussians)

        return gaussians


class ImageTokenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layernorm = nn.LayerNorm(config.width, bias=False)
        self.linear = nn.Linear(
                config.width,
                (config.patch_size**2)
                * (3 + (config.gaussians_sh_degree + 1) ** 2 * 3 + 3 + 4 + 1),
                bias=False,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.width, 2 * config.width, bias=True)
        )

    def forward(self, img_tokens, t_embedding):
        """
        img_tokens: [b, n_patches, d]
        t_embedding: [b, d]
        output: [b, n_patches, dd]
        """
        shift, scale = self.adaLN_modulation(t_embedding).chunk(2, dim=1)
        img_tokens = modulate(self.layernorm(img_tokens), shift, scale)
        img_tokens = self.linear(img_tokens)
        return img_tokens


###################### AutoEncoder
@diffusionGS.register("diffusion-gs-model")
class DGSDenoiser(BaseModule):
    r"""
    An image to gaussian diffusion models.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        use_downsample: bool = False
        num_latents: int = 256
        width: int = 1024
        in_channels: int = 3
        patch_size: int = 16
        n_gaussians: int = 2
        dim_heads: int = 64
        num_layers: int = 24
        ray_pe_type: str = "relative_plk"
        hard_pixelalign: bool = True
        clip_xyz: bool = True
        ##gaussian relative
        gaussians_sh_degree: int = 0
        use_gssplat: bool = False
        ##diffusion relative
        prior_distribution: str = "gaussian"
        ####
        use_flash: bool = False
        use_checkpoint: bool = True
        grad_checkpoint_every: int = 1

    cfg: Config
    def configure(self) -> None:
        super().configure()
        ##
        ##time embedder
        self.t_embedder = TimestepEmbedder(self.cfg.width)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        ##image tokenlizer
                # 这是 patchify and linear layer
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.cfg.patch_size,
                pw=self.cfg.patch_size,
            ),
            nn.Linear(
                self.cfg.in_channels
                * (self.cfg.patch_size**2),
                self.cfg.width,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)
        ##gaussian pos embedding
        self.gaussians_pos_embedding = nn.Parameter(
            torch.randn(
                self.cfg.n_gaussians,     # 3d gaussian 的个数？n_gaussians = 2?
                self.cfg.width,             # d = 1024
            )
        )
        nn.init.trunc_normal_(self.gaussians_pos_embedding, std=0.02)


        self.transformer_input_layernorm = nn.LayerNorm(
            self.cfg.width, bias=False
        )
        self.transformer = nn.ModuleList(
            [
                DiTBlock(
                    self.cfg.width, self.cfg.width // self.cfg.dim_heads
                )
                for _ in range(self.cfg.num_layers)
            ]
        )

        self.transformer.apply(_init_weights)

        self.upsampler = GaussiansUpsampler(self.cfg)
        self.upsampler.apply(_init_weights)     
        self.image_token_decoder = ImageTokenDecoder(self.cfg)
        self.image_token_decoder.apply(_init_weights)
        # encoder
        self.gs_renderer = Renderer(self.cfg)

        # renderer
        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained shape model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'model' in pretrained_ckpt.keys():
                pret_weights = pretrained_ckpt['model']
                _pretrained_ckpt = {}
                # breakpoint()
                for k, v in pret_weights.items():
                    if k.startswith('denoiser.') and not k.startswith('denoiser.loss_computer'):
                        _pretrained_ckpt[k.replace('denoiser.', '')] = v
                        #
                pretrained_ckpt = _pretrained_ckpt

            # if 'state_dict' in pretrained_ckpt:
            #     _pretrained_ckpt = {}
            #     for k, v in pretrained_ckpt['state_dict'].items():
            #         if k.startswith('shape_model.'):
            #             _pretrained_ckpt[k.replace('shape_model.', '')] = v
            #     pretrained_ckpt = _pretrained_ckpt
            # else:
            #     _pretrained_ckpt = {}
            #     for k, v in pretrained_ckpt.items():
            #         if k.startswith('shape_model.'):
            #             _pretrained_ckpt[k.replace('shape_model.', '')] = v
            #     pretrained_ckpt = _pretrained_ckpt
            # breakpoint()
            self.load_state_dict(pretrained_ckpt, strict=True)

    def forward(self, input_batch, timesteps):
        guassians_parameters,_ = self.image_to_gaussians(input_batch['image'], input_batch['ray_o'], input_batch['ray_d'], timesteps)
        rendered_images = self.render_gaussians(guassians_parameters, input_batch['c2w'], input_batch['fxfycxcy'], input_batch['image'].shape[3], input_batch['image'].shape[4])
        return rendered_images, self.prepare_to_save(guassians_parameters)
    

    def prepare_to_save(self,gaussians_parameters):
        gaussians = []
        for b in range(gaussians_parameters.xyz.size(0)):
            self.gs_renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.gs_renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    gaussians_parameters.xyz[b].detach().float(),
                    gaussians_parameters.features[b].detach().float(),
                    gaussians_parameters.scaling[b].detach().float(),
                    gaussians_parameters.rotation[b].detach().float(),
                    gaussians_parameters.opacity[b].detach().float(),
                )
            )
        return gaussians

    def image_to_gaussians(self,
                        images:torch.FloatTensor,
                        ray_o:torch.FloatTensor,
                        ray_d:torch.FloatTensor,
                        t:torch.LongTensor,
                        training: bool = False):
        if self.cfg.ray_pe_type == "relative_plk":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d           
            posed_images = torch.cat(
                [
                    images[:, :, :3, :, :] * 2.0 - 1.0,
                    ray_d,
                    nearest_pts,
                ],
                dim=2,
            )
        else:
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            posed_images = torch.cat(
                [
                    images[:, :, :3, :, :] * 2.0 - 1.0,
                    o_cross_d,
                    ray_d,
                ],
                dim=2,
            )
        b, v, c, h, w = posed_images.size()
        img_tokens = self.image_tokenizer(posed_images)
        t = self.t_embedder(t)  # [b, d]

        _, n_patches, d = img_tokens.size()  # [b*v, n_patches, d]
        img_tokens = img_tokens.reshape(b, v * n_patches, d)  # [b, v*n_patches, d]

        # [b, n_gaussians, d]
        gaussians_tokens = self.gaussians_pos_embedding.expand(b, -1, -1)   # 复制 b 份

        checkpoint_every = self.cfg.grad_checkpoint_every
        concat_nerf_img_tokens = torch.cat((gaussians_tokens, img_tokens), dim=1)
        concat_nerf_img_tokens = self.transformer_input_layernorm(
            concat_nerf_img_tokens
        )
        for i in range(0, len(self.transformer), checkpoint_every):
            concat_nerf_img_tokens = torch.utils.checkpoint.checkpoint(
                self.run_layers(i, i + checkpoint_every),
                concat_nerf_img_tokens,
                t,
                use_reentrant=False,
            )
        gaussians_tokens, img_tokens = concat_nerf_img_tokens.split(
            [self.cfg.n_gaussians, v * n_patches], dim=1
        )
        gaussians = self.upsampler(gaussians_tokens, t)

        # [b, v*n_patches, p*p*gs]
        img_aligned_gaussians = self.image_token_decoder(img_tokens, t)
        img_aligned_gaussians = img_aligned_gaussians.reshape(
            b,
            -1,
            (3 + (self.cfg.gaussians_sh_degree + 1) ** 2 * 3 + 3 + 4 + 1),
        )  # [b, v*pixels, gs]
        n_img_aligned_gaussians = img_aligned_gaussians.size(1)
        all_gaussians = torch.cat((gaussians, img_aligned_gaussians), dim=1)
        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(all_gaussians)
        img_aligned_xyz = xyz[:, -n_img_aligned_gaussians:, :]  # 把 image 对应的 Gaussians 模型 xyz 取出来
        img_aligned_xyz = rearrange(
            img_aligned_xyz,
            "b (v h w ph pw) c -> b v c (h ph) (w pw)",
            v=v,
            h=h // self.cfg.patch_size,
            w=w // self.cfg.patch_size,
            ph=self.cfg.patch_size,
            pw=self.cfg.patch_size,
        )

        # 对 image_aligned_xyz 进行矫正
        if self.cfg.hard_pixelalign:   #这是 true, 要执行的
            depth_preact_bias = 0. 
            img_aligned_xyz = torch.sigmoid(
                img_aligned_xyz.mean(dim=2, keepdim=True) + depth_preact_bias
            )
            # stx()
            #breakpoint()
            if self.cfg.ray_pe_type == 'relative_plk':
                img_aligned_xyz = (2.0 * img_aligned_xyz - 1.0) * 1.8 + o_dot_d
                # print(f"Using augmented plucker coordinates to compute xyz")
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d
            # breakpoint()
            # 将坐标限制在 -1 到 1 之间
            if self.cfg.clip_xyz and training:
                img_aligned_xyz = img_aligned_xyz.clamp(-1.0, 1.0)

            img_aligned_xyz_reshape = rearrange(
                img_aligned_xyz,
                "b v c (h ph) (w pw) -> b (v h w ph pw) c",
                ph=self.cfg.patch_size,
                pw=self.cfg.patch_size,
            )
            xyz = torch.cat(
                (xyz[:, :-n_img_aligned_gaussians, :], img_aligned_xyz_reshape), dim=1
            )
            result_softpa = edict(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
            )


        return result_softpa, img_aligned_xyz



    def render_gaussians(self, gaussian_params, c2w, fxfycxcy, height, width):
        # breakpoint()
        render_input = self.gs_renderer(
                    gaussian_params.xyz,
                    gaussian_params.features,
                    gaussian_params.scaling,
                    gaussian_params.rotation,
                    gaussian_params.opacity,
                    height,
                    width,
                    C2W=c2w,               # 
                    fxfycxcy=fxfycxcy,       # 
                )
        #self.gs_renderer()
        return render_input
    
    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype
    
    def run_layers(self, start, end):
        def custom_forward(concat_nerf_img_tokens, t):
            for i in range(start, min(end, len(self.transformer))):
                concat_nerf_img_tokens = self.transformer[i](concat_nerf_img_tokens, t)
            return concat_nerf_img_tokens

        return custom_forward