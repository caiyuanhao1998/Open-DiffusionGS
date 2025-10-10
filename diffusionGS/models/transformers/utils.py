import torch.nn as nn
import torch

from timm.models.vision_transformer import Mlp, DropPath
#from utils import timestep_embedding

import math
# --------------------------------------------------------
# Timestep embedding
# References:
# Point-E: https://github.com/openai/point-e
# --------------------------------------------------------
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

class MLP(nn.Module):
    def __init__(self, *,
                 width: int,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


# Code borrow from: https://github.com/facebookresearch/PointInfinity/blob/main/modules.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# RIN: https://arxiv.org/pdf/2212.11972
# --------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            num_heads=16,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        kv_dim = dim if not kv_dim else kv_dim
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        B, N_kv, _ = x_kv.shape
        # [B, N_q, C] -> [B, N_q, H, C/H] -> [B, H, N_q, C/H]
        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B, N_kv, C] -> [B, N_kv, H, C/H] -> [B, H, N_kv, C/H]
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B, N_kv, C] -> [B, N_kv, H, C/H] -> [B, H, N_kv, C/H]
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [B, H, N_q, C/H] @ [B, H, C/H, N_kv] -> [B, H, N_q, N_kv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # [B, H, N_q, N_kv] @ [B, H, N_kv, C/H] -> [B, H, N_q, C/H]
        x = attn @ v

        # [B, H, N_q, C/H] -> [B, N_q, C]
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Compute_Block(nn.Module):

    def __init__(self, z_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_z1 = norm_layer(z_dim)
        self.attn = CrossAttention(
            z_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_z2 = norm_layer(z_dim)
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z):
        zn = self.norm_z1(z)
        z = z + self.drop_path(self.attn(zn, zn))
        z = z + self.drop_path(self.mlp(self.norm_z2(z)))
        return z

class Read_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_x = norm_layer(x_dim)
        self.norm_z1 = norm_layer(z_dim)
        self.attn = CrossAttention(
            z_dim, x_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_z2 = norm_layer(z_dim)
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        z = z + self.drop_path(self.attn(self.norm_z1(z), self.norm_x(x)))
        z = z + self.drop_path(self.mlp(self.norm_z2(z)))
        return z

class Write_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_z = norm_layer(z_dim)
        self.norm_x1 = norm_layer(x_dim)
        self.attn = CrossAttention(
            x_dim, z_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_x2 = norm_layer(x_dim)
        mlp_hidden_dim = int(x_dim * mlp_ratio)
        self.mlp = Mlp(in_features=x_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        x = x + self.drop_path(self.attn(self.norm_x1(x), self.norm_z(z)))
        x = x + self.drop_path(self.mlp(self.norm_x2(x)))
        return x

class RCW_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_compute_layers=4, num_heads=16, 
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.read = Read_Block(z_dim, x_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                   attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.write = Write_Block(z_dim, x_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                   attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.compute = nn.ModuleList([
            Compute_Block(z_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(num_compute_layers)
        ])

    def forward(self, z, x):
        z = self.read(z, x)
        for layer in self.compute:
            z = layer(z)
        x = self.write(z, x)
        return z, x

class Denoiser_backbone(nn.Module):
    def __init__(self, input_channels=3, output_channels=3,
                 num_z=256, z_dim=768, x_dim=512, 
                 num_blocks=6, num_compute_layers=4, num_heads=8, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.num_z = num_z
        self.z_dim = z_dim

        # input blocks
        self.input_proj = nn.Linear(input_channels, x_dim)
        self.ln_pre = nn.LayerNorm(x_dim)
        #self.z_init = nn.Parameter(torch.zeros(1, num_z, z_dim))

        # timestep embedding
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.time_embed = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim)

        # RCW blocks
        self.latent_mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ln_latent = nn.LayerNorm(z_dim)
        self.blocks = nn.ModuleList([
            RCW_Block(z_dim, x_dim, num_compute_layers=num_compute_layers, 
                      num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                      drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
                      act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(num_blocks)
        ])

        # output blocks
        self.ln_post = nn.LayerNorm(x_dim)
        self.output_proj = nn.Linear(x_dim, output_channels)

        self.initialize_weights()   

    def initialize_weights(self):       
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        nn.init.constant_(self.ln_latent.weight, 0)
        nn.init.constant_(self.ln_latent.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, cond, t):
        """
        Forward pass of the model.

        Parameters:
        x: [B, num_x, C_in]
        t: [B]
        cond: [B, num_cond, C_latent]
        prev_latent: [B, num_cond + 1, C_latent]

        Returns:
        x_denoised: [B, num_x, C_out]
        z: [B, num_cond + 1, C_latent]
        """
        B, num_x, _ = x.shape
        num_cond = cond.shape[1]
        #assert num_x == self.num_x
        # timestep embedding, [B, 1, z_dim]
        t_embed = self.time_embed(timestep_embedding(t, self.z_dim)).unsqueeze(1)

        # project x -> [B, num_x, C_x]
        x = self.input_proj(x)
        x = self.ln_pre(x)

        # latent self-conditioning
        #z = self.z_init.repeat(B, 1, 1) # [B, num_z, z_dim]
        z = torch.cat([cond, t_embed], dim=1) # [B, num_z + num_cond + 1, z_dim]
        #prev_latent = prev_latent + self.latent_mlp(prev_latent.detach())
        #z = z # + self.ln_latent(prev_latent)

        # compute
        for blk in self.blocks:
            z, x = blk(z, x)
        
        # output proj
        x = self.ln_post(x)
        x_denoised = self.output_proj(x)
        return x_denoised, z