import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import xformers.ops as xops
except ImportError as e:
    print("Please install xformers to use flashatt v2")
    raise e

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(x)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L162-L168
def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ImageTokenizer(nn.Module):
    """
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L134-L214
    """

    def __init__(
        self,
        image_size,
        patch_size,
        d,
        in_channels=3,
        conv_bias=False,
        patch_token_dropout=0.0,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), f"Image size {image_size} must be divisible by the patch size {patch_size}."
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.conv = nn.Conv2d(
            in_channels, d, kernel_size=patch_size, stride=patch_size, bias=conv_bias
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.patch_token_dropout = nn.Dropout(p=patch_token_dropout)

    def forward(self, x):
        # (b, c, h, w) --> (b, l, d)
        # x in [-1, 1]
        x = self.conv(x).flatten(2).transpose(1, 2)
        x = self.patch_token_dropout(x + self.pos_embed)
        return x


class MLP(nn.Module):
    """
    MLP layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    Ignore: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L62
    """

    def __init__(
        self,
        d,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        mlp_dim=None,
    ):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = d * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim, bias=mlp_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, d, bias=mlp_bias),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        d,
        d_head,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        use_flashatt_v2=True,
    ):
        super().__init__()
        assert (
            d % d_head == 0
        ), f"Token dimension {d} should be divisible by head dimension {d_head}"
        self.d = d
        self.d_head = d_head
        self.attn_dropout = attn_dropout

        self.to_qkv = nn.Linear(d, 3 * d, bias=attn_qkv_bias)
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)

        self.use_flashatt_v2 = use_flashatt_v2

    def forward(self, x, subset_attention_size=None):
        """
        x: (b, l, d)
        """
        # token split, multi-head attention, token cat
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        if self.use_flashatt_v2:
            # Use the flash attention support from the xformers library
            # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
            q, k, v = map(
                lambda t: rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.d_head),
                (q, k, v),
            )

            if subset_attention_size is not None and subset_attention_size < q.shape[1]:
                x_subset = xops.memory_efficient_attention(
                    q[:, :subset_attention_size, :, :].contiguous(),
                    k[:, :subset_attention_size, :, :].contiguous(),
                    v[:, :subset_attention_size, :, :].contiguous(),
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
                x_rest = xops.memory_efficient_attention(
                    q[:, subset_attention_size:, :, :].contiguous(),
                    k,
                    v,
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
                # print(x_subset.shape, x_rest.shape)
                x = torch.cat([x_subset, x_rest], dim=1)
            else:
                # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
                x = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
            x = rearrange(x, "b l nh dh -> b l (nh dh)")
        else:
            # Use the flash attention support from the pytorch library
            q, k, v = (
                rearrange(q, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(k, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(v, "b l (nh dh) -> b nh l dh", dh=self.d_head),
            )
            # https://discuss.pytorch.org/t/flash-attention/174955/14
            dropout_p = self.attn_dropout if self.training else 0.0
            if subset_attention_size is not None and subset_attention_size < q.shape[2]:
                x = F.scaled_dot_product_attention(
                    q[:, :, :subset_attention_size, :].contiguous(),
                    k[:, :, :subset_attention_size, :].contiguous(),
                    v[:, :, :subset_attention_size, :].contiguous(),
                    dropout_p=dropout_p,
                )
                x_rest = F.scaled_dot_product_attention(
                    q[:, :, subset_attention_size:, :].contiguous(),
                    k,
                    v,
                    dropout_p=dropout_p,
                )
                x = torch.cat([x, x_rest], dim=2)
            else:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.attn_fc_dropout(self.fc(x))
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    Note: move drop_path to SelfAttention and MLP
    """

    def __init__(
        self,
        d,
        d_head,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)
        self.attn = SelfAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x, subset_attention_size=None):
        x = x + self.attn(self.norm1(x), subset_attention_size=subset_attention_size)
        x = x + self.mlp(self.norm2(x))
        return x



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        '''
            x 是 latent feature, 已经经过了 patch embedding
            c 是 conditioning, 主要有两部分 - timestep embedding 和 label embedding
            此处就只有一个 timestep embedding
        '''
        # 这算作 conditioning 的 MLP
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )   # 将 c 通过 adaLN_modulation 网络, 分成 6 个部分

        # 第一个模块 - MSA
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # 第二个模块 - Pointwise Feedforward
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x



class CrossAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        input_dim,
        d_head=64,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        use_flashatt_v2=True,
        num_heads=None,
        ctx_dim=None,
        causal=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_head = d_head
        self.num_heads = num_heads if num_heads is not None else input_dim // d_head
        self.ctx_dim = ctx_dim if ctx_dim is not None else input_dim
        self.att_dim = self.num_heads * self.d_head

        self.to_q = nn.Linear(self.input_dim, self.att_dim, bias=attn_qkv_bias)
        self.to_k = nn.Linear(self.ctx_dim, self.att_dim, bias=attn_qkv_bias)
        self.to_v = nn.Linear(self.ctx_dim, self.att_dim, bias=attn_qkv_bias)
        self.fc = nn.Linear(self.att_dim, self.input_dim, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)

        self.attn_dropout = attn_dropout
        assert self.attn_dropout == 0.0
        self.use_flashatt_v2 = use_flashatt_v2
        self.causal = causal

    def forward(self, x, y=None):
        """
        x: (b, l, d)
        y: (b, l', d)
        x cross-att y
        """
        if y is None:
            y = x

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        if self.use_flashatt_v2:
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.num_heads),
                (q, k, v),
            )

            # Use the flash attention support from the xformers library
            if self.causal:
                attention_bias = xops.LowerTriangularMask()
            else:
                attention_bias = None

            # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attention_bias,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )

            x = rearrange(x, "b n h d -> b n (h d)")
        else:
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
                (q, k, v),
            )

            # https://discuss.pytorch.org/t/flash-attention/174955/14
            dropout_p = self.attn_dropout if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

            x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.attn_fc_dropout(self.fc(x))
        return x

    def extra_repr(self) -> str:
        return (
            f"use_flashatt_v2={self.use_flashatt_v2}, "
            f"num_heads={self.num_heads}, "
            f"input_dim={self.input_dim}, "
            f"ctx_dim={self.ctx_dim}, "
            f"att_dim={self.att_dim}, "
        )


class CrossTransformerBlock(nn.Module):
    """
    Cross-att Transformer block
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    Note: move drop_path to SelfAttention and MLP
    """

    def __init__(
        self,
        d,
        d_head,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)

        self.attn = CrossAttention(
            d, d_head, attn_qkv_bias, attn_dropout, attn_fc_bias, attn_fc_dropout
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x), self.norm1(y))
        x = x + self.mlp(self.norm2(x))
        return x


class FixedLengthTransformerLayer(nn.Module):
    """ """

    def __init__(
        self,
        dim,
        context_dim=None,
        fixed_length=None,
        num_heads=8,
        head_dim=64,
        use_ln_context=True,
        mlp_dim=None,
    ):
        """
        :param dim: The input dim of x
        :param context_dim: The input dim of context
        :param fixed_length: The length of attention tokens
        :param num_heads: The number of attention heads
        :param head_dim: The dim of each attention head
        """
        super().__init__()
        self.has_cross_att = context_dim is not None
        self.dim = dim
        self.fixed_length = fixed_length

        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(
            input_dim=dim,
            d_head=head_dim,
            num_heads=num_heads,
        )

        if self.has_cross_att:
            self.ln_cross = nn.LayerNorm(dim)
            if use_ln_context:
                self.ln_context = nn.LayerNorm(context_dim)
            else:
                self.ln_context = nn.Identity()
            self.cross_attn = CrossAttention(
                input_dim=dim,
                ctx_dim=context_dim,
                d_head=head_dim,
                num_heads=num_heads,
            )

        self.ln_fc = nn.LayerNorm(dim)
        self.fc = MLP(
            d=dim,
            mlp_dim=mlp_dim,
        )

    def init_weight(self, total_layers):
        # Find all the linear layers that will contribute to the residual.
        linear_layer_list = [
            self.self_attn.fc,
            self.fc.mlp[2],
        ]
        if self.has_cross_att:
            linear_layer_list.append(self.cross_attn.fc)

        # divided the output by the total number of layers
        for linear_layer in linear_layer_list:
            linear_layer.weight.data /= total_layers
            if linear_layer.bias is not None:
                linear_layer.bias.data = 0.0

    def forward(self, x, context=None):
        """
        :param x: b, h_windows, w_windows, window_size, window_size, c;
                    can be in arbitrary shape but the layout is the same.
        :param context: b, context_length, context_dim
        :return:
        """
        batch_size, orig_length, orig_dim = x.shape
        context = context or x

        assert orig_dim % self.dim == 0, f"orig_dim: {orig_dim}, dim: {self.dim}"
        if self.fixed_length is not None:
            assert (orig_length * orig_dim) % (self.fixed_length * self.dim) == 0, (
                f"orig_length: {orig_length}, token_length: {self.fixed_length}"
                f"orig_dim: {orig_dim}, dim: {self.dim}."
                f"The product of orig_length * orig_dim must be divisible by token_length * dim."
                f"O.w., it will break the batches"
            )

        # Fixed-length self attention [-1, highres_tokens, highres_dim], e.g., [-1, 8192, 16]
        x = x.reshape(-1, self.fixed_length or orig_length, self.dim)
        x = x + self.self_attn(self.ln_self(x))

        # Cross attention [batch_size, -1, highres_dim], e.g., [8, -1, 16]
        if self.has_cross_att:
            x = x.reshape(batch_size, -1, self.dim)
            x = x + self.cross_attn(self.ln_cross(x), self.ln_context(context))

        # MLP layers [?, ?, highres_dim], e.g., [?, ?, 16]
        x = x + self.fc(self.ln_fc(x))

        # Reshape back to [batch_size, lowres_tokens, lowres_dim], e.g., [8, 1024, 1024]
        x = x.reshape(batch_size, orig_length, orig_dim)

        return x

    def extra_repr(self) -> str:
        return f"token_length={self.fixed_length}, dim={self.dim}"



class QK_Norm_SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """
    def __init__(
        self,
        d,
        d_head,
        attn_qkv_bias=False,
        attn_fc_bias=True,
        attn_dropout=0.0,
        attn_fc_dropout=0.0,
        use_qk_norm=True,
    ):
        super().__init__()
        assert d % d_head == 0, f"Token dimension {d} should be divisible by head dimension {d_head}"
        self.d = d
        self.d_head = d_head
        self.attn_dropout = attn_dropout
        self.to_qkv = nn.Linear(d, 3 * d, bias=attn_qkv_bias)
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(d_head)
            self.k_norm = RMSNorm(d_head)
    def forward(self, x, attn_bias=None):
        """
        x: (b, l, d)
        attn_bias: xformers BlockDiagonalMask
        """
        # token split, multi-head attention, token cat
        q, k, v = self.to_qkv(x).split(self.d, dim=2)
        # Use the flash attention support from the xformers library
        # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.d_head) for t in (q, k, v))
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # The memory_efficient_attention takes the input as (batch, seq_len, heads, dim)
        x = xops.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
            p=self.attn_dropout if self.training else 0.0,
            op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
        )
        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        return x




# 修改，加入 timestep condition 来加以控制
class DiTBlock_QK_Norm(nn.Module):
    """
    Transformer block
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    Note: move drop_path to SelfAttention and MLP
    """
    def __init__(
        self,
        d,
        d_head,
        ln_bias=True,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=True,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=True,
        mlp_dropout=0.0,
        use_qk_norm=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)
        self.attn = QK_Norm_SelfAttention(
            d, d_head, attn_qkv_bias, attn_fc_bias, attn_dropout, attn_fc_dropout, use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d, 6 * d, bias=True)
        )
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )   # 将 c 通过 adaLN_modulation 网络, 分成 6 个部分

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # 第二个模块 - Pointwise Feedforward
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x