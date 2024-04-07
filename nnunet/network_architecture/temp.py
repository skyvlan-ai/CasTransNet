# -*-coding:utf-8-*-
import math
from re import L
from einops import rearrange
from copy import deepcopy

from timm.layers import to_2tuple

from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from mmcv.cnn.bricks import ConvModule


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W*C)

        # Ensure the model's weights are on the same device as input x
        self.fc1.to(x.device)
        self.fc2.to(x.device)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.layernorm = LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_q = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 'same', bias=True, groups=dim),
            nn.GELU(),
        )
        self.layernorm_q = LayerNorm(dim, eps=1e-5, data_format='channels_first')

        self.layer_k = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 'same', bias=True, groups=dim),
            nn.GELU(),
        )
        self.layernorm_k = LayerNorm(dim, eps=1e-5, data_format='channels_first')

        self.layer_v = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 'same', bias=True, groups=dim),
            nn.GELU(),
        )
        self.layernorm_v = LayerNorm(dim, eps=1e-5, data_format='channels_first')

    def _build_projection(self, x, mode):
        # x shape [batch,channel,size,size]
        # mode:0->q,1->k,2->v,for torch.script can not script str

        if mode == 0:
            x1 = self.layer_q(x)
            proj = self.layernorm_q(x1)
        elif mode == 1:
            x1 = self.layer_k(x)
            proj = self.layernorm_k(x1)
        elif mode == 2:
            x1 = self.layer_v(x)
            proj = self.layernorm_v(x1)

        return proj

    def get_qkv(self, x):
        q = self._build_projection(x, 0)
        k = self._build_projection(x, 1)
        v = self._build_projection(x, 2)

        return q, k, v

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # x = x.permute(2, 0, 1)
        # q, k, v = self.get_qkv(x)
        # q = q.permute(1, 2, 0)
        # k = k.permute(1, 2, 0)
        # v = v.permute(1, 2, 0)
        # q = self.qkv(q)
        # k = self.qkv(k)
        # v = self.qkv(v)
        #
        # q = q.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k = k.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # v = v.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = LayerNorm(dim)

        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask):
        B, H, W, C = x.shape
        assert H * W == self.input_resolution[0] * self.input_resolution[1], "input feature has wrong size"

        shortcut = x

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up = nn.ConvTranspose2d(dim, dim // 2, 2, 2)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 channelAttention_reduce=4,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim
        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim * 2,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                channelAttention_reduce=channelAttention_reduce,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i,
            )
            for i in range(depth)])
        self.layernorm = LayerNorm(dim)
        self.layer = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, stride=1, padding="same"),
            nn.GELU()
        )
        self.bn = nn.BatchNorm2d(dim)
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim * 2, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, scale, relative_pos_enc=None):

        # norms
        # x = x.permute(0, 2, 3, 1)
        # x = self.layernorm(x)
        # x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        #Multi scale input
        x = torch.cat((self.layer(scale), x), axis=1)

        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim

        # build blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                i_block=i)
            for i in range(depth)])

        self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer)

    def forward(self, x, skip, H, W):
        x_up = self.Upsample(x, H, W)
        x = x_up + skip
        H, W = H * 2, W * 2
        for blk in self.blocks:
            x = blk(x)

        return x, H, W


class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        # self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm2(x)
            x = x.permute(0, 3, 1, 2)
        return x


class project_up(nn.Module):
    def __init__(self, in_dim, out_dim, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_block = int(np.log2(patch_size[0]))
        self.project_block = []
        self.dim = [int(embed_dim) // (2 ** (i+1)) for i in range(self.num_block)]
        self.dim.append(in_chans)
        self.dim = self.dim[::-1]  # in_ch, embed_dim/2, embed_dim or in_ch, embed_dim/4, embed_dim/2, embed_dim

        for i in range(self.num_block)[:-1]:
            self.project_block.append(project(self.dim[i], self.dim[i+1], 2, 1, nn.GELU, norm_layer, False))
        self.project_block.append(project(self.dim[-2], self.dim[-1], 2, 1, nn.GELU, norm_layer, True))
        self.project_block = nn.ModuleList(self.project_block)

        if norm_layer is not None:
            self.norm = norm_layer(self.dim[-1])
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - W % self.patch_size[0]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        for blk in self.project_block:
            x = blk(x)

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim // 2, Wh, Ww)

        return x


class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=[224, 224],
                 patch_size=[4, 4],
                 in_chans=1,
                 embed_dim=96,
                 depths=[3, 3, 3, 3],
                 channelAttention_reduce=4,
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 14, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (i_layer-1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer),
                depth=depths[i_layer],
                channelAttention_reduce=channelAttention_reduce,
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x, scale):
        """Forward function."""

        x = self.patch_embed(x)
        down = []

        Wh, Ww = x.size(2), x.size(3)

        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, scale[i])
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = x_out.permute(0, 2, 3, 1)
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                down.append(out)
        return down


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.num_block = int(np.log2(patch_size[0])) - 2
        self.project_block = []
        self.dim_list = [int(dim) // (2 ** i) for i in range(self.num_block + 1)]
        # dim, dim/2, dim/4
        for i in range(self.num_block):
            self.project_block.append(project_up(self.dim_list[i], self.dim_list[i + 1], nn.GELU, nn.LayerNorm, False))
        self.project_block = nn.ModuleList(self.project_block)
        self.up_final = nn.ConvTranspose2d(self.dim_list[-1], num_class, 4, 4)

    def forward(self, x):
        for blk in self.project_block:
            x = blk(x)
        x = self.up_final(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(input_channels, internal_neurons, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(internal_neurons, input_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        max_result = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        avg_result = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class PFB(nn.Module):
    def __init__(self, dim):
        super(PFB, self).__init__()
        self.conv1 = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(inplace=False)

    def forward(self, saf, caf):
        # caf = F.interpolate(caf, scale_factor=2.0, mode='nearest')
        # caf = F.avg_pool2d(caf, kernel_size=3, stride=1, padding=1)
        # saf = self.conv2(saf)
        out = torch.cat([saf, caf], dim=1)
        out = self.conv1(out)
        out = self.act(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 input_resolution,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.layer_q = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=channels),
            nn.LeakyReLU(),
        )
        self.layernorm_q = nn.LayerNorm([channels, input_resolution, input_resolution], eps=1e-5)

        self.layer_k = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.LeakyReLU(),
        )
        self.layernorm_k = nn.LayerNorm([channels, input_resolution, input_resolution], eps=1e-5)

        self.layer_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.LeakyReLU(),
        )
        self.layernorm_v = nn.LayerNorm([channels, input_resolution, input_resolution], eps=1e-5)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True, dropout=proj_drop)
        self.conv = nn.Conv2d(channels, channels, 3, 1, padding="same")

    def _build_projection(self, x, mode):
        # x shape [batch,channel,size,size]
        # mode:0->q,1->k,2->v,for torch.script can not script str

        if mode == 0:
            x1 = self.layer_q(x)
            proj = self.layernorm_q(x1)
        elif mode == 1:
            x1 = self.layer_k(x)
            proj = self.layernorm_k(x1)
        elif mode == 2:
            x1 = self.layer_v(x)
            proj = self.layernorm_v(x1)

        return proj

    def get_qkv(self, x):
        q = self._build_projection(x, 0)
        k = self._build_projection(x, 1)
        v = self._build_projection(x, 2)

        return q, k, v

    def forward(self, x):
        q, k, v = self.get_qkv(x)
        
        q = q.view(q.shape[0], q.shape[1], q.shape[2] * q.shape[3])
        k = k.view(k.shape[0], k.shape[1], k.shape[2] * k.shape[3])
        v = v.view(v.shape[0], v.shape[1], v.shape[2] * v.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        # x = self.conv(x)
        # x1 = torch.add(x1, x)

        # return nn.Sigmoid(x1)
        return x1


class MixAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, channelAttention_reduce=4, input_resolution=None,
                 num_heads=None, window_size=7, shift_size=0, qkv_bias=True, qk_scale=None, drop_path=0.):
        super().__init__()

        self.C = in_channels
        self.O = in_channels//2

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.act = nn.GELU()
        # self.swtrans = SwinTransformerBlock(dim=in_channels, input_resolution=input_resolution,
        #                                     num_heads=num_heads, window_size=window_size,
        #                                     shift_size=0 if (i_block % 2 == 0) else window_size // 2,
        #                                     mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                     drop=0., attn_drop=0., drop_path=drop_path)
        self.mhsa = MultiHeadAttention(in_channels, num_heads, input_resolution, drop_path)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        # self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.ln = LayerNorm(in_channels, data_format="channels_first",eps=1e-5)

    def forward(self, inputs):
        #   Global Perceptron
        x1 = self.bn(inputs)
        x1 = self.conv(x1)
        x1 = self.act(x1)
        channel_att_vec = self.ca(x1)
        x1 = channel_att_vec * x1
        x2 = self.ln(inputs)
        x2 = self.mhsa(x2)
        # x2 = spatial_att_vec * x2
        # x = torch.cat([x1, x2], dim=1)
        # x = self.proj(x2) + x1 ## STE
        # spatial = self.swtrans(x_s, mask)
        return x1, x2
    
    
#   The common FFN Block used in many Transformer and MLP models.
# Multi-scale Feed-forward Network
class FFNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.,):
        super().__init__()

        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,hidden_features,kernel_size=(1,1), bias=False, padding=0),
            nn.BatchNorm2d(hidden_features),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels,out_features,kernel_size=(1,1), bias=False, padding=0),
            nn.BatchNorm2d(out_features)
        )
    
        self.dconv1 = nn.Conv2d(hidden_features,hidden_features,kernel_size=(1,1),padding=(0,0),groups=hidden_features)
        self.dconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), padding=(1, 1),
                                groups=hidden_features)
        self.dconv5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(5, 5), padding=(2, 2),
                                groups=hidden_features)
        self.dconv7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(7, 7), padding=(3, 3),
                                groups=hidden_features)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.dconv1(x)
        x3 = self.dconv3(x)
        x5 = self.dconv5(x)
        x7 = self.dconv7(x)
        x = x + x1 + x3 + x5 + x7
        x = self.act(x)
        x = self.conv2(x)
        
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, input_resolution=None, num_heads=None,
                 window_size=None, channelAttention_reduce=4, shift_size=0, i_block=None, qkv_bias=None,
                 qk_scale=None, ffn_expand=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=dim)
        
        self.pfb = PFB(dim=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention = MixAttentionBlock(in_channels=dim, out_channels=dim,
                                           channelAttention_reduce=channelAttention_reduce,
                                           input_resolution=input_resolution[0], num_heads=num_heads,
                                           window_size=window_size,
                                           shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop_path=drop_path
                                     )
        self.ffn_block = FFNBlock(dim, dim * ffn_expand, drop=drop_path)

    def forward(self, x):
        input = x.clone()
        c, s = self.attention(x)
        x1 = self.pfb(s, c)
        x1 = input + self.drop_path(x1)
        x2 = self.bn(x1)
        x2 = self.ffn_block(x2)
        x = x1 + self.drop_path(x2)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionGate_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm(F_int, data_format="channels_first")
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU()
        # self.gelu = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        # self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        x_init = self.dconv5_5(x)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x) #
        out = spatial_att * x
        out = self.conv(out)

        return out


class CASCADE(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], channelAttention_reduce=16, num_heads=[16, 8, 4, 2],
                 img_size=[224, 224], window_size=[7, 7, 7, 7], dpr=None):
        super(CASCADE, self).__init__()
    
        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.Conv_3x3 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding="same")
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
        # self.ConvBlock4 = FFNBlock(channels[0], channels[0]*4, channels[0] )
        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = AttentionGate_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2 * channels[1], ch_out=channels[1])
        # self.ConvBlock3 = FFNBlock(2 * channels[1], 2 * channels[1] * 4, channels[1])
        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = AttentionGate_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2 * channels[2], ch_out=channels[2])
        # self.ConvBlock2 = FFNBlock(2 * channels[2], 2 * channels[2]*4, channels[2])
        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = AttentionGate_block(F_g=channels[3], F_l=channels[3], F_int=32)
        self.ConvBlock1 = conv_block(ch_in=2 * channels[3], ch_out=channels[3])
        # self.ConvBlock1 = FFNBlock(2 * channels[3], 2 * channels[3]*4, channels[3])

        self.CA4 = ChannelAttention(channels[0], internal_neurons=channels[0] // channelAttention_reduce)
        self.CA3 = ChannelAttention(2 * channels[1], internal_neurons=2 * channels[1] // channelAttention_reduce)
        self.CA2 = ChannelAttention(2 * channels[2], internal_neurons=2 * channels[2] // channelAttention_reduce)
        self.CA1 = ChannelAttention(2 * channels[3], internal_neurons=2 * channels[3] // channelAttention_reduce)
        self.SA4 = SpatialAttention(channels[0])
        self.SA3 = SpatialAttention(2 * channels[1])
        self.SA2 = SpatialAttention(2 * channels[2])
        self.SA1 = SpatialAttention(2 * channels[3])
        # self.mhsa4 = MultiHeadAttention(channels[0], num_heads[0], img_size[0]//32, dpr[0])
        # self.mhsa3 = MultiHeadAttention(2 * channels[1], num_heads[1], img_size[0]//16, dpr[1])
        # self.mhsa2 = MultiHeadAttention(2 * channels[2], num_heads[2], img_size[0]//8, dpr[2])
        # self.mhsa1 = MultiHeadAttention(2 * channels[3], num_heads[3], img_size[0]//4, dpr[3])

        self.bn4 = nn.BatchNorm2d(num_features=channels[0])
        # self.bn3 = nn.BatchNorm2d(num_features=channels[1])
        # self.bn2 = nn.BatchNorm2d(num_features=channels[2])
        # self.bn1 = nn.BatchNorm2d(num_features=channels[3])
        self.ln4 = LayerNorm(channels[0], data_format="channels_first")
        # self.ln3 = LayerNorm(channels[1])
        # self.ln2 = LayerNorm(channels[2])
        # self.ln1 = LayerNorm(channels[3])
        self.pfb4 = PFB(channels[0])
        self.pfb3 = PFB(2*channels[1])
        self.pfb2 = PFB(2*channels[2])
        self.pfb1 = PFB(2*channels[3])
        self.act = nn.GELU()
        # self.act = nn.LeakyReLU()

    def forward(self, x, skips):
        d4c = self.bn4(x)
        d4c = self.Conv_1x1(x)
        # d4 = self.act(d4)
        # d4_c = self.bn4(x)
        # d4_c = self.Conv_1x1(d4_c)
        # d4_c = self.act(d4_c)
        # CAM4
        d4c = self.CA4(d4c) * d4c
        d4s = self.ln4(x)
        d4s = self.SA4(d4s)
        # d4s = self.SA4(d4s) * d4s
        # d4s = self.mhsa4(d4s)
        d4 = self.pfb4(d4s, d4c)
        d4 = self.ConvBlock4(d4)

        # upconv3 1024*7*7->512*14*14
        d3 = self.Up3(d4)

        # AG3 14*14
        x3 = self.AG3(g=d3, x=skips[2])

        # Concat 3
        d3 = torch.cat((x3, d3), axis=1)

        # CAM3
        d3c = self.CA3(d3) * d3
        # # d3_s = self.ln4(d3)
        # # # _, _, H3, W3 = d3.shape
        # # d3_s = d3_s.permute(0, 2, 3, 1)
        # # d3_s = self.SA3(d3_s)
        # # d3_s = d3_s.permute(0, 3, 1, 2)
        d3s = self.SA3(d3)
        # d3s = self.SA3(d3) * d3
        
        # d3s = self.mhsa3(d3)
        d3 = self.pfb3(d3s, d3c)  # 1024*14*14
        d3 = self.ConvBlock3(d3)  # 512*14*14

        # upconv2 512*14*14->256*28*28
        d2 = self.Up2(d3)

        # AG2
        x2 = self.AG2(g=d2, x=skips[1])

        # Concat 2 512*28*28
        d2 = torch.cat((x2, d2), axis=1)
        # d2_c = self.bn3(d2)
        # CAM2
        d2c = self.CA2(d2) * d2
        # # d2_s = self.ln3(d2)
        # # # _, _, H2, W2 = d2.shape
        d2s = self.SA2(d2)
        # d2s = self.SA2(d2) * d2
       
        # d2s = self.mhsa2(d2)
        d2 = self.pfb2(d2s, d2c)
        d2 = self.ConvBlock2(d2)  # 512*28*28->256*28*28

        # upconv1 256*28*28->128*56*56
        d1 = self.Up1(d2)

        # AG1
        x1 = self.AG1(g=d1, x=skips[0])

        # Concat 1 256*56*56
        d1 = torch.cat((x1, d1), axis=1)
        # d1_c = self.bn2(d1)
        # CAM1
        d1c = self.CA1(d1) * d1
        # # d1_s = self.ln2(d1)
        # # # _, _, H1, W1 = d1.shape
        # d1 = d1.permute(0, 2, 3, 1)
        # # d1_s = self.SA1(d1_s)
        # d1 = d1.permute(0, 3, 1, 2)
        d1s = self.SA1(d1)
        # d1s = self.SA1(d1) * d1
        # d1s = self.mhsa1(d1)
        d1 = self.pfb1(d1s, d1c)
        d1 = self.ConvBlock1(d1)  # ->128*28*28
        return d4, d3, d2, d1


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class CASTransv2(SegmentationNetwork):
    def __init__(self,
                 config,
                 num_input_channels,
                 embedding_dim,
                 num_heads,
                 num_classes,
                 deep_supervision,
                 conv_op=nn.Conv2d):
        super(CASTransv2, self).__init__()

        # Don't uncomment conv_op
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.do_ds = deep_supervision
        self.embed_dim = embedding_dim
        self.depths = config.hyper_parameter.blocks_num
        self.num_heads = num_heads
        self.crop_size = config.hyper_parameter.crop_size
        self.patch_size = [config.hyper_parameter.convolution_stem_down, config.hyper_parameter.convolution_stem_down]
        self.window_size = config.hyper_parameter.window_size
        self.channelAttention_reduce = config.hyper_parameter.channelAttention_reduce
        self.dropout = nn.Dropout(0.1)
        # if window size of the encoder is [7,7,14,7], then decoder's is [14,7,7]. In short, reverse the list and start from the index of 1
        self.model_down = encoder(
            pretrain_img_size=self.crop_size,
            window_size=self.window_size,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            depths=self.depths,
            num_heads=self.num_heads,
            in_chans=self.num_input_channels,
            channelAttention_reduce=self.channelAttention_reduce,
            norm_layer=LayerNorm
        )
        num_features = [int(self.embed_dim * 2 ** i) for i in range(4)]
        self.filters = num_features[::-1][0:]
        num_windows = self.window_size[::-1]
        drop_path_rate = 0.2

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]  # stochastic depth decay rule
        # decoder initialization
        self.decoder = CASCADE(channels=self.filters, channelAttention_reduce=16,
                               num_heads=[16, 8, 4, 2], img_size=self.crop_size,
                               window_size=self.window_size[::-1][0:], dpr=self.dpr)
        # prediction heads
        self.segmentation_head1 = SegmentationHead(
            in_channels=self.filters[0],
            out_channels=4,
            kernel_size=1,
            upsampling=32,
        )
        self.segmentation_head2 = SegmentationHead(
            in_channels=self.filters[1],
            out_channels=4,
            kernel_size=1,
            upsampling=16,
        )
        self.segmentation_head3 = SegmentationHead(
            in_channels=self.filters[2],
            out_channels=4,
            kernel_size=1,
            upsampling=8,
        )
        self.segmentation_head4 = SegmentationHead(
            in_channels=self.filters[3],
            out_channels=4,
            kernel_size=1,
            upsampling=4,
        )
        # Multi-scale input
        self.scale_img1 = nn.AvgPool2d(4, 4)
        self.scale_img2 = nn.AvgPool2d(2, 2)
        self.final = []
        for i in range(len(self.depths) - 1):
            self.final.append(
                final_patch_expanding(self.embed_dim * 2 ** i, self.num_classes, patch_size=self.patch_size))
        self.final = nn.ModuleList(self.final)

    def forward(self, x):
        # Multi-scale input
        scale = []
        scale_img_1 = self.scale_img1(x)  # x shape[batch_size,channel(1),224,224] -> shape[batch,1,56,56]
        scale.append(scale_img_1)
        scale_img_2 = self.scale_img2(scale_img_1)  # shape[batch,1,28,28]
        scale.append(scale_img_2)
        scale_img_3 = self.scale_img2(scale_img_2)  # shape[batch,1,14,14]
        scale.append(scale_img_3)
        scale_img_4 = self.scale_img2(scale_img_3)  # shape[batch,1,7,7]
        scale.append(scale_img_4)
        skips = self.model_down(x, scale)
        neck = skips[-1]
        skips = skips[0:3]

        x1_o, x2_o, x3_o, x4_o = self.decoder(neck, skips)
        p1 = self.dropout(self.segmentation_head1(x1_o))
        p2 = self.dropout(self.segmentation_head2(x2_o))
        p3 = self.dropout(self.segmentation_head3(x3_o))
        p4 = self.dropout(self.segmentation_head4(x4_o))
        # w, x, y, z = 1.0, 1.0, 1.0, 1.0
        # output = w*p1 + x*p2 + y*p3 + z*p4
        return p1, p2, p3, p4
        # return output
        # if self.do_ds:
        #     # for training
        #     return output[::-1]
        #     # size [[224,224],[112,112],[56,56]]
        #
        # else:
        #     # for validation and testing
        #     return output[-1]
        #     # size [[224,224]]








