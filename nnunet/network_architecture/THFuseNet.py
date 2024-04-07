# -*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.Vit import Channel, Spatial
from nnunet.network_architecture.neural_network import SegmentationNetwork


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        # x = x.cuda()
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        # else:
        #     out = F.relu(out, inplace=True)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.res = ResidualBlock(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res(x)
        x = self.conv2(x)
        return x


class project_up(nn.Module):
    def __init__(self, in_dim, out_dim, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=1)
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


class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.num_block = int(np.log2(patch_size[0])) - 2
        self.project_block = []
        for i in range(self.num_block):
            self.project_block.append(project_up(64, 64, nn.GELU, nn.LayerNorm, False))
        self.project_block = nn.ModuleList(self.project_block)
        self.up_final = nn.ConvTranspose2d(64, num_class, 4, 4)

    def forward(self, x):
        for blk in self.project_block:
            x = blk(x)
        x = self.up_final(x)
        return x

class Block_up(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # conv
        self.bn = nn.BatchNorm2d(num_features=dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 channelAttention_reduce=4,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                 ):
        super().__init__()
        self.depth = depth
        self.dim = dim

        # build blocks
        self.blocks = nn.ModuleList([
            Block_up(dim=dim)
            for i in range(depth)])

        self.Upsample = upsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, skip, H, W):
        x_up = self.Upsample(x, H, W)
        x = x_up + skip
        H, W = H * 2, W * 2

        for blk in self.blocks:
            x = blk(x)

        return x, H, W


class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)

    def forward(self, x, H, W):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up(x)
        return x


class decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=(4, 4),
                 depths=(3, 3, 3),
                 channelAttention_reduce=4,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=embed_dim,
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1)),

                depth=depths[i_layer],
                channelAttention_reduce=channelAttention_reduce,
                drop_path=dpr[sum(
                    depths[:(len(depths) - i_layer - 1)]):sum(depths[:(len(depths) - i_layer)])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        outs = []
        H, W = x.size(2), x.size(3)
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            x, H, W, = layer(x, skips[i], H, W)
            outs.append(x)
        return outs


class Fusenet(SegmentationNetwork):
    def __init__(self,
                 config,
                 num_input_channels,
                 embedding_dim,
                 num_heads,
                 num_classes,
                 deep_supervision,
                 conv_op=nn.Conv2d):
        super(Fusenet, self).__init__()
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
        self.channelAttention_reduce = config.hyper_parameter.channelAttention_reduce
        kernel_size = 1
        stride = 1
        # ---------------CNN--------------
        self.conv_in = ConvLayer(self.num_input_channels, 16, 3, 1)

        self.Conv_d = ConvLayer(16, 8, 3, 1)
        self.layers = nn.ModuleDict({
            'DenseConv1': ConvLayer(8, 8, 3, 1),
            'DenseConv2': ConvLayer(16, 8, 3, 1),
            'DenseConv3': ConvLayer(24, 8, 3, 1)
        })

        self.Conv1 = ConvLayer(16, 32, 3, 2)
        self.Conv2 = ConvLayer(32, 64, 3, 2)
        self.Conv3 = ConvLayer(64, 32, 3, 2)
        self.Upsample = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)

        # ----------------vit--------------
        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(4)
        self.down3 = nn.AvgPool2d(8)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_in1 = ConvLayer(self.num_input_channels, self.num_input_channels, kernel_size, stride)
        self.conv_out = ConvLayer(64, self.num_classes, kernel_size, stride=2, is_last=True)

        self.en0 = Encoder(64, 64, kernel_size, stride)
        self.en1 = Encoder(64, 64, kernel_size, stride)
        self.en2 = Encoder(64, 64, kernel_size, stride)
        self.en3 = Encoder(64, 64, kernel_size, stride)

        self.ctrans3 = Channel(size=32, embed_dim=128, patch_size=8, channel=64)
        self.strans3 = Spatial(size=256, embed_dim=1024 * 2, patch_size=4, channel=64)
        self.decoder = decoder(
            pretrain_img_size=self.crop_size,
            embed_dim=64,
            patch_size=self.patch_size,
            depths=[2, 1, 1],
            channelAttention_reduce=self.channelAttention_reduce
        )
        self.final = []
        for i in range(len(self.depths) - 1):
            self.final.append(
                final_patch_expanding(64, self.num_classes, patch_size=self.patch_size))
        self.final = nn.ModuleList(self.final)

    def forward(self, x):
        seg_outputs = []
        x = self.conv_in1(x)
        x = self.conv_in(x)
        x_d = self.Conv_d(x)

        for i in range(len(self.layers)):
            out = self.layers['DenseConv' + str(i + 1)](x_d)
            x_d = torch.cat([x_d, out], 1)

        x_s = self.Conv1(x)
        x_s = self.Conv2(x_s)
        x_s = self.Conv3(x_s)
        x_s = self.Upsample(x_s)

        x0 = torch.cat([x_d, x_s], dim=1)
        down0 = self.down2(x0)
        for i in range(0, int(self.depths[0])):
            x0 = self.en0(down0)
        down1 = self.down1(x0)
        for i in range(0, int(self.depths[1])):
            x1 = self.en1(down1)
        down2 = self.down1(x1)
        for i in range(0, int(self.depths[2])):
            x2 = self.en1(down2)
        down3 = self.down1(x2)
        for i in range(0, int(self.depths[3])):
            x3 = self.en1(down3)
        # x1 = self.en1(self.down1(x0))
        # x2 = self.en2(self.down1(x1))
        # x3 = self.en3(self.down1(x2))

        x3t = self.strans3(self.ctrans3(x3))
        x3m = x3t
        x3r = x3 * x3m
        x2m = self.up1(x3m)
        x2r = x2 * x2m
        x1m = self.up1(x2m) + self.up2(x3m)
        x1r = x1 * x1m
        x0m = self.up1(x1m) + self.up2(x2m) + self.up3(x3m)
        x0r = x0 * x0m
        x0r = self.up3(x3r) + self.up2(x2r) + self.up1(x1r) + x0r
        skips = []
        skips.append(x0r)
        skips.append(x1r)
        skips.append(x2r)
        skips.append(x3r)

        neck = skips[-1]
        out = self.decoder(neck, skips)

        for i in range(len(out)):
            seg_outputs.append(self.final[-(i + 1)](out[i]))
        if self.do_ds:
            # for training
            return seg_outputs[::-1]
            # size [[224,224],[112,112],[56,56]]

        else:
            # for validation and testing
            return seg_outputs[-1]
            # size [[224,224]]

        # f1 = self.conv_out(other)
        # out = self.final(f1)
        # return f1
