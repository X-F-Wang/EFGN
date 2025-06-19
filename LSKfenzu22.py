import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import pdb
from bsrnutils import Conv2d1x1, Conv2d3x3, CCA,  DepthwiseSeparableConv2d,PixelMixer
import torch.nn.functional as f




class ShiftConv2d1x1(nn.Conv2d):
    r"""

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), bias: bool = True, shift_mode: str = '+', val: float = 1.,
                 **kwargs) -> None:
        super(ShiftConv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                             dilation=dilation, groups=1, bias=bias, **kwargs)

        assert in_channels % 5 == 0, f'{in_channels} % 5 != 0.'

        channel_per_group = in_channels // 5
        self.mask = nn.Parameter(torch.zeros((in_channels, 1, 3, 3)), requires_grad=False)
        if shift_mode == '+':
            self.mask[0 * channel_per_group:1 * channel_per_group, 0, 1, 2] = val
            self.mask[1 * channel_per_group:2 * channel_per_group, 0, 1, 0] = val
            self.mask[2 * channel_per_group:3 * channel_per_group, 0, 2, 1] = val
            self.mask[3 * channel_per_group:4 * channel_per_group, 0, 0, 1] = val
            self.mask[4 * channel_per_group:, 0, 1, 1] = val
        elif shift_mode == 'x':
            self.mask[0 * channel_per_group:1 * channel_per_group, 0, 0, 0] = val
            self.mask[1 * channel_per_group:2 * channel_per_group, 0, 0, 2] = val
            self.mask[2 * channel_per_group:3 * channel_per_group, 0, 2, 0] = val
            self.mask[3 * channel_per_group:4 * channel_per_group, 0, 2, 2] = val
            self.mask[4 * channel_per_group:, 0, 1, 1] = val
        else:
            raise NotImplementedError(f'Unknown shift mode for ShiftConv2d1x1: {shift_mode}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.conv2d(input=x, weight=self.mask, bias=None,
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=self.in_channels)
        x = f.conv2d(input=x, weight=self.weight, bias=self.bias,
                     stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        return x



class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn




class BlueprintSeparableConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3),
                 stride: tuple = 1, padding: tuple = 1, dilation: tuple = (1, 1), bias: bool = True,
                 mid_channels: int = None, **kwargs) -> None:
        super(BlueprintSeparableConv, self).__init__()

        if mid_channels is not None:
            self.pw = nn.Sequential(Conv2d1x1(in_channels, mid_channels, bias=False),
                                    Conv2d1x1(mid_channels, out_channels, bias=False))

        else:
            self.pw = Conv2d1x1(in_channels, out_channels, bias=False)

        self.dw = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=out_channels,
                                  bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.pw(x))


class TokenMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.token_mixer = PixelMixer(planes=dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.token_mixer(x) - x)


class ESA(nn.Module):

    def __init__(self, in_channels, planes: int = None, num_conv: int = 3, conv_layer=BlueprintSeparableConv,
                 **kwargs) -> None:
        super(ESA, self).__init__()

        planes = planes or in_channels // 4
        self.head_conv = Conv2d1x1(in_channels, planes)

        self.stride_conv = conv_layer(planes, planes, stride=(2, 2), **kwargs)
        conv_group = list()
        for i in range(num_conv):
            if i != 0:
                conv_group.append(nn.ReLU(inplace=True))
            conv_group.append(conv_layer(planes, planes, **kwargs))
        self.group_conv = nn.Sequential(*conv_group)
        self.useless_conv = Conv2d1x1(planes, planes)  # maybe nn.Identity()?

        self.tail_conv = Conv2d1x1(planes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_output = self.head_conv(x)

        stride_output = self.stride_conv(head_output)
        pool_output = f.max_pool2d(stride_output, kernel_size=7, stride=3)
        group_output = self.group_conv(pool_output)
        upsample_output = f.interpolate(group_output, (x.size(2), x.size(3)),
                                        mode='bilinear', align_corners=False)

        tail_output = self.tail_conv(upsample_output + self.useless_conv(head_output))
        sig_output = torch.sigmoid(tail_output)

        return x * sig_output




class ESDB(nn.Module):
    def __init__(self, planes: int, distillation_rate: float = 0.5,
                 ) -> None:
        super(ESDB, self).__init__()

        distilled_channels = int(planes * distillation_rate)

        self.c1_d = Conv2d1x1(planes, distilled_channels)
        self.c1_r = TokenMixer(planes)

        self.c2_d = Conv2d1x1(planes, distilled_channels)
        self.c2_r = TokenMixer(planes)

        self.c3_d = Conv2d1x1(planes, distilled_channels)
        self.c3_r = TokenMixer(planes)

        self.c4_r = Conv2d1x1(planes, distilled_channels)

        self.c5 = Conv2d1x1(distilled_channels * 4, planes)

        self.cca = CCA(planes)
        self.esa = ESA(planes)

        self.act = nn.GELU()

        self.norm = nn.BatchNorm2d(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        r_c1 = self.act(x + r_c1)

        d_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c1 + r_c2)

        d_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c2 + r_c3)

        r_c4 = self.c4_r(r_c3)
        r_c4 = self.act(r_c4)

        out = torch.cat([d_c1, d_c2, d_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.cca(self.esa(out))

        return self.norm(out_fused + x)



class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)

class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class LayerNorm4D(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class TransLayermul(nn.Module):
    def __init__(self, planes: int, b_blocks:int,) -> None:
        super().__init__()
        m = []

        for i in range(b_blocks):
            m.append(TransLayer(planes))
        self.net = nn.Sequential(*m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.net(x)
        res += x
        return res

class MLP4D(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer: nn.Module = nn.GELU) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = ShiftConv2d1x1(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = ShiftConv2d1x1(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransLayer(nn.Module):
    def __init__(self, planes: int, ) -> None:
        super().__init__()
        self.esdb = ESDB(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.esdb(x)
        return x

class TransLayer2(nn.Module):
    def __init__(self, planes: int, ) -> None:
        super().__init__()
        self.lsk = LSKblock(planes)
        self.norm1 = LayerNorm4D(planes)

        self.mlp = MLP4D(planes, planes * 2, )
        self.norm2 = LayerNorm4D(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.lsk(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x






class Pre_ProcessLayer_Graph(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride, bias = True):
        super(Pre_ProcessLayer_Graph, self).__init__()
        self.head = prosessing_conv(in_feats, out_feats, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.head(x)
        [B, C, H, W] = x.shape
        y = torch.reshape(x, [B, C, H*W])
        N = H*W
        y = y.permute(0,2,1).contiguous()
        adj = torch.zeros(B, N, N).cuda()
        k = 9
        for b in range(B):
            dist = cdist(y[b,:,:].cpu().detach().numpy(), y[b,:,:].cpu().detach().numpy(), metric='euclidean')
            dist = np.where(dist.argsort(1).argsort(1) <= 6, 1, 0)
            dist = torch.from_numpy(dist).type(torch.FloatTensor)
            dist = torch.unsqueeze(dist, 0)
            adj[b,:,:] = dist
        return y, adj


class ProcessLayer_Graph(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride, bias = True):
        super(ProcessLayer_Graph, self).__init__()
        self.last = transpose_conv(in_feats, out_feats, kernel_size, stride, bias=bias)

    def forward(self, x):
        y = self.last(x)
        return y





class GCN_CNN_Unit(nn.Module):
    def __init__(self, b_blocks,in_feats, out_feats, up_scale, use_tail=True, conv=default_conv, ):
        super(GCN_CNN_Unit, self).__init__()
        kernel_size = 3
        self.pre = conv(in_feats, out_feats, kernel_size)
        self.head = TransLayermul(out_feats,b_blocks)
        self.last = conv(out_feats, out_feats, kernel_size)
        self.upsample = Upsampler(conv, up_scale, out_feats)
        self.tail = True
        if use_tail:
            self.tail = conv(out_feats, in_feats, kernel_size)

    def forward(self, x):
        y = self.pre(x)
        GCN_result = self.head(y)
        y = GCN_result
        y = self.last(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)
        return y

class GCN_CNN_Unit2(nn.Module):
    def __init__(self,  out_feats, up_scale, use_tail=False, conv=default_conv):
        super(GCN_CNN_Unit2, self).__init__()
        kernel_size = 3
        self.head = TransLayer2(out_feats)




    def forward(self, x):
        GCN_result = self.head(x)
        y = GCN_result
        return y




class SSPN(nn.Module):
    def __init__(self, out_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()


        m = []

        for i in range(n_blocks):
            m.append(GCN_CNN_Unit2(out_feats,up_scale=2,use_tail = False))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x


        return res


class Spatial_Spectral_Unit(nn.Module):
    def __init__(self, in_feats, out_feats, n_blocks, act, res_scale, up_scale, use_tail=False, conv=default_conv):
        super(Spatial_Spectral_Unit, self).__init__()
        kernel_size = 3
        self.head = conv(in_feats, out_feats, kernel_size)
        self.body = SSPN(out_feats, n_blocks, act, res_scale)
        self.upsample = Upsampler(conv, up_scale, out_feats)



        if use_tail:
            self.tail = conv(out_feats, in_feats, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)


        return y


class LSKfenzu22(nn.Module):
    def __init__(self, n_subs, n_ovls, in_feats,b_blocks, n_blocks, out_feats, n_scale, res_scale, use_share=True, conv=default_conv):
        super(LSKfenzu22, self).__init__()
        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)

        self.G = math.ceil((in_feats - n_ovls) / (n_subs - n_ovls))
        self.start_idx = []
        self.end_idx = []
        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > in_feats:
                end_ind = in_feats
                sta_ind = in_feats - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = GCN_CNN_Unit(b_blocks,n_subs, out_feats, up_scale=n_scale//2, use_tail=True, conv=default_conv)
        else:
            self.branch = nn.ModuleList
            for i in range(self.G):
                self.branch.append(GCN_CNN_Unit(n_subs, out_feats, up_scale=n_scale//2, use_tail=True, conv=default_conv))
        self.trunk = Spatial_Spectral_Unit(in_feats, out_feats, n_blocks, act, res_scale, up_scale=2, use_tail=False, conv=default_conv)
        self.skip_conv = conv(in_feats, out_feats, kernel_size)
        self.final = conv(out_feats, in_feats, kernel_size)
        self.sca = n_scale//2

    def forward(self, x, lms):
        b, c, h, w = x.shape

        y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()

        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]

            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            if self.shared:
                xi = self.branch(xi)
            else:
                xi = self .branch[g](xi)
                print("xi.shape:", xi.shape)

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1





        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.trunk(y)
        y = y + self.skip_conv(lms)
        y = self.final(y)

        return y