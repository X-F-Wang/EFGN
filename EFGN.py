import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import pdb



class PConv(nn.Module):
    def __init__(self, planes: int, num_touched: int, kernel_size: int ,
                forward_type: str = 'splitting') -> None:
        super().__init__()

        self.num_touched = num_touched
        self.num_untouched = planes - num_touched

        self.conv = nn.Conv2d(self.num_touched,
                              self.num_touched,
                              (kernel_size, kernel_size), stride=1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)

        if forward_type == 'splitting':
            self.forward = self.forward_splitting
        elif forward_type == 'slicing':
            self.forward = self.forward_slicing
        else:
            raise NotImplementedError

    def forward_splitting(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.num_touched, self.num_untouched], dim=1)
        return torch.cat((self.conv(x1), x2), dim=1)

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        x[:, :self.num_touched, :, :] = self.conv(x[:, :self.num_touched, :, :])
        return x


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


class SSRGM(nn.Module):
    def __init__(self, planes: int, kernel_size: int, pks1:int, pratio1:int) -> None:
        super().__init__()

        self.h_mixer = nn.Conv2d(planes, planes, kernel_size=(kernel_size, 1),
                                 padding=(kernel_size // 2, 0), groups=planes)
        self.w_mixer = nn.Conv2d(planes, planes, kernel_size=(1, kernel_size),
                                 padding=(0, kernel_size // 2), groups=planes)
        self.c_mixer = Conv2d1x1(planes, planes)
        self.pconv = PConv(planes, planes // pratio1,pks1)

        self.proj = Conv2d1x1(planes, planes)
        self.channelatt = CALayer(planes, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        _hw = self.h_mixer(x) * self.w_mixer(x)
        _hw=x+_hw
        _hw=self.pconv(_hw)

        _chw1 =  self.c_mixer(_hw)
        _chw2 = self.channelatt(_hw)
        _chw = _chw1 * _chw2
        _chw = _hw + _chw

        return self.proj(_chw)

class TDSSRGM(nn.Module):
    def __init__(self, planes: int, kernel_size: int,pks2:int,pratio2:int) -> None:
        super().__init__()

        self.h_mixer = nn.Conv3d(1, 1, kernel_size=(kernel_size,kernel_size, 1),
                                 padding=(kernel_size // 2, kernel_size // 2,0), groups=1)
        self.w_mixer = nn.Conv3d(1, 1, kernel_size=(1, 1,kernel_size),
                                 padding=(0, 0,kernel_size // 2), groups=1)
        self.c_mixer = Conv2d1x1(planes, planes)
        self.pconv = PConv(planes, planes // pratio2,pks2)

        self.proj = Conv2d1x1(planes, planes)
        self.channelatt = CALayer(planes, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=x.unsqueeze(1)
        _hw = self.h_mixer(x) * self.w_mixer(x)
        _hw=x+_hw

        _hw = _hw.squeeze(1)
        _hw=self.pconv(_hw)
        _chw1 =  self.c_mixer(_hw)
        _chw2 = self.channelatt(_hw)
        _chw = _chw1 * _chw2
        _chw = _hw + _chw

        return self.proj(_chw)

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
    def __init__(self, planes: int, b_blocks:int,kernel_size: int,pks1:int,pratio1:int) -> None:
        super().__init__()
        m = []

        for i in range(b_blocks):
            m.append(TransLayer(planes,kernel_size,pks1,pratio1))
        self.net = nn.Sequential(*m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.net(x)
        res += x
        return res



class TransLayer(nn.Module):
    def __init__(self, planes: int, kernel_size: int,pks1:int,pratio1:int) -> None:
        super().__init__()

        self.sg = SSRGM(planes, kernel_size,pks1,pratio1)
        self.norm1 = LayerNorm4D(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sg(self.norm1(x))
        return x

class TransLayer2(nn.Module):
    def __init__(self, planes: int, kernel_size: int,pks2:int,pratio2:int) -> None:
        super().__init__()

        self.sg = TDSSRGM(planes, kernel_size,pks2,pratio2)
        self.norm1 = LayerNorm4D(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sg(self.norm1(x))
        return x









class Subgroup(nn.Module):
    def __init__(self, b_blocks,in_feats, out_feats1,largeks1,pks1,pratio1, up_scale, use_tail=True, conv=default_conv, conv2=default_conv2):  # up_scale
        super(Subgroup, self).__init__()
        kernel_size = 3
        self.pre = conv(in_feats, out_feats1, kernel_size)
        self.head = TransLayermul(out_feats1,b_blocks,largeks1,pks1,pratio1)
        self.process = conv2(out_feats1, out_feats1, kernel_size,dilation=k)
        self.processbef = Conv2d1x1(out_feats1, out_feats1)
        self.last = conv(out_feats1, out_feats1, kernel_size)
        self.upsample = Upsampler(conv, up_scale, out_feats1)
        self.tail = True
        if use_tail:
            self.tail = conv(out_feats1, in_feats, kernel_size)

    def forward(self, x):
        y = self.pre(x)
        y3=self.processbef(y)
        GCN_result = self.head(y)
        y = GCN_result
        y1 = GCN_result
        y1 = self.process(y1)
        y = self.last(y)

        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)
        return y,y1,y3

class Totalgroup(nn.Module):
    def __init__(self,  out_feats2,largeks2,pks2,pratio2, up_scale, use_tail=False, conv=default_conv):
        super(Totalgroup, self).__init__()
        kernel_size = 3
        self.pre = conv( out_feats2, out_feats2, kernel_size)
        self.head = TransLayer2(out_feats2,largeks2,pks2,pratio2)
        self.last = conv(out_feats2, out_feats2, kernel_size)




    def forward(self, x):
        y = self.pre(x)
        GCN_result = self.head(y)
        y = GCN_result
        y = self.last(y)
        return y


class SSPN(nn.Module):
    def __init__(self, out_feats2,largeks2,pks2,pratio2, n_blocks, act, res_scale):
        super(SSPN, self).__init__()


        m = []

        for i in range(n_blocks):
            m.append(Totalgroup(out_feats2,largeks2,pks2,pratio2,up_scale=2,use_tail = False))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x

        return res


class Spatial_Spectral_Unit(nn.Module):
    def __init__(self, in_feats, out_feats2,largeks2,pks2,pratio2, n_blocks, act, res_scale, up_scale, use_tail=False, conv=default_conv):
        super(Spatial_Spectral_Unit, self).__init__()
        kernel_size = 3
        self.head = conv(in_feats, out_feats2, kernel_size)
        self.body = SSPN(out_feats2,largeks2,pks2,pratio2, n_blocks, act, res_scale)
        self.upsample = Upsampler(conv, up_scale, out_feats2)



        if use_tail:
            self.tail = conv(out_feats2, in_feats, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)


        return y

class ShuffleDown(nn.Module):
    def __init__(self, n_scale):
        super(ShuffleDown, self).__init__()
        self.scale = n_scale


    def forward(self, x):
        b, cin, hin, win= x.size()
        cout = cin * self.scale ** 2
        hout = hin // self.scale
        wout = win // self.scale
        output = x.view(b, cin, hout, self.scale, wout, self.scale)
        output = output.permute(0, 1, 5, 3, 2, 4).contiguous()
        output = output.view(b, cout, hout, wout)
        return output


class EFGN(nn.Module):
    def __init__(self, b_blocks,n_subs, n_ovls, in_feats, n_blocks, out_feats1,out_feats2,largeks1,largeks2,pks1,pratio1,pks2,pratio2, n_scale, res_scale, use_share=True, conv=default_conv):
        super(EFGN, self).__init__()
        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)

        self.G = math.ceil((in_feats - n_ovls) / (n_subs - n_ovls))
        self.start_idx = []
        self.end_idx = []
        print("G=",self.G)

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > in_feats:
                end_ind = in_feats
                sta_ind = in_feats - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = Subgroup(b_blocks,n_subs, out_feats1,largeks1,pks1,pratio1, up_scale=n_scale//2, use_tail=True, conv=default_conv)

        else:
            self.branch = nn.ModuleList
            for i in range(self.G):
                self.branch.append(Subgroup(n_subs, out_feats1,largeks1,pks1,pratio1, up_scale=n_scale//2, use_tail=True, conv=default_conv))




        self.trunk = Spatial_Spectral_Unit(in_feats, out_feats2,largeks2,pks2,pratio2,  n_blocks, act, res_scale, up_scale=2, use_tail=False, conv=default_conv)
        self.skip_conv = conv(in_feats, out_feats2, kernel_size)
        self.final = conv(out_feats2, in_feats, kernel_size)
        self.sca = n_scale//2
        self.down=ShuffleDown(n_scale//2)
        self.out_feats1 = out_feats1
        self.n_subs = n_subs

        self.process = conv(2*out_feats1 + n_subs , n_subs, kernel_size)

    def forward(self, x, lms):
        b, c, h, w = x.shape

        y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()

        channel_counter = torch.zeros(c).cuda()
        xi2 = torch.zeros(b, self.out_feats1, h, w).cuda()
        xi3 = torch.zeros(b, self.out_feats1, h, w).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]

            end_ind = self.end_idx[g]
            xi = self.process(torch.cat([x[:, sta_ind:end_ind, :, :], xi2, xi3], dim=1))

            if self.shared:
                xi1, xi2,xi3 = self.branch(xi)

            else:
                xi = self.branch[g](xi)
                print("xi.shape:", xi.shape)
            y[:, sta_ind:end_ind, :, :] += xi1

            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.trunk(y)
        y = y + self.skip_conv(lms)
        y = self.final(y)

        return y