import math
import torch
import torch.nn as nn
import torch.nn.functional as F





def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=k):
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1) // 2, bias=bias)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation)

    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation)



def default_conv2(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=k):

       padding = int((kernel_size - 1) / 2) * dilation
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           stride, padding=padding, bias=bias, dilation=dilation)


def prosessing_conv(in_channels, out_channels, kernel_size, stride, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=bias)

def transpose_conv(in_channels, out_channels, kernel_size, stride, bias=True):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1, bias=bias)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat


        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        B, N, C = x.size()
        h = torch.matmul(x, self.W)
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N*N, self.out_features), h.repeat(1, N, 1)], dim=2).view(-1, N, N, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e12 * torch.ones_like(e)


        print(adj.device)
        print(e.device)
        print(zero_vec.device)

        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        h_prime = torch.bmm(attention, h)


        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(out_features * n_heads, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=2)


class CALayer(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialResBlock(nn.Module):
    def __init__(self, conv, in_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SpatialResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(in_feats, in_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class SpectralAttentionResBlock(nn.Module):
    def __init__(self, conv, in_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SpectralAttentionResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(in_feats, in_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(in_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, up_scale, in_feats, bn=False, act=False, bias=True):
        m = []
        if (up_scale & (up_scale - 1)) == 0:
            for _ in range(int(math.log(up_scale, 2))):
                m.append(conv(in_feats, 4 * in_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(in_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(in_feats))

        elif up_scale == 3:
            m.append(conv(in_feats, 9 * in_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(in_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)