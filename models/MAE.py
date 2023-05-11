import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


def ConvBnRelu(channel_in, channel_out, kernal_size):
    conv_bn_relu = nn.Sequential(
        nn.Conv1d(channel_in, channel_out, kernal_size),
        nn.BatchNorm1d(channel_out),
        nn.LeakyReLU(0.2, inplace=True))
    return conv_bn_relu


def DConvBnRelu(channel_in, channel_out, kernal_size):
    d_conv_bn_relu = nn.Sequential(
        nn.ConvTranspose1d(channel_in, channel_out, kernal_size),
        nn.BatchNorm1d(channel_out),
        nn.LeakyReLU(0.2, inplace=True))
    return d_conv_bn_relu


class MAE(nn.Module):
    ''' Vanilla AE '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, 3),
            ConvBnRelu(16, 32, 3),
            ConvBnRelu(32, 64, 3))

        self.mem = MemoryModule(mem_dim=1000, fea_dim=64, shrink_thres=0.0025)

        self.decoder = nn.Sequential(
            DConvBnRelu(64, 32, 3),
            DConvBnRelu(32, 16, 3),
            nn.ConvTranspose1d(16, 3, 3))

    def forward(self, x, args, mode):
        feature = self.encoder(x)
        att, feature = self.mem(args, feature)
        output = self.decoder(feature)

        if mode == 'train':
            loss = self.mse(x, output)
            return loss
        else:
            return output


def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    ''' Hard Shrinking '''
    return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + epsilon)


class MemoryModule(nn.Module):
    ''' Memory Module '''

    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        # attention
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # [C, M]
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        ''' init memory elements : Very Important !! '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, args, x):
        ''' x [B,C,H,W] : latent code Z'''
        B, C, H = x.shape
        x = x.permute(0, 2, 1).flatten(end_dim=1)  # Fea : [NxC]  N=BxHxW
        # calculate attention weight
        att_weight = F.linear(x, self.weight)  # Fea*Mem^T : [NxC] x [CxM] = [N, M]
        att_weight = F.softmax(att_weight, dim=1)  # [N, M]

        # generate code z'
        mem_T = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_T)  # Fea*Mem^T^T : [N, M] x [M, C] = [N, C]
        output = output.view(B, H, C).permute(0, 2, 1)  # [N,C,H,W]

        return att_weight, output

