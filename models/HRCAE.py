import torch
import torch.nn as nn


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


class Inception(nn.Module):
    def __init__(self, kernal_size):
        super(Inception, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernal_size),
            ConvBnRelu(16, 32, kernal_size),
            ConvBnRelu(32, 64, kernal_size))
        self.decoder = nn.Sequential(
            DConvBnRelu(64, 32, kernal_size),
            DConvBnRelu(32, 16, kernal_size),
            nn.ConvTranspose1d(16, 3, kernal_size))

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class HRCAE(nn.Module):
    def __init__(self):
        super(HRCAE, self).__init__()
        self.mse = nn.MSELoss()

        self.incep1 = Inception(kernal_size=3)
        self.incep2 = Inception(kernal_size=3)

    def forward(self, x, args, mode):
        output1 = self.incep1(x)
        output2 = self.incep2(x)

        if mode == 'train':
            cos_loss1 = cosine_similarity(x, output1)
            cos_loss2 = cosine_similarity(x, output2)
            loss1 = self.mse(x, output1) + args.lambd * cos_loss1
            loss2 = self.mse(x, output2) + args.lambd * cos_loss2
            loss = loss1 + loss2
            return loss
        else:
            return output1, output2


def cosine_similarity(x, y):
    num = torch.sum(x*y, dim=2)
    denom = x.norm(p=2, dim=2)*y.norm(p=2, dim=2)
    cos = torch.mean(num / denom)
    return 0.5*(1 - cos)
