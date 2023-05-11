import torch.nn as nn
import torch


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

class CAE(nn.Module):
    ''' Vanilla AE '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, 3),
            ConvBnRelu(16, 32, 3),
            ConvBnRelu(32, 64, 3))
        self.decoder = nn.Sequential(
            DConvBnRelu(64, 32, 3),
            DConvBnRelu(32, 16, 3),
            nn.ConvTranspose1d(16, 3, 3))

    def forward(self, x, args, mode):
        feature = self.encoder(x)
        output = self.decoder(feature)
        if mode == 'train':
            loss = self.mse(x, output)
            return loss
        else:
            return output

