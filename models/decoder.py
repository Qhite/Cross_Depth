import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.LeakyReLU(),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.LeakyReLU(), 
        )


class Decoder(nn.Module): # Interpolation & Conv
    def __init__(self, channels=[]):
        super(Decoder, self).__init__()
        self.dim = 16
        chs = 288

        self.conv0 = nn.Sequential(
            depthwise(chs, 3),
            pointwise(chs, self.dim)
        )

        self.skip_conv1 = nn.Sequential(
            depthwise(channels[1], 3),
            pointwise(channels[1], self.dim),
            nn.Dropout(0.1)
        )
        self.res_conv1 = nn.Sequential(
            depthwise(self.dim *2, 3),
            pointwise(self.dim *2, self.dim),
        )

        self.skip_conv2 = nn.Sequential(
            depthwise(channels[2], 3),
            pointwise(channels[2], self.dim),
            nn.Dropout(0.1)
        )
        self.res_conv2 = nn.Sequential(
            depthwise(self.dim *2, 3),
            pointwise(self.dim *2, self.dim)
        )

        self.skip_conv3 = nn.Sequential(
            depthwise(channels[3], 3),
            pointwise(channels[3], self.dim),
            nn.Dropout(0.1)
        )
        self.res_conv3 = nn.Sequential(
            depthwise(self.dim *2, 3),
            pointwise(self.dim *2, self.dim),
        )

        self.skip_conv4 = nn.Sequential(
            depthwise(channels[4], 3),
            pointwise(channels[4], self.dim),
            nn.Dropout(0.1)
        )
        self.res_conv4 = nn.Sequential(
            depthwise(self.dim *2, 3),
            pointwise(self.dim *2, self.dim),
        )

    def forward(self, f):
        x = self.conv0(f[-1]) # Decoder input

        x = torch.cat( ( F.interpolate(x, size=f[-2].shape[-2:], mode='bilinear', align_corners=True), self.skip_conv1(f[-2]) ), 1 )
        x = self.res_conv1(x)

        x = torch.cat( ( F.interpolate(x, size=f[-3].shape[-2:], mode='bilinear', align_corners=True), self.skip_conv2(f[-3]) ), 1 )
        x = self.res_conv2(x)

        x = torch.cat( ( F.interpolate(x, size=f[-4].shape[-2:], mode='bilinear', align_corners=True), self.skip_conv3(f[-4]) ), 1 )
        x = self.res_conv3(x)

        x = torch.cat( ( F.interpolate(x, size=f[-5].shape[-2:], mode='bilinear', align_corners=True), self.skip_conv4(f[-5]) ), 1 )
        x = self.res_conv4(x)
        
        return x