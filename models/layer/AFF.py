import torch
import torch.nn as nn


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual,fuse=False):
        B, C, H, W = x.shape
        B, L, C = residual.shape
        assert L == H * W, "input feature has wrong size"
        residual = residual.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        if fuse:
            xo = 2 * x * wei + 2 * residual * (1 - wei)
            xo = xo.permute(0, 2, 3, 1).contiguous()
            xo = xo.view(B, -1, C)
            return xo
        else:
            x, residual =  x * wei,residual * (1 - wei)
            residual = residual.permute(0, 2, 3, 1).contiguous()
            residual = residual.view(B, -1, C)
            return  x, residual
