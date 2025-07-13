import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x):
        return x + self.block(x) * self.res_scale

class Student(nn.Module):
    def __init__(self, scale=4, n_resblocks=16, n_feats=128, res_scale=1.0):
        super().__init__()
        self.scale = scale
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.res_scale = res_scale
        
        # define head module
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        
        # define body module
        self.body = nn.ModuleList([
            ResidualBlock(n_feats, res_scale) for _ in range(n_resblocks)
        ])
        self.body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        
        # define tail module - simple upsampling
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )
        
    def forward(self, x):
        x = self.head(x)
        res = x
        for block in self.body:
            res = block(res)
        res += x
        x = self.tail(res)
        return x
