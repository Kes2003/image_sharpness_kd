import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class EnhancedResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = ChannelAttention(channels)
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.attention(res)
        return x + res * self.res_scale

class EnhancedStudent(nn.Module):
    def __init__(self, scale=4, n_resblocks=20, n_feats=128, res_scale=0.1):
        super().__init__()
        self.scale = scale
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.res_scale = res_scale
        
        # Enhanced head module with more features
        self.head = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced body module with attention
        self.body = nn.ModuleList([
            EnhancedResidualBlock(n_feats, res_scale) for _ in range(n_resblocks)
        ])
        self.body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        
        # Enhanced tail module with progressive upsampling
        self.tail = nn.Sequential(
            # First upsampling (2x)
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            
            # Second upsampling (2x)
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            
            # Final output
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.head(x)
        res = x
        
        for block in self.body:
            res = block(res)
        
        res += x
        x = self.tail(res)
        return x 