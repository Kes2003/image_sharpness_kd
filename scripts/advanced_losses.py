import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import numpy as np

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.device = device

    def forward(self, pred, target):
        pred_vgg = self.vgg(pred)
        target_vgg = self.vgg(target)
        return F.mse_loss(pred_vgg, target_vgg)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM loss. Uses torchmetrics if available, otherwise falls back to SSIMLoss.
    """
    def __init__(self, data_range=1.0, device='cpu'):
        super().__init__()
        self.data_range = data_range
        self.device = device
        if HAS_TORCHMETRICS:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        else:
            self.ms_ssim = None
            self.ssim = SSIMLoss()

    def forward(self, img1, img2):
        if HAS_TORCHMETRICS:
            # torchmetrics expects (N, C, H, W) and values in [0, 1]
            return 1 - self.ms_ssim(img1, img2)
        else:
            # Fallback to single-scale SSIM
            return self.ssim(img1, img2)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, device='cpu'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.perceptual = VGGPerceptualLoss(device=device)
        self.ssim = SSIMLoss()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # Perceptual weight
        self.gamma = gamma  # SSIM weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        perceptual_loss = self.perceptual(pred, target)
        ssim_loss = self.ssim(pred, target)
        
        total_loss = self.alpha * mse_loss + self.beta * perceptual_loss + self.gamma * ssim_loss
        return total_loss, mse_loss, perceptual_loss, ssim_loss 