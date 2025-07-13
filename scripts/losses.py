import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = vgg19(weights='IMAGENET1K_V1').features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)

    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y))

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, device='cpu'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.perceptual = VGGPerceptualLoss(device=device)
        self.alpha = alpha  # MSE weight
        self.beta = beta    # Distillation weight
        self.gamma = gamma  # Perceptual weight

    def forward(self, pred, target, teacher_out):
        return (
            self.alpha * self.mse(pred, target) +
            self.beta * self.mse(pred, teacher_out) +
            self.gamma * self.perceptual(pred, target)
        )
