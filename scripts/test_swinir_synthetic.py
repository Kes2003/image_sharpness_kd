"""
Test SwinIR with synthetic LR/HR pairs to verify model performance.
"""
import torch
import numpy as np
from PIL import Image
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.swinir_teacher import SwinIRTeacher
from torch.nn.functional import mse_loss, interpolate
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image

def test_swinir_synthetic():
    """Test SwinIR with synthetic data that has full dynamic range."""
    
    # Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available!')
    device = torch.device('cuda')
    print(f'Using device: {device}')
    
    # Create SwinIR teacher
    model = SwinIRTeacher(device=device).to(device)
    model.eval()
    
    # Create synthetic test data with full dynamic range
    # Test 1: Create a synthetic LR image with full [0, 1] range
    lr_synthetic = torch.zeros(1, 3, 64, 64)
    
    # Create a gradient pattern
    for i in range(64):
        for j in range(64):
            # Red channel: horizontal gradient
            lr_synthetic[0, 0, i, j] = i / 63.0
            # Green channel: vertical gradient  
            lr_synthetic[0, 1, i, j] = j / 63.0
            # Blue channel: diagonal gradient
            lr_synthetic[0, 2, i, j] = (i + j) / 126.0
    
    lr_synthetic = lr_synthetic.to(device)
    
    # Create corresponding HR target (4x upscaled)
    hr_synthetic = torch.zeros(1, 3, 256, 256)
    for i in range(256):
        for j in range(256):
            # Red channel: horizontal gradient
            hr_synthetic[0, 0, i, j] = i / 255.0
            # Green channel: vertical gradient
            hr_synthetic[0, 1, i, j] = j / 255.0
            # Blue channel: diagonal gradient
            hr_synthetic[0, 2, i, j] = (i + j) / 510.0
    
    hr_synthetic = hr_synthetic.to(device)
    
    print(f"Synthetic test:")
    print(f"  LR input shape: {lr_synthetic.shape}")
    print(f"  LR input range: [{lr_synthetic.min().item():.4f}, {lr_synthetic.max().item():.4f}]")
    print(f"  LR input mean: {lr_synthetic.mean().item():.4f}")
    print(f"  HR target shape: {hr_synthetic.shape}")
    print(f"  HR target range: [{hr_synthetic.min().item():.4f}, {hr_synthetic.max().item():.4f}]")
    print(f"  HR target mean: {hr_synthetic.mean().item():.4f}")
    
    with torch.no_grad():
        output = model(lr_synthetic)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean: {output.mean().item():.4f}")
        
        # Resize output to match HR
        output_resized = interpolate(output, size=hr_synthetic.shape[-2:], mode='bicubic', align_corners=False)
        
        # Calculate PSNR and SSIM
        psnr = 10 * torch.log10(1 / mse_loss(output_resized, hr_synthetic)).item()
        print(f"  PSNR: {psnr:.2f} dB")
        
        # SSIM
        out_img = output_resized[0].cpu().permute(1, 2, 0).numpy()
        hr_img = hr_synthetic[0].cpu().permute(1, 2, 0).numpy()
        ssim_val = ssim(out_img, hr_img, data_range=1.0, channel_axis=-1)
        print(f"  SSIM: {ssim_val:.4f}")
        
        # Save debug images
        save_image(output_resized.clamp(0, 1), "outputs/synthetic_output.png")
        save_image(hr_synthetic.clamp(0, 1), "outputs/synthetic_hr.png")
        save_image(lr_synthetic.clamp(0, 1), "outputs/synthetic_lr.png")
        
        print(f"\nDebug images saved to outputs/")
        print(f"  - synthetic_lr.png: Input 64x64 image")
        print(f"  - synthetic_hr.png: Target 256x256 image") 
        print(f"  - synthetic_output.png: SwinIR output 256x256 image")

if __name__ == "__main__":
    test_swinir_synthetic() 