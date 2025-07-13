"""
Debug script to test SwinIR teacher model with synthetic data.
"""
import torch
import numpy as np
from PIL import Image
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.swinir_teacher import SwinIRTeacher

def test_swinir_with_synthetic():
    """Test SwinIR with synthetic data to understand input/output behavior."""
    
    # Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available!')
    device = torch.device('cuda')
    print(f'Using device: {device}')
    
    # Create SwinIR teacher
    model = SwinIRTeacher(device=device).to(device)
    model.eval()
    
    # Create synthetic test data
    # Test 1: Simple gradient pattern
    lr_gradient = torch.linspace(0, 1, 64*64).reshape(1, 64, 64).repeat(3, 1, 1)
    lr_gradient = lr_gradient.unsqueeze(0).to(device)  # [1, 3, 64, 64]
    
    print(f"Test 1 - Gradient input:")
    print(f"  Input shape: {lr_gradient.shape}")
    print(f"  Input range: [{lr_gradient.min().item():.4f}, {lr_gradient.max().item():.4f}]")
    
    with torch.no_grad():
        output = model(lr_gradient)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean: {output.mean().item():.4f}")
    
    # Test 2: Random noise
    lr_noise = torch.rand(1, 3, 64, 64).to(device)
    
    print(f"\nTest 2 - Random noise input:")
    print(f"  Input shape: {lr_noise.shape}")
    print(f"  Input range: [{lr_noise.min().item():.4f}, {lr_noise.max().item():.4f}]")
    
    with torch.no_grad():
        output = model(lr_noise)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean: {output.mean().item():.4f}")
    
    # Test 3: Real image from dataset
    from datasets.sharpness_dataset import SharpnessDataset
    from torch.utils.data import DataLoader
    
    dataset = SharpnessDataset(lr_dir='data/train/LR', hr_dir='data/train/HR')
    dataloader = DataLoader(dataset, batch_size=1)
    
    lr_real, hr_real = next(iter(dataloader))
    lr_real, hr_real = lr_real.to(device), hr_real.to(device)
    
    print(f"\nTest 3 - Real image input:")
    print(f"  LR input shape: {lr_real.shape}")
    print(f"  LR input range: [{lr_real.min().item():.4f}, {lr_real.max().item():.4f}]")
    print(f"  LR input mean: {lr_real.mean().item():.4f}")
    print(f"  HR target shape: {hr_real.shape}")
    print(f"  HR target range: [{hr_real.min().item():.4f}, {hr_real.max().item():.4f}]")
    print(f"  HR target mean: {hr_real.mean().item():.4f}")
    
    with torch.no_grad():
        output = model(lr_real)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean: {output.mean().item():.4f}")
        
        # Calculate PSNR and SSIM for this single image
        from torch.nn.functional import mse_loss
        from skimage.metrics import structural_similarity as ssim
        
        # Resize output to match HR
        from torch.nn.functional import interpolate
        output_resized = interpolate(output, size=hr_real.shape[-2:], mode='bicubic', align_corners=False)
        
        # Print more detailed statistics
        print(f"  Output resized range: [{output_resized.min().item():.4f}, {output_resized.max().item():.4f}]")
        print(f"  Output resized mean: {output_resized.mean().item():.4f}")
        
        psnr = 10 * torch.log10(1 / mse_loss(output_resized, hr_real)).item()
        print(f"  PSNR: {psnr:.2f} dB")
        
        # SSIM
        out_img = output_resized[0].cpu().permute(1, 2, 0).numpy()
        hr_img = hr_real[0].cpu().permute(1, 2, 0).numpy()
        ssim_val = ssim(out_img, hr_img, data_range=1.0, channel_axis=-1)
        print(f"  SSIM: {ssim_val:.4f}")
        
        # Save debug images
        from torchvision.utils import save_image
        save_image(output_resized.clamp(0, 1), "outputs/debug_output_normalized.png")
        save_image(hr_real.clamp(0, 1), "outputs/debug_hr_normalized.png")
        save_image(lr_real.clamp(0, 1), "outputs/debug_lr_normalized.png")

if __name__ == "__main__":
    test_swinir_with_synthetic() 