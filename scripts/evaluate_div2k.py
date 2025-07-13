"""
Evaluation script for DIV2K validation set.
Monitors SSIM and PSNR progress during training.
"""
import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.div2k_dataset import DIV2KValidationDataset
from models.student_enhanced import EnhancedStudent
from models.swinir_teacher import SwinIRTeacher
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse

def calculate_ssim_psnr(sr, hr):
    """Calculate SSIM and PSNR between SR and HR images."""
    sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Ensure values are in [0, 1] range
    sr_np = np.clip(sr_np, 0, 1)
    hr_np = np.clip(hr_np, 0, 1)
    
    # Get image dimensions
    h, w = sr_np.shape[:2]
    
    # Calculate SSIM with appropriate window size
    if h < 7 or w < 7:
        # For very small images, use smaller window or skip SSIM
        win_size = min(3, h, w)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            ssim_val = 0.0  # Skip SSIM for very small images
        else:
            ssim_val = ssim(sr_np, hr_np, win_size=win_size, channel_axis=2, data_range=1.0)
    else:
        # Use default window size for larger images
        ssim_val = ssim(sr_np, hr_np, channel_axis=2, data_range=1.0)
    
    # Calculate PSNR
    psnr_val = psnr(hr_np, sr_np, data_range=1.0)
    
    return ssim_val, psnr_val

def evaluate_div2k(model_type='student', model_path=None):
    """Evaluate model on DIV2K validation set."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available!')
    device = torch.device('cuda')
    print(f'Using device: {device}')
    
    # Load model
    if model_type == 'student':
        model = EnhancedStudent().to(device)
        if model_path is None:
            model_path = 'models/student_div2k_best.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded student model from {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path}")
            return
    elif model_type == 'teacher':
        model = SwinIRTeacher(device=device).to(device)
        print("✅ Loaded teacher model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    
    # Load DIV2K validation dataset
    val_dataset = DIV2KValidationDataset(
        lr_dir="data/div2k/val/LR",
        hr_dir="data/div2k/val/HR",
        scale=4
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"📊 Evaluating on {len(val_dataset)} validation images...")
    
    # Evaluation
    total_ssim = 0.0
    total_psnr = 0.0
    ssim_values = []
    psnr_values = []
    
    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(val_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            
            sr_img = model(lr_img)
            
            # Calculate metrics
            ssim_val, psnr_val = calculate_ssim_psnr(sr_img, hr_img)
            total_ssim += ssim_val
            total_psnr += psnr_val
            ssim_values.append(ssim_val)
            psnr_values.append(psnr_val)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(val_dataset)} images...")
    
    # Calculate averages
    avg_ssim = total_ssim / len(val_dataset)
    avg_psnr = total_psnr / len(val_dataset)
    
    # Calculate standard deviations
    ssim_std = np.std(ssim_values)
    psnr_std = np.std(psnr_values)
    
    print(f"\n📊 DIV2K Validation Results ({model_type.upper()}):")
    print(f"   SSIM: {avg_ssim:.4f} ± {ssim_std:.4f}")
    print(f"   PSNR: {avg_psnr:.2f} ± {psnr_std:.2f} dB")
    print(f"   Target SSIM: 0.9+")
    
    if avg_ssim >= 0.9:
        print(f"🎉 SUCCESS! SSIM {avg_ssim:.4f} >= 0.9")
    else:
        print(f"📈 Progress: SSIM {avg_ssim:.4f} (need {0.9 - avg_ssim:.4f} more)")
    
    return avg_ssim, avg_psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on DIV2K validation set')
    parser.add_argument('--model', choices=['student', 'teacher'], default='student',
                       help='Which model to evaluate')
    parser.add_argument('--path', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    evaluate_div2k(args.model, args.path) 