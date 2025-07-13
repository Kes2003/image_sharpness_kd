"""
Optimized training script for DIV2K dataset to achieve SSIM scores of 0.9+.
Uses enhanced dataset, better loss functions, and optimized training parameters.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.student_enhanced import EnhancedStudent
from datasets.div2k_dataset import DIV2KDataset, DIV2KValidationDataset
from scripts.advanced_losses import CombinedLoss
from torchvision.utils import save_image
import torch.nn.functional as F
import time
from models.swinir_teacher import SwinIRTeacher
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_ssim_psnr(sr, hr):
    """Calculate SSIM and PSNR between SR and HR images. Downscale if too large for memory."""
    sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Ensure values are in [0, 1] range
    sr_np = np.clip(sr_np, 0, 1)
    hr_np = np.clip(hr_np, 0, 1)
    
    # Downscale if image is too large for SSIM (e.g., >1024x1024)
    max_dim = 1024
    h, w = sr_np.shape[:2]
    if h > max_dim or w > max_dim:
        from skimage.transform import resize
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        sr_np = resize(sr_np, (new_h, new_w), order=1, mode='reflect', anti_aliasing=True)
        hr_np = resize(hr_np, (new_h, new_w), order=1, mode='reflect', anti_aliasing=True)
    
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

def train_div2k():
    """Train on DIV2K dataset with optimized parameters for SSIM 0.9+."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available! Please check your PyTorch installation and GPU drivers.')
    device = torch.device('cuda')
    print(f'Using device: {device}')

    # Enhanced dataset with larger patches
    train_dataset = DIV2KDataset(
        lr_dir="data/div2k/train/LR",
        hr_dir="data/div2k/train/HR",
        patch_size=192,  # Larger patches for better training
        scale=4,
        augment=True
    )
    
    val_dataset = DIV2KValidationDataset(
        lr_dir="data/div2k/val/LR",
        hr_dir="data/div2k/val/HR",
        scale=4
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")

    # Enhanced student model
    student = EnhancedStudent().to(device)
    
    # Advanced loss function with much higher SSIM weight
    combined_loss = CombinedLoss(alpha=1.0, beta=0.1, gamma=0.3, device=device)  # gamma=0.3 for strong SSIM focus
    # Remove redundant MSE/L1 losses from main loss
    # mse_loss = torch.nn.MSELoss()
    # l1_loss = torch.nn.L1Loss()
    
    # Optimizer with better parameters
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    os.makedirs("results/train_outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Training parameters optimized for DIV2K
    epochs = 300  # Increased epochs for better convergence
    best_ssim = 0.0
    patience = 50  # Increased patience for early stopping
    patience_counter = 0

    print("🎯 Starting DIV2K training with knowledge distillation...")
    
    # Load teacher model
    teacher = SwinIRTeacher(device=device).to(device)
    teacher.eval()  # Freeze teacher
    print("📚 Teacher model loaded")

    # Knowledge distillation parameters
    alpha = 0.7  # Weight for ground truth loss
    beta = 0.3   # Weight for distillation loss
    temperature = 4.0  # Temperature for soft targets

    print(f"🚀 Training for {epochs} epochs with {len(train_dataset)} samples per epoch")
    print(f"📈 Target: SSIM > 0.9 on DIV2K validation set")

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_pbar):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            optimizer.zero_grad()
            # Forward pass
            sr_imgs = student(lr_imgs)
            # Teacher prediction (for knowledge distillation)
            with torch.no_grad():
                teacher_sr = teacher(lr_imgs)
            # Calculate combined loss (focus on SSIM)
            total_combined_loss, mse_loss_val, perceptual_loss_val, ssim_loss_val = combined_loss(sr_imgs, hr_imgs)
            # Knowledge distillation loss (keep as before)
            distill_loss = F.mse_loss(
                F.log_softmax(sr_imgs / temperature, dim=1),
                F.log_softmax(teacher_sr / temperature, dim=1)
            )
            # Main loss: focus on combined loss (SSIM) and distillation
            loss = alpha * total_combined_loss + beta * distill_loss
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MSE': f'{mse_loss_val.item():.4f}',
                'SSIM': f'{ssim_loss_val.item():.4f}',
                'Distill': f'{distill_loss.item():.4f}'
            })
        
        # Validation
        if (epoch + 1) % 5 == 0:
            student.eval()
            val_ssim = 0.0
            val_psnr = 0.0
            val_count = 0
            
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img = lr_img.to(device)
                    hr_img = hr_img.to(device)
                    
                    sr_img = student(lr_img)
                    
                    # Calculate metrics
                    ssim_val, psnr_val = calculate_ssim_psnr(sr_img, hr_img)
                    val_ssim += ssim_val
                    val_psnr += psnr_val
                    val_count += 1
                    
                    # Save sample outputs
                    if val_count <= 3:
                        save_image(sr_img, f"results/train_outputs/epoch_{epoch+1:03d}_sample_{val_count}.png")
            
            val_ssim /= val_count
            val_psnr /= val_count
            
            print(f"📊 Epoch {epoch+1}: SSIM={val_ssim:.4f}, PSNR={val_psnr:.2f}dB")
            
            # Save best model
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(student.state_dict(), "models/student_div2k_best.pt")
                patience_counter = 0
                print(f"🎉 New best SSIM: {best_ssim:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(student.state_dict(), f"models/student_div2k_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(student.state_dict(), "models/student_div2k_final.pt")
    print(f"✅ Training complete! Best SSIM: {best_ssim:.4f}")
    print(f"📁 Models saved in models/ directory")

if __name__ == "__main__":
    train_div2k() 