# To run: python -m scripts.train_advanced
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.student_enhanced import EnhancedStudent
from datasets.sharpness_dataset import SharpnessDataset
from scripts.advanced_losses import CombinedLoss
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
import time
from models.swinir_teacher import SwinIRTeacher

def train_advanced():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available! Please check your PyTorch installation and GPU drivers.')
    device = torch.device('cuda')
    print(f'Using device: {device}')

    train_dataset = SharpnessDataset(
        lr_dir="data/div2k/train/LR",
        hr_dir="data/div2k/train/HR"
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    print(f"📊 Training samples: {len(train_dataset)}")

    # Enhanced student model
    student = EnhancedStudent().to(device)
    
    # Advanced loss function
    combined_loss = CombinedLoss(alpha=1.0, beta=0.1, gamma=0.1, device=device)
    mse_loss = torch.nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    os.makedirs("results/train_outputs", exist_ok=True)

    # Training parameters
    alpha = 0.6  # Weight for ground truth loss
    beta = 0.4   # Weight for distillation loss
    temperature = 4.0  # Temperature for soft targets

    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    print("🎯 Starting advanced training with knowledge distillation...")
    
    # Load teacher model
    teacher = SwinIRTeacher(device=device).to(device)
    teacher.eval()  # Freeze teacher
    print("📚 Teacher model loaded")

    for epoch in range(1, 101):
        student.train()
        total_loss = 0
        total_gt_loss = 0
        total_kd_loss = 0
        total_combined_loss = 0
        
        epoch_start = time.time()
        
        for i, (lr, hr) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            lr, hr = lr.to(device), hr.to(device)
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                t_out = teacher(lr)
                if t_out.shape[-2:] != hr.shape[-2:]:
                    t_out = F.interpolate(t_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)
            
            # Ground truth loss (combined loss)
            gt_loss, mse_comp, perceptual_comp, ssim_comp = combined_loss(student(lr), hr)
            
            # Knowledge distillation loss
            kd_loss = mse_loss(student(lr), t_out)
            
            # Combined loss
            loss = alpha * gt_loss + beta * kd_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_gt_loss += gt_loss.item()
            total_kd_loss += kd_loss.item()
            total_combined_loss += gt_loss.item()
            
            if i == 0 and epoch % 10 == 0:
                save_image(student(lr), f"results/train_outputs/epoch_{epoch:02d}.png")
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        avg_gt_loss = total_gt_loss / len(train_loader)
        avg_kd_loss = total_kd_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)
        
        print(f"📉 Epoch {epoch} | Total Loss: {avg_loss:.4f} | GT Loss: {avg_gt_loss:.4f} | KD Loss: {avg_kd_loss:.4f} | Combined Loss: {avg_combined_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), "models/student_advanced_best.pt")
            patience_counter = 0
            print(f"💾 New best model saved! Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break
    
    # Save final model
    torch.save(student.state_dict(), "models/student_advanced_final.pt")
    print("✅ Advanced training complete!")
    print(f"📁 Best model: models/student_advanced_best.pt")
    print(f"📁 Final model: models/student_advanced_final.pt")

if __name__ == "__main__":
    train_advanced() 