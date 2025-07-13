# To run: python -m scripts.quick_test
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.student import Student
from datasets.sharpness_dataset import SharpnessDataset
from torchvision.utils import save_image
import torch.nn.functional as F
from models.swinir_teacher import SwinIRTeacher

def quick_test():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available! Please check your PyTorch installation and GPU drivers.')
    device = torch.device('cuda')
    print(f'Using device: {device}')

    train_dataset = SharpnessDataset(
        lr_dir="data/train/LR",
        hr_dir="data/train/HR"
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    print(f"📊 Training samples: {len(train_dataset)}")

    student = Student().to(device)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    os.makedirs("results/quick_test", exist_ok=True)

    # Load teacher model
    teacher = SwinIRTeacher(device=device).to(device)
    teacher.eval()
    print("📚 Teacher model loaded")

    # Knowledge distillation parameters
    alpha = 0.7  # Weight for ground truth loss
    beta = 0.3   # Weight for distillation loss

    print("🎯 Starting quick knowledge distillation test (10 epochs)...")
    
    for epoch in range(1, 11):
        student.train()
        total_loss = 0
        total_gt_loss = 0
        total_kd_loss = 0
        
        for i, (lr, hr) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            lr, hr = lr.to(device), hr.to(device)
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                t_out = teacher(lr)
                if t_out.shape[-2:] != hr.shape[-2:]:
                    t_out = F.interpolate(t_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)
            
            # Ground truth loss
            gt_loss = mse_loss(student(lr), hr)
            
            # Knowledge distillation loss
            kd_loss = mse_loss(student(lr), t_out)
            
            # Combined loss
            loss = alpha * gt_loss + beta * kd_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_gt_loss += gt_loss.item()
            total_kd_loss += kd_loss.item()
            
            if i == 0:
                save_image(student(lr), f"results/quick_test/epoch_{epoch:02d}.png")
        
        avg_loss = total_loss / len(train_loader)
        avg_gt_loss = total_gt_loss / len(train_loader)
        avg_kd_loss = total_kd_loss / len(train_loader)
        
        print(f"📉 Epoch {epoch} | Total Loss: {avg_loss:.4f} | GT Loss: {avg_gt_loss:.4f} | KD Loss: {avg_kd_loss:.4f}")
    
    torch.save(student.state_dict(), "models/student_quick_test.pt")
    print("✅ Quick test complete. Model saved at models/student_quick_test.pt")

if __name__ == "__main__":
    quick_test() 