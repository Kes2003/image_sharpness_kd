"""
Training script for the student super-resolution model.
Trains on paired LR/HR images using MSE and MS-SSIM loss.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.student import Student
from datasets.sharpness_dataset import SharpnessDataset
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
from scripts.advanced_losses import MSSSIMLoss

# To run: python -m scripts.train
def train(finetune=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available! Please check your PyTorch installation and GPU drivers.')
    device = torch.device('cuda')
    print(f'Using device: {device}')

    # Prepare dataset and dataloader
    train_dataset = SharpnessDataset(
        lr_dir="data/div2k/train/LR",
        hr_dir="data/div2k/train/HR"
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    print(f"📊 Training samples: {len(train_dataset)}")

    # Initialize model, loss functions, optimizer, and scheduler
    student = Student().to(device)
    mse_loss = torch.nn.MSELoss()
    ms_ssim_loss = MSSSIMLoss(device=device)
    lr = 1e-4 if not finetune else 1e-5
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs("results/train_outputs", exist_ok=True)

    for epoch in range(1, 101):
        student.train()
        total_loss = 0
        for i, (lr_img, hr_img) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            s_out = student(lr_img)
            # Ensure output matches HR size
            if s_out.shape[-2:] != hr_img.shape[-2:]:
                s_out = F.interpolate(s_out, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
            # Compute combined loss (MS-SSIM-focused)
            loss_mse = mse_loss(s_out, hr_img)
            loss_ms_ssim = ms_ssim_loss(s_out, hr_img)
            loss = 0.2 * loss_mse + 1.0 * loss_ms_ssim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Save sample output every 10 epochs
            if i == 0 and epoch % 10 == 0:
                save_image(s_out, f"results/train_outputs/epoch_{epoch:02d}.png")
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    # Save the trained model
    torch.save(student.state_dict(), "models/student_trained.pt")
    print("✅ Training complete. Model saved at models/student_trained.pt")

if __name__ == "__main__":
    # First train with default lr, then optionally fine-tune with lower lr
    train(finetune=False)
    # To fine-tune, run: train(finetune=True)
