import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

def check_training_data():
    lr_dir = "data/train/LR"
    hr_dir = "data/train/HR"
    
    print("🔍 Checking training data...")
    
    # Check if directories exist
    if not os.path.exists(lr_dir):
        print(f"❌ LR directory not found: {lr_dir}")
        return
    if not os.path.exists(hr_dir):
        print(f"❌ HR directory not found: {hr_dir}")
        return
    
    # Get file lists
    lr_files = [f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    hr_files = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📊 Found {len(lr_files)} LR images and {len(hr_files)} HR images")
    
    # Check matching files
    matching_files = set(lr_files) & set(hr_files)
    print(f"📊 {len(matching_files)} matching files found")
    
    if len(matching_files) == 0:
        print("❌ No matching LR/HR pairs found!")
        return
    
    # Check a few sample pairs
    sample_files = list(matching_files)[:3]
    
    for filename in sample_files:
        lr_path = os.path.join(lr_dir, filename)
        hr_path = os.path.join(hr_dir, filename)
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        print(f"\n📸 {filename}:")
        print(f"  LR size: {lr_img.size}")
        print(f"  HR size: {hr_img.size}")
        
        # Check if they're actually different
        if lr_img.size == hr_img.size:
            print(f"  ⚠️  WARNING: LR and HR have same size!")
        
        # Convert to tensor and check ranges
        tf = transforms.ToTensor()
        lr_tensor = tf(lr_img)
        hr_tensor = tf(hr_img)
        
        print(f"  LR range: [{lr_tensor.min():.3f}, {lr_tensor.max():.3f}]")
        print(f"  HR range: [{hr_tensor.min():.3f}, {hr_tensor.max():.3f}]")
        
        # Check if they're identical
        if torch.allclose(lr_tensor, hr_tensor, atol=1e-6):
            print(f"  ⚠️  WARNING: LR and HR are identical!")
    
    print(f"\n✅ Data check complete. Found {len(matching_files)} valid pairs.")

if __name__ == "__main__":
    check_training_data() 