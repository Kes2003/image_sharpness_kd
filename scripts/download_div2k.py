"""
Download and prepare DIV2K dataset for super-resolution training.
DIV2K is the official dataset used to train SwinIR and other SOTA models.
"""
import os
import requests
import zipfile
import shutil
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import argparse

def download_file(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def create_lr_hr_pairs(hr_dir, lr_dir, scale_factor=4, patch_size=256):
    """Create LR/HR pairs from high-resolution images."""
    os.makedirs(lr_dir, exist_ok=True)
    
    # Get all HR images
    hr_files = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Creating LR/HR pairs from {len(hr_files)} HR images...")
    
    for hr_file in tqdm(hr_files, desc="Processing images"):
        hr_path = os.path.join(hr_dir, hr_file)
        
        # Load HR image
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Create patches if image is larger than patch_size
        if hr_img.width >= patch_size and hr_img.height >= patch_size:
            # Create multiple patches from each HR image
            for i in range(0, hr_img.height - patch_size + 1, patch_size // 2):  # 50% overlap
                for j in range(0, hr_img.width - patch_size + 1, patch_size // 2):
                    # Extract HR patch
                    hr_patch = hr_img.crop((j, i, j + patch_size, i + patch_size))
                    
                    # Create LR patch by downscaling
                    lr_size = patch_size // scale_factor
                    lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)
                    
                    # Save patches
                    patch_name = f"{os.path.splitext(hr_file)[0]}_patch_{i}_{j}"
                    hr_patch.save(os.path.join(hr_dir, f"{patch_name}.png"))
                    lr_patch.save(os.path.join(lr_dir, f"{patch_name}.png"))
            
            # Remove original large image
            os.remove(hr_path)
        else:
            # For smaller images, just create one LR/HR pair
            lr_size = (hr_img.width // scale_factor, hr_img.height // scale_factor)
            lr_img = hr_img.resize(lr_size, Image.BICUBIC)
            
            # Save with new names
            base_name = os.path.splitext(hr_file)[0]
            hr_img.save(os.path.join(hr_dir, f"{base_name}_hr.png"))
            lr_img.save(os.path.join(lr_dir, f"{base_name}_lr.png"))
            
            # Remove original
            os.remove(hr_path)

def setup_div2k_dataset():
    """Download and setup DIV2K dataset."""
    
    # Create directories
    os.makedirs("data/div2k", exist_ok=True)
    os.makedirs("data/div2k/train/HR", exist_ok=True)
    os.makedirs("data/div2k/train/LR", exist_ok=True)
    os.makedirs("data/div2k/val/HR", exist_ok=True)
    os.makedirs("data/div2k/val/LR", exist_ok=True)
    
    # DIV2K dataset URLs (these are the official URLs)
    div2k_urls = {
        "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "train_lr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
        "val_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "val_lr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
    }
    
    print("DIV2K Dataset Setup")
    print("===================")
    print("This will download ~11GB of data.")
    print("The DIV2K dataset contains 1000 high-quality 2K images.")
    print("It's the official dataset used to train SwinIR and other SOTA models.")
    print()
    
    # Check if files already exist
    existing_files = []
    for name, url in div2k_urls.items():
        filename = f"data/div2k/{name}.zip"
        if os.path.exists(filename):
            existing_files.append(name)
    
    if existing_files:
        print(f"Found existing files: {existing_files}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Using existing files.")
        else:
            for name in existing_files:
                filename = f"data/div2k/{name}.zip"
                os.remove(filename)
    
    # Download files
    for name, url in div2k_urls.items():
        filename = f"data/div2k/{name}.zip"
        if not os.path.exists(filename):
            print(f"Downloading {name}...")
            try:
                download_file(url, filename)
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                print("You may need to download manually from:")
                print(f"  {url}")
                continue
    
    # Extract files
    print("\nExtracting files...")
    for name in div2k_urls.keys():
        filename = f"data/div2k/{name}.zip"
        if os.path.exists(filename):
            print(f"Extracting {name}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall("data/div2k/")
    
    # Organize files
    print("\nOrganizing files...")
    
    # Move training files
    if os.path.exists("data/div2k/DIV2K_train_HR"):
        for file in os.listdir("data/div2k/DIV2K_train_HR"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(
                    f"data/div2k/DIV2K_train_HR/{file}",
                    f"data/div2k/train/HR/{file}"
                )
    
    if os.path.exists("data/div2k/DIV2K_train_LR_bicubic/X4"):
        for file in os.listdir("data/div2k/DIV2K_train_LR_bicubic/X4"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(
                    f"data/div2k/DIV2K_train_LR_bicubic/X4/{file}",
                    f"data/div2k/train/LR/{file}"
                )
    
    # Move validation files
    if os.path.exists("data/div2k/DIV2K_valid_HR"):
        for file in os.listdir("data/div2k/DIV2K_valid_HR"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(
                    f"data/div2k/DIV2K_valid_HR/{file}",
                    f"data/div2k/val/HR/{file}"
                )
    
    if os.path.exists("data/div2k/DIV2K_valid_LR_bicubic/X4"):
        for file in os.listdir("data/div2k/DIV2K_valid_LR_bicubic/X4"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(
                    f"data/div2k/DIV2K_valid_LR_bicubic/X4/{file}",
                    f"data/div2k/val/LR/{file}"
                )
    
    # Clean up
    print("\nCleaning up...")
    for folder in ["DIV2K_train_HR", "DIV2K_train_LR_bicubic", "DIV2K_valid_HR", "DIV2K_valid_LR_bicubic"]:
        if os.path.exists(f"data/div2k/{folder}"):
            shutil.rmtree(f"data/div2k/{folder}")
    
    # Remove zip files
    for name in div2k_urls.keys():
        filename = f"data/div2k/{name}.zip"
        if os.path.exists(filename):
            os.remove(filename)
    
    # Create patches for training (optional)
    print("\nCreating training patches...")
    create_lr_hr_pairs(
        "data/div2k/train/HR",
        "data/div2k/train/LR",
        scale_factor=4,
        patch_size=256
    )
    
    # Print statistics
    train_hr_count = len([f for f in os.listdir("data/div2k/train/HR") if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    train_lr_count = len([f for f in os.listdir("data/div2k/train/LR") if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    val_hr_count = len([f for f in os.listdir("data/div2k/val/HR") if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    val_lr_count = len([f for f in os.listdir("data/div2k/val/LR") if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nDataset Statistics:")
    print(f"  Training HR images: {train_hr_count}")
    print(f"  Training LR images: {train_lr_count}")
    print(f"  Validation HR images: {val_hr_count}")
    print(f"  Validation LR images: {val_lr_count}")
    
    print(f"\nDIV2K dataset setup complete!")
    print(f"Dataset location: data/div2k/")
    print(f"\nTo use with your project:")
    print(f"  1. Copy files: data/div2k/train/* -> data/train/*")
    print(f"  2. Copy files: data/div2k/val/* -> data/val/*")
    print(f"  3. Run evaluation: python -m scripts.evaluate --model teacher")

if __name__ == "__main__":
    setup_div2k_dataset() 