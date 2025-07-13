"""
Visualization script to create side-by-side images of LR, SR, and HR for comparison.
"""
import os
from PIL import Image
from tqdm import tqdm
import argparse

def visualize_side_by_side(lr_dir, sr_dir, hr_dir, output_dir):
    """
    Create and save side-by-side images (LR | SR | HR) for all matching filenames.
    Args:
        lr_dir (str): Directory with low-res images.
        sr_dir (str): Directory with super-resolved images.
        hr_dir (str): Directory with high-res ground truth images.
        output_dir (str): Directory to save side-by-side images.
    """
    os.makedirs(output_dir, exist_ok=True)
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    sr_files = sorted([f for f in os.listdir(sr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    common_files = sorted(set(lr_files) & set(sr_files) & set(hr_files))
    if not common_files:
        print(f"No matching files found in all three folders.")
        return
    for fname in tqdm(common_files, desc="Creating side-by-side images"):
        lr_path = os.path.join(lr_dir, fname)
        sr_path = os.path.join(sr_dir, fname)
        hr_path = os.path.join(hr_dir, fname)
        lr_img = Image.open(lr_path).convert('RGB')
        sr_img = Image.open(sr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        # Resize LR to match SR/HR height for fair comparison
        target_height = sr_img.height
        if lr_img.height != target_height:
            scale = target_height / lr_img.height
            new_width = int(lr_img.width * scale)
            lr_img = lr_img.resize((new_width, target_height), Image.BICUBIC)
        if hr_img.height != target_height:
            scale = target_height / hr_img.height
            new_width = int(hr_img.width * scale)
            hr_img = hr_img.resize((new_width, target_height), Image.BICUBIC)
        # Concatenate horizontally: LR | SR | HR
        total_width = lr_img.width + sr_img.width + hr_img.width
        combined = Image.new('RGB', (total_width, target_height))
        combined.paste(lr_img, (0, 0))
        combined.paste(sr_img, (lr_img.width, 0))
        combined.paste(hr_img, (lr_img.width + sr_img.width, 0))
        combined.save(os.path.join(output_dir, fname))
    print(f"Side-by-side images (LR | SR | HR) saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize LR, SR, and HR images side by side')
    parser.add_argument('--lr_dir', required=True, help='Directory with low-res images')
    parser.add_argument('--sr_dir', required=True, help='Directory with super-resolved images')
    parser.add_argument('--hr_dir', required=True, help='Directory with high-res ground truth images')
    parser.add_argument('--output_dir', required=True, help='Directory to save side-by-side images')
    args = parser.parse_args()
    visualize_side_by_side(args.lr_dir, args.sr_dir, args.hr_dir, args.output_dir) 