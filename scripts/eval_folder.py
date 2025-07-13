"""
Script to evaluate PSNR and SSIM between two folders of images (SR and HR).
"""
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from PIL import Image
from tqdm import tqdm

import argparse

def eval_folder(sr_dir, hr_dir, output_csv=None):
    """
    Evaluate PSNR and SSIM between all matching images in two folders.
    Args:
        sr_dir (str): Directory with super-resolved images.
        hr_dir (str): Directory with ground truth high-res images.
        output_csv (str or None): Optional path to save metrics as CSV.
    """
    sr_files = sorted([f for f in os.listdir(sr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    common_files = sorted(set(sr_files) & set(hr_files))
    if not common_files:
        print(f"No matching files found in {sr_dir} and {hr_dir}")
        return

    psnrs, ssims = [], []
    for fname in tqdm(common_files, desc="Evaluating"):
        sr_path = os.path.join(sr_dir, fname)
        hr_path = os.path.join(hr_dir, fname)
        sr_img = np.array(Image.open(sr_path).convert('RGB')).astype(np.float32) / 255.0
        hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(np.float32) / 255.0
        psnr = compare_psnr(hr_img, sr_img, data_range=1.0)
        ssim = compare_ssim(hr_img, sr_img, data_range=1.0, channel_axis=-1)
        psnrs.append(psnr)
        ssims.append(ssim)
    print(f"Average PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Average SSIM: {np.mean(ssims):.4f}")
    print(f"Processed {len(common_files)} images successfully")
    if output_csv:
        np.savetxt(output_csv, np.vstack([psnrs, ssims]).T, header='psnr,ssim', delimiter=',')
        print(f"Metrics saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PSNR/SSIM between two folders of images')
    parser.add_argument('--sr_dir', required=True, help='Directory with super-resolved images')
    parser.add_argument('--hr_dir', required=True, help='Directory with ground truth high-res images')
    parser.add_argument('--output_csv', default=None, help='Optional path to save metrics as CSV')
    args = parser.parse_args()
    eval_folder(args.sr_dir, args.hr_dir, args.output_csv) 