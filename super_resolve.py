"""
Super-resolve images using the trained student model.
Supports single image or batch folder inference with optional sharpening.
"""
import os
import argparse
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from models.student import Student
from tqdm import tqdm
import time

def super_resolve(input_path, output_path, device=None, max_dim=1000, sharpen=False):
    """
    Super-resolve images using the trained student model.
    Args:
        input_path (str): Path to input image or folder.
        output_path (str): Path to output image or folder.
        device (str or None): 'cpu', 'cuda', or None for auto.
        max_dim (int): Maximum allowed width or height for input images.
        sharpen (bool): Whether to apply sharpening after upscaling.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = Student().to(device)
    ckpt_path = 'models/student_trained.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    tf = transforms.ToTensor()
    inv = transforms.ToPILImage()

    def process_image(img_path, out_path):
        # Load and process a single image
        img = Image.open(img_path).convert('RGB')
        if img.width > max_dim or img.height > max_dim:
            tqdm.write(f"⏭️ Skipping {img_path} (too large: {img.width}x{img.height})")
            return
        lr = tf(img).unsqueeze(0).to(device)
        start = time.time()
        with torch.no_grad():
            sr = model(lr)
        elapsed = time.time() - start
        if elapsed > 10:
            tqdm.write(f"⚠️ {img_path} took {elapsed:.1f}s to process.")
        sr_img = inv(sr.squeeze().cpu().clamp(0, 1))
        if sharpen:
            sr_img = sr_img.filter(ImageFilter.SHARPEN)
        sr_img.save(out_path)

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        image_files = [fname for fname in os.listdir(input_path)
                      if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        for fname in tqdm(image_files, desc="Super-resolving images"):
            in_file = os.path.join(input_path, fname)
            out_file = os.path.join(output_path, fname)
            try:
                process_image(in_file, out_file)
            except Exception as e:
                tqdm.write(f"❌ Error processing {in_file}: {e}")
        print(f"✅ Output saved to {output_path}")
    else:
        # Single image
        img = Image.open(input_path).convert('RGB')
        if img.width > max_dim or img.height > max_dim:
            print(f"⏭️ Skipping {input_path} (too large: {img.width}x{img.height})")
            return
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        lr = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            sr = model(lr)
        sr_img = inv(sr.squeeze().cpu().clamp(0, 1))
        if sharpen:
            sr_img = sr_img.filter(ImageFilter.SHARPEN)
        sr_img.save(output_path)
        print(f"✅ Output saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Super-resolve images using Student model')
    parser.add_argument('input', help='Path to input image or folder')
    parser.add_argument('output', help='Path to output image or folder')
    parser.add_argument('--device', default=None, help='Device to use: cpu or cuda (default: auto)')
    parser.add_argument('--max-dim', type=int, default=1000, help='Maximum allowed width or height for input images (default: 1000)')
    parser.add_argument('--sharpen', action='store_true', help='Apply sharpening after upscaling')
    args = parser.parse_args()
    super_resolve(args.input, args.output, args.device, args.max_dim, args.sharpen) 