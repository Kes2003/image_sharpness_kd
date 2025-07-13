"""
Inference script for the student super-resolution model.
Supports single image or batch folder inference with optional sharpening.
"""
import os
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from models.student import Student
import argparse
from tqdm import tqdm

def infer(input_path, output_path, device=None, sharpen=False, max_dim=1000):
    """
    Run inference using the trained student model.
    Args:
        input_path (str): Path to input image or folder.
        output_path (str): Path to output image or folder.
        device (str or None): 'cpu', 'cuda', or None for auto.
        sharpen (bool): Whether to apply sharpening after upscaling.
        max_dim (int): Maximum allowed width or height for input images.
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
        try:
            lr = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(lr)
            img_out = inv(out.squeeze().cpu().clamp(0,1))
            if sharpen:
                img_out = img_out.filter(ImageFilter.SHARPEN)
            img_out.save(out_path)
            # Clear GPU cache after each image
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                tqdm.write(f"❌ CUDA OOM for {img_path}, trying CPU...")
                try:
                    # Fallback to CPU
                    model_cpu = model.cpu()
                    lr = tf(img).unsqueeze(0)
                    with torch.no_grad():
                        out = model_cpu(lr)
                    img_out = inv(out.squeeze().clamp(0,1))
                    if sharpen:
                        img_out = img_out.filter(ImageFilter.SHARPEN)
                    img_out.save(out_path)
                    model.to(device)  # Move back to GPU
                    tqdm.write(f"✅ Processed {img_path} on CPU")
                except Exception as cpu_e:
                    tqdm.write(f"❌ CPU processing failed for {img_path}: {cpu_e}")
            else:
                tqdm.write(f"❌ Error processing {img_path}: {e}")

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        image_files = [fname for fname in os.listdir(input_path)
                      if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        for fname in tqdm(image_files, desc="Student Inference"):
            in_file = os.path.join(input_path, fname)
            out_file = os.path.join(output_path, fname)
            process_image(in_file, out_file)
        print(f"✅ Output saved to {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        process_image(input_path, output_path)
        print(f"✅ Output saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Student model inference')
    parser.add_argument('input', help='Input image or folder')
    parser.add_argument('output', help='Output image or folder')
    parser.add_argument('--device', default=None, help='Device to use: cpu or cuda (default: auto)')
    parser.add_argument('--sharpen', action='store_true', help='Apply sharpening after upscaling')
    parser.add_argument('--max-dim', type=int, default=1000, help='Maximum allowed width or height for input images (default: 1000)')
    args = parser.parse_args()
    infer(args.input, args.output, args.device, args.sharpen, args.max_dim)
