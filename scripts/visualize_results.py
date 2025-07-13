import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def load_img(path):
    return np.array(Image.open(path).convert('RGB'))

def make_composite(lr, sr, hr, psnr_val, ssim_val, out_path):
    # Resize LR to match SR for fair comparison
    lr_up = np.array(Image.fromarray(lr).resize((sr.shape[1], sr.shape[0]), Image.BICUBIC))
    imgs = [lr_up, sr]
    labels = ["LR (bicubic)", "SR (output)"]
    if hr is not None:
        imgs.append(hr)
        labels.append("HR (ground truth)")
    composite = Image.new('RGB', (sr.shape[1]*len(imgs), sr.shape[0]+40), (255,255,255))
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for i, (img, label) in enumerate(zip(imgs, labels)):
        composite.paste(Image.fromarray(img), (i*sr.shape[1], 40))
        draw = ImageDraw.Draw(composite)
        draw.text((i*sr.shape[1]+10, 10), label, fill=(0,0,0), font=font)
    # Overlay metrics
    draw = ImageDraw.Draw(composite)
    draw.text((10, sr.shape[0]+10), f"PSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}", fill=(0,0,0), font=font)
    composite.save(out_path)

def main():
    parser = argparse.ArgumentParser(description="Visualize SR results with metrics.")
    parser.add_argument('--lr', required=True, help='Folder with LR images')
    parser.add_argument('--sr', required=True, help='Folder with SR images')
    parser.add_argument('--hr', default=None, help='Folder with HR images (optional)')
    parser.add_argument('--outdir', default='visualizations', help='Output folder for visualizations')
    parser.add_argument('--csv', default='visualizations/metrics.csv', help='CSV file to save metrics')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import csv
    with open(args.csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'psnr', 'ssim'])
        for fname in os.listdir(args.sr):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            sr_path = os.path.join(args.sr, fname)
            lr_path = os.path.join(args.lr, fname)
            hr_path = os.path.join(args.hr, fname) if args.hr else None
            if not os.path.exists(lr_path):
                continue
            sr_img = load_img(sr_path)
            lr_img = load_img(lr_path)
            hr_img = load_img(hr_path) if hr_path and os.path.exists(hr_path) else None
            # Compute metrics if HR available
            if hr_img is not None:
                # Resize SR to match HR dimensions for fair comparison
                sr_resized = np.array(Image.fromarray(sr_img).resize((hr_img.shape[1], hr_img.shape[0]), Image.BICUBIC))
                _psnr = psnr(hr_img, sr_resized, data_range=255)
                _ssim = ssim(hr_img, sr_resized, data_range=255, channel_axis=-1)
            else:
                _psnr = _ssim = float('nan')
            out_path = os.path.join(args.outdir, fname)
            make_composite(lr_img, sr_img, hr_img, _psnr, _ssim, out_path)
            writer.writerow([fname, _psnr, _ssim])

if __name__ == '__main__':
    main() 