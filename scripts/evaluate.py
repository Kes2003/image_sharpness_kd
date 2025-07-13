"""
Evaluation script for the student or teacher super-resolution model.
Computes PSNR and SSIM on a dataset and saves metrics to CSV.
"""
import torch, os, numpy as np
from torch.nn.functional import mse_loss, interpolate
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from datasets.sharpness_dataset import SharpnessDataset
from models.student import Student
import argparse
from torchvision.utils import save_image
from models.swinir_teacher import SwinIRTeacher

def evaluate(model_type='student'):
    """
    Evaluate the student or teacher model on paired LR/HR images.
    Computes PSNR and SSIM for each image and saves results to outputs/eval_metrics.csv.
    """
    # Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available! Please check your PyTorch installation and GPU drivers.')
    device = torch.device('cuda')
    print(f'Using device: {device}')
    if model_type == 'student':
        model = Student().to(device)
        ckpt_path = 'models/student_trained.pt'
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    elif model_type == 'teacher':
        model = SwinIRTeacher(device=device).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.eval()

    # Prepare dataset and dataloader
    ds = SharpnessDataset(lr_dir='data/train/LR', hr_dir='data/train/HR')
    loader = DataLoader(ds, batch_size=1)
    ps, ss = [], []
    debug_count = 0
    with torch.no_grad():
        for idx, (lr, hr) in enumerate(loader):
            print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
            max_dim = 512  # Prevent OOM on large images
            if lr.shape[2] > max_dim or lr.shape[3] > max_dim:
                print(f"Skipping image with shape {lr.shape} (too large for evaluation)")
                continue
            try:
                lr, hr = lr.to(device), hr.to(device)
                out = model(lr)
                # Resize output to match HR dimensions for fair evaluation
                out_resized = interpolate(out, size=hr.shape[-2:], mode='bicubic', align_corners=False)
                # For teacher, output and HR are in [0, 1]; no conversion needed
                if model_type == 'teacher':
                    # SwinIR outputs are already in [0, 1] range
                    pass
                # Debug: print stats for first 3 images
                if debug_count < 3:
                    print(f"out_resized shape: {out_resized.shape}, hr shape: {hr.shape}")
                    print(f"out_resized min: {out_resized.min().item():.4f}, max: {out_resized.max().item():.4f}, mean: {out_resized.mean().item():.4f}")
                    print(f"hr min: {hr.min().item():.4f}, max: {hr.max().item():.4f}, mean: {hr.mean().item():.4f}")
                    save_image(out_resized.clamp(0, 1), f"outputs/debug_out_{idx}.png")
                    save_image(hr.clamp(0, 1), f"outputs/debug_hr_{idx}.png")
                    debug_count += 1
                # Compute PSNR
                ps.append(10*torch.log10(1/mse_loss(out_resized,hr)).item())
                # Compute SSIM for each image in the batch
                for i in range(out_resized.shape[0]):
                    out_img = out_resized[i].cpu().permute(1, 2, 0).numpy()
                    hr_img = hr[i].cpu().permute(1, 2, 0).numpy()
                    h, w = out_img.shape[:2]
                    min_dim = min(h, w)
                    win_size = min(7, min_dim)
                    if win_size % 2 == 0:
                        win_size -= 1
                    if win_size < 3:
                        print(f"Skipping SSIM for image with size {h}x{w} (too small for SSIM)")
                        continue
                    ss.append(
                        ssim(
                            out_img,
                            hr_img,
                            data_range=1.0,
                            channel_axis=-1,
                            win_size=win_size
                        )
                    )
                # Clear GPU cache after each image
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM for image with shape {lr.shape}, skipping...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error processing image: {e}")
                    continue
    # Print and save results
    if ps:
        print(f"Average PSNR: {sum(ps)/len(ps):.2f} dB")
    if ss:
        print(f"Average SSIM: {sum(ss)/len(ss):.4f}")
    print(f"Processed {len(ps)} images successfully")
    np.savetxt('outputs/eval_metrics.csv', np.vstack([ps,ss]).T,
               header='psnr,ssim', delimiter=',')

# To run: python -m scripts.evaluate --model teacher
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='student', help='student or teacher')
    args = parser.parse_args()
    os.makedirs('outputs', exist_ok=True)
    evaluate(model_type=args.model)
