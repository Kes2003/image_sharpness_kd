import sys
import os
import torch

# Add BasicSR to the path
BASICSRC = os.path.join(os.path.dirname(__file__), '../external/BasicSR/BasicSR-master')
sys.path.append(os.path.abspath(BASICSRC))

try:
    from basicsr.archs.edsr_arch import EDSR
    print('Imported EDSR from BasicSR successfully.')
except Exception as e:
    print('Failed to import EDSR:', e)
    sys.exit(1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Creating EDSR model...')
    try:
        model = EDSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=256,
            num_block=32,
            upscale=4,
            res_scale=1.0,
            img_range=255.0,
            rgb_mean=(0.4488, 0.4371, 0.4040)
        ).to(device)
        print('Model created.')
        model.eval()
        # Pass a tensor of ones
        x = torch.ones(1, 3, 64, 64).to(device)
        print('Running inference...')
        with torch.no_grad():
            y = model(x)
        print('Output stats:', y.min().item(), y.max().item(), y.mean().item())
    except Exception as e:
        print('Error during model creation or inference:', e)

if __name__ == '__main__':
    main() 