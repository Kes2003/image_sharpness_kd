import sys
import os
import torch
from types import SimpleNamespace

EDSR_SRC = os.path.join(os.path.dirname(__file__), '../external/EDSR-PyTorch/EDSR-PyTorch-master/src')
sys.path.append(os.path.abspath(EDSR_SRC))
from model.edsr import make_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = SimpleNamespace(
        scale=[4],
        n_resblocks=32,
        n_feats=256,
        res_scale=1.0,
        rgb_range=255,
        n_colors=3
    )
    model = make_model(config).to(device)
    weights_path = os.path.join(os.path.dirname(__file__), '../external/EDSR-PyTorch/EDSR-PyTorch-master/edsr_x4.pt')
    if not os.path.exists(weights_path):
        print(f"Missing weights: {weights_path}")
        return
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # Pass a tensor of ones
    x = torch.ones(1, 3, 64, 64).to(device) * 255.0
    with torch.no_grad():
        y = model(x)
    print('Output stats:', y.min().item(), y.max().item(), y.mean().item())

if __name__ == '__main__':
    main() 