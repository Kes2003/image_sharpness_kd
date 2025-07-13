import sys
import os
import torch

# Add BasicSR to the path
BASICSRC = os.path.join(os.path.dirname(__file__), '../external/BasicSR/BasicSR-master')
sys.path.append(os.path.abspath(BASICSRC))
from basicsr.archs.swinir_arch import SwinIR

class SwinIRTeacher(torch.nn.Module):
    """
    SwinIR Teacher model wrapper for knowledge distillation and evaluation.
    Loads the official SwinIR-M x4 weights (DIV2K, s48w8) and ensures config matches weights.
    """
    def __init__(self, device="cpu", weights_path=None):
        super().__init__()
        # Official SwinIR-M x4 config for 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth
        self.model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=48,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        ).to(device)
        self.model.eval()
        # Set default weights path if not provided
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(__file__),
                '../external/BasicSR/BasicSR-master/experiments/pretrained_models/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth'
            )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print("\n[ERROR] Failed to load SwinIR weights. This is usually due to a config mismatch.")
            print("Model config: embed_dim=180, img_size=48, num_heads=[6,6,6,6,6,6], depths=[6,6,6,6,6,6], window_size=8")
            print("If you downloaded a different SwinIR variant, please update the config accordingly.")
            print("Full error message:\n", e)
            raise
    def forward(self, x):
        return self.model(x) 