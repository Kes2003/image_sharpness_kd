# Image Sharpness Knowledge Distillation (SR-KD)

A state-of-the-art super-resolution framework that uses knowledge distillation to achieve SSIM scores of 0.9+ on the DIV2K dataset. This project implements a lightweight student model that learns from a powerful SwinIR teacher model.

## 🎯 Features

- **High Performance**: Achieves SSIM > 0.9 on DIV2K validation set
- **Knowledge Distillation**: Student model learns from SwinIR teacher
- **Multiple Training Modes**: Basic, KD, and Advanced training pipelines
- **DIV2K Dataset Support**: Full support for the official DIV2K dataset
- **Easy-to-Use**: Simple command-line interface for training and inference

## 📊 Results

| Model              | SSIM      | PSNR      | Model Size |
| ------------------ | --------- | --------- | ---------- |
| Student (Basic)    | ~0.85     | ~28dB     | 23MB       |
| Student (KD)       | ~0.88     | ~29dB     | 23MB       |
| Student (Advanced) | **0.90+** | **30+dB** | 29MB       |
| SwinIR Teacher     | 0.95+     | 32+dB     | 164MB      |

## 🚀 Quick Start

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Kes2003/image_sharpness_kd.git
   cd image_sharpness_kd
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. **Download DIV2K dataset**

   ```bash
   python scripts/download_div2k.py
   ```

2. **Download pre-trained models**
   ```bash
   python main.py setup
   ```

### Training

**DIV2K Optimized Training (Recommended)**

```bash
python scripts/train_div2k.py
```

**Other Training Modes**

```bash
python scripts/train.py           # Basic training
python scripts/train_kd.py        # Knowledge distillation
python scripts/train_advanced.py  # Advanced training
```

### Evaluation

**Evaluate on DIV2K validation**

```bash
python scripts/evaluate_div2k.py --model student
```

**General Evaluation**

```bash
python scripts/evaluate.py
```

### Inference

**Upscale a single image**

```bash
python main.py infer input.jpg output.jpg --student --sharpen
```

**Upscale a folder**

```bash
python main.py infer input_folder/ output_folder/ --student --sharpen
```

## 📁 Project Structure

```
image-sharpness-kd/
├── models/                 # Model architectures
│   ├── student.py
│   ├── student_enhanced.py
│   └── swinir_teacher.py
├── scripts/                # Training and utility scripts
│   ├── train.py
│   ├── train_kd.py
│   ├── train_advanced.py
│   ├── train_div2k.py
│   ├── evaluate.py
│   ├── evaluate_div2k.py
│   ├── eval_folder.py
│   ├── inference.py
│   ├── advanced_losses.py
│   ├── losses.py
│   ├── download_div2k.py
│   ├── download_edsr_weights.py
│   ├── visualize_results.py
│   └── visualize_side_by_side.py
├── utils/                  # Utility functions
│   └── dataset.py
├── main.py                 # Main CLI interface
├── requirements.txt        # Dependencies
├── README.md
├── LICENSE
├── .gitignore
```

## 🧪 Technical Details

- **Student Model**: Lightweight EDSR-based architecture
- **Teacher Model**: SwinIR-M (state-of-the-art transformer)
- **Knowledge Distillation**: Soft target learning with temperature scaling
- **Loss Functions**: MSE + Perceptual + SSIM + Distillation loss
- **Data Augmentation**: Random flips, rotations, color jittering
- **Patch-based Training**: 192x192 HR patches from DIV2K images
- **Early Stopping**: Prevents overfitting with patience mechanism

## 📈 Performance Optimization

- Use DIV2K dataset: Full 800 training images
- Larger patches: 192x192
- Knowledge distillation: Learn from SwinIR teacher
- Advanced loss functions: Combined MSE + Perceptual + SSIM
- Proper augmentation: Random flips, rotations, color jittering

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SwinIR](https://github.com/JingyunLiang/SwinIR) for the teacher model
- [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) for the student architecture
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset providers
- [BasicSR](https://github.com/XPixelGroup/BasicSR) framework

## 📬 Contact

- **Author**: Kesler Concesso
- **Email**: kesler.concesso@btech.christuniversity.in
- **GitHub**: [@Kes2003](https://github.com/Kes2003)

---

⭐ **Star this repository if you find it helpful!**
