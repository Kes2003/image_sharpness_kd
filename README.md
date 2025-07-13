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

**Basic Training (MSE Loss Only)**

```bash
python main.py train --mode basic
```

**Knowledge Distillation Training**

```bash
python main.py train --mode kd
```

**Advanced Training (Full Pipeline)**

```bash
python main.py train --mode advanced
```

**DIV2K Optimized Training (Recommended)**

```bash
python scripts/train_div2k.py
```

### Evaluation

**Evaluate student model**

```bash
python main.py evaluate --model student
```

**Evaluate on DIV2K validation**

```bash
python scripts/evaluate_div2k.py --model student
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
├── models/                 # Model architectures and weights
│   ├── student.py         # Basic student model
│   ├── student_enhanced.py # Enhanced student model
│   └── swinir_teacher.py  # SwinIR teacher model
├── datasets/              # Dataset classes
│   ├── sharpness_dataset.py
│   └── div2k_dataset.py
├── scripts/               # Training and utility scripts
│   ├── train.py          # Basic training
│   ├── train_kd.py       # Knowledge distillation
│   ├── train_advanced.py # Advanced training
│   ├── train_div2k.py    # DIV2K optimized training
│   ├── evaluate.py       # Evaluation
│   └── download_div2k.py # Dataset download
├── data/                  # Dataset storage (not in repo)
├── results/              # Training outputs (not in repo)
├── main.py               # Main CLI interface
└── requirements.txt      # Dependencies
```

## 🔬 Technical Details

### Architecture

- **Student Model**: Lightweight EDSR-based architecture
- **Teacher Model**: SwinIR-M (state-of-the-art transformer)
- **Knowledge Distillation**: Soft target learning with temperature scaling
- **Loss Functions**: MSE + L1 + MS-SSIM + Distillation loss

### Training Strategy

1. **Data Augmentation**: Random flips, rotations, color jittering
2. **Patch-based Training**: 192x192 HR patches from DIV2K images
3. **Progressive Learning**: Multi-scale training with curriculum
4. **Early Stopping**: Prevents overfitting with patience mechanism

### Dataset

- **Training**: DIV2K 800 images (2048x1080 resolution)
- **Validation**: DIV2K 100 images
- **Scale Factor**: 4x upscaling (LR: 512x270 → HR: 2048x1080)

## 📈 Performance Optimization

### For SSIM > 0.9:

1. **Use DIV2K dataset**: Full 800 training images
2. **Larger patches**: 192x192 instead of 64x64
3. **Knowledge distillation**: Learn from SwinIR teacher
4. **Advanced loss functions**: Combined MSE + L1 + MS-SSIM
5. **Proper augmentation**: Random flips, rotations, color jittering

### Training Tips:

- **GPU Memory**: Use batch size 8 for 8GB+ GPUs
- **Learning Rate**: Start with 2e-4, use cosine annealing
- **Patience**: Allow 30+ epochs without improvement
- **Validation**: Check every 5 epochs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SwinIR](https://github.com/JingyunLiang/SwinIR) for the teacher model
- [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) for the student architecture
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset providers
- [BasicSR](https://github.com/XPixelGroup/BasicSR) framework

## 📞 Contact

- **Author**: Kesler Concesso
- **Email**: kesler.concesso@btech.christuniversity.in
- **GitHub**: [@Kes2003](https://github.com/Kes2003)

---

⭐ **Star this repository if you find it helpful!**
