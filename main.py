import argparse
import os
import sys
import subprocess
import shutil

# Helper functions for setup
def install_requirements():
    if not os.path.exists('requirements.txt'):
        print('requirements.txt not found!')
        return
    print('📦 Installing dependencies...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def download_edsr_weights():
    weights_path = os.path.join('models', 'edsr_x4.pt')
    if os.path.exists(weights_path):
        print('✅ EDSR weights already present.')
        return
    print('⬇️  Downloading EDSR x4 weights...')
    os.makedirs('models', exist_ok=True)
    import urllib.request
    url = 'https://github.com/sanghyun-son/EDSR-PyTorch/releases/download/EDSR_X4/edsr_x4-4f62e9ef.pt'
    urllib.request.urlretrieve(url, weights_path)
    print('✅ Downloaded EDSR weights.')

def download_div2k():
    # Download a small subset for demo if data/train/LR is empty
    lr_dir = 'data/train/LR'
    hr_dir = 'data/train/HR'
    if os.listdir(lr_dir) and os.listdir(hr_dir):
        print('✅ Training data already present.')
        return
    print('⬇️  Downloading sample DIV2K images for demo...')
    import zipfile
    import requests
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    # Download a few images from DIV2K (public domain)
    base_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/'
    lr_url = base_url + 'DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic_X4_part1.zip'
    hr_url = base_url + 'DIV2K_train_HR/DIV2K_train_HR_part1.zip'
    for url, out_dir in [(lr_url, lr_dir), (hr_url, hr_dir)]:
        local_zip = out_dir + '.zip'
        with requests.get(url, stream=True) as r:
            with open(local_zip, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(out_dir)
        os.remove(local_zip)
    print('✅ Downloaded sample DIV2K images.')

def setup():
    install_requirements()
    download_edsr_weights()
    download_div2k()

def run_train(mode='basic'):
    print(f'🚀 Starting {mode} training...')
    if mode == 'basic':
        subprocess.run([sys.executable, '-m', 'scripts.train'])
    elif mode == 'kd':
        subprocess.run([sys.executable, '-m', 'scripts.train_kd'])
    elif mode == 'advanced':
        subprocess.run([sys.executable, '-m', 'scripts.train_advanced'])

def run_evaluate(model='student'):
    print(f'🔍 Evaluating {model} model...')
    if model == 'student':
        subprocess.run([sys.executable, '-m', 'scripts.evaluate'])
    elif model == 'student_kd':
        # Update evaluate script to use different model
        subprocess.run([sys.executable, '-m', 'scripts.evaluate'], env={'MODEL_PATH': 'models/student_kd_trained.pt'})
    elif model == 'student_advanced':
        subprocess.run([sys.executable, '-m', 'scripts.evaluate'], env={'MODEL_PATH': 'models/student_advanced_best.pt'})

def run_infer(input_path, output_path, use_student, sharpen, max_dim=1000):
    print('🖼️  Running inference...')
    if use_student:
        args = [sys.executable, '-m', 'scripts.inference', input_path, output_path]
        if sharpen:
            args.append('--sharpen')
        args.extend(['--max-dim', str(max_dim)])
        subprocess.run(args)
    else:
        args = [sys.executable, 'super_resolve.py', input_path, output_path]
        if sharpen:
            args.append('--sharpen')
        args.extend(['--max-dim', str(max_dim)])
        subprocess.run(args)

def main():
    parser = argparse.ArgumentParser(description='Image Sharpness KD: Super-Resolution Pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_setup = subparsers.add_parser('setup', help='Install dependencies and download weights/dataset')

    parser_train = subparsers.add_parser('train', help='Train the student model')
    parser_train.add_argument('--mode', choices=['basic', 'kd', 'advanced'], default='basic', 
                             help='Training mode: basic (MSE only), kd (knowledge distillation), advanced (full pipeline)')

    parser_eval = subparsers.add_parser('evaluate', help='Evaluate the student model')
    parser_eval.add_argument('--model', choices=['student', 'student_kd', 'student_advanced'], default='student',
                            help='Which trained model to evaluate')

    parser_infer = subparsers.add_parser('infer', help='Run inference (upscale/sharpen images)')
    parser_infer.add_argument('input', help='Input image or folder')
    parser_infer.add_argument('output', help='Output image or folder')
    parser_infer.add_argument('--student', action='store_true', help='Use student model for inference')
    parser_infer.add_argument('--sharpen', action='store_true', help='Apply sharpening after upscaling')
    parser_infer.add_argument('--max-dim', type=int, default=1000, help='Maximum allowed width or height for input images (default: 1000)')

    args = parser.parse_args()

    if args.command == 'setup':
        setup()
    elif args.command == 'train':
        run_train(args.mode)
    elif args.command == 'evaluate':
        run_evaluate(args.model)
    elif args.command == 'infer':
        run_infer(args.input, args.output, args.student, args.sharpen, args.max_dim)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 