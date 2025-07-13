import os
from PIL import Image
import shutil

def fix_training_data():
    print("🔧 Fixing training data...")
    
    # Use images from data/raw as HR (high resolution)
    hr_source_dir = "data/raw"
    lr_dir = "data/train/LR"
    hr_dir = "data/train/HR"
    
    # Create directories if they don't exist
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
    
    # Process each image in data/raw
    raw_files = [f for f in os.listdir(hr_source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    print(f"📊 Found {len(raw_files)} images in data/raw")
    
    processed_count = 0
    for filename in raw_files:
        raw_path = os.path.join(hr_source_dir, filename)
        hr_path = os.path.join(hr_dir, filename)
        lr_path = os.path.join(lr_dir, filename)
        
        try:
            # Load original image
            hr_img = Image.open(raw_path).convert('RGB')
            
            # Only use images that are at least 256x256 for HR
            if hr_img.width < 256 or hr_img.height < 256:
                print(f"⏭️ Skipping {filename} (too small for HR: {hr_img.size})")
                continue
            
            # Create LR image (4x smaller)
            lr_width = hr_img.width // 4
            lr_height = hr_img.height // 4
            
            # Downsample using BICUBIC (this creates the "low resolution" version)
            lr_img = hr_img.resize((lr_width, lr_height), Image.BICUBIC)
            
            # Save HR image (copy original)
            hr_img.save(hr_path)
            
            # Save LR image (downsampled)
            lr_img.save(lr_path)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"✅ Processed {processed_count}/{len(raw_files)} images...")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    print(f"\n🎉 Training data fixed!")
    print(f"📊 Total images found: {len(raw_files)}")
    print(f"📊 Successfully processed: {processed_count}")
    print(f"📊 HR images: {processed_count} (original size, >=256x256)")
    print(f"📊 LR images: {processed_count} (4x downsampled)")

if __name__ == "__main__":
    fix_training_data() 