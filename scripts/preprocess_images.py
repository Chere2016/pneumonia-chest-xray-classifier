import os
from PIL import Image, ImageOps
from tqdm import tqdm

def pad_and_resize(img_path, output_path, target_size=(224, 224)):
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if it's grayscale to ensure consistency
            img = img.convert("RGB")
            
            # Calculate padding to make it square
            width, height = img.size
            
            if width == height:
                padding = (0, 0, 0, 0)
            elif width > height:
                diff = width - height
                top_pad = diff // 2
                bottom_pad = diff - top_pad
                padding = (0, top_pad, 0, bottom_pad) # left, top, right, bottom
            else:
                diff = height - width
                left_pad = diff // 2
                right_pad = diff - left_pad
                padding = (left_pad, 0, right_pad, 0)
                
            # Apply padding (black borders)
            img_padded = ImageOps.expand(img, padding, fill=(0, 0, 0))
            
            # Resize to target size using high-quality LANCZOS filter
            img_resized = img_padded.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save the new image
            img_resized.save(output_path, format="JPEG", quality=95)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    data_dir = '/home/falcon/student1/mscs/medical_classifier/data'
    new_data_dir = '/home/falcon/student1/mscs/medical_classifier/data_resized'
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    # Create new directory structure
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(new_data_dir, split, cls), exist_ok=True)
            
    # Process all images
    for split in splits:
        for cls in classes:
            cls_dir = os.path.join(data_dir, split, cls)
            out_dir = os.path.join(new_data_dir, split, cls)
            
            if os.path.exists(cls_dir):
                images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
                
                print(f"Processing {split}/{cls} ({len(images)} images)...")
                for img_name in tqdm(images, leave=False):
                    img_path = os.path.join(cls_dir, img_name)
                    output_path = os.path.join(out_dir, img_name)
                    pad_and_resize(img_path, output_path)

    print("\nDataset preprocessing complete! All images are now 224x224 squares without distortion.")

if __name__ == '__main__':
    main()
