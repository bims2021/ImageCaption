"""
Helper script to prepare captions.txt from various formats
Supports Flickr8k and custom formats
"""

import os
import pandas as pd
from pathlib import Path

def prepare_flickr8k_captions(token_file_path, output_path='data/captions.txt'):
    """
    Convert Flickr8k captions.txt format to our format
    
    Flickr8k format: image_name.jpg#0\tCaption text
    Our format: image_name.jpg|Caption text
    """
    
    print(" Preparing Flickr8k captions...")
    
    captions = []
    
    with open(token_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse Flickr8k format: "image.jpg#0\tCaption"
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            image_info, caption = parts
            image_name = image_info.split('#')[0]  # Remove #0, #1, etc.
            
            # Convert to our format
            captions.append(f"{image_name}|{caption}")
    
    # Write to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(captions))
    
    print(f" Prepared {len(captions)} captions")
    print(f" Saved to: {output_path}")

def prepare_custom_captions(images_dir='data/images', output_path='data/captions.txt'):
    """
    Create a template captions file for manual editing
    Lists all images in the directory
    """
    
    print(" Creating captions template...")
    
    images_dir = Path(images_dir)
    
    if not images_dir.exists():
        print(f" Images directory not found: {images_dir}")
        return
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    for ext in image_extensions:
        images.extend(images_dir.glob(f'*{ext}'))
        images.extend(images_dir.glob(f'*{ext.upper()}'))
    
    if not images:
        print(f" No images found in: {images_dir}")
        return
    
    # Create template
    captions = []
    for img in sorted(images):
        img_name = img.name
        # Add 5 caption slots per image
        for i in range(5):
            captions.append(f"{img_name}|[Write caption {i+1} here]")
    
    # Write to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Image Caption File\n")
        f.write("# Format: image_name.jpg|Caption text here\n")
        f.write("# You can have multiple captions per image\n")
        f.write("# Lines starting with # are comments\n\n")
        f.write('\n'.join(captions))
    
    print(f" Created template for {len(images)} images")
    print(f" Total caption slots: {len(captions)}")
    print(f" Saved to: {output_path}")
    print(f"\n Next step: Edit {output_path} and replace [Write caption X here] with actual captions")

def validate_captions_file(captions_file='data/captions.txt', images_dir='data/images'):
    """Validate captions file"""
    
    print(" Validating captions file...")
    
    if not os.path.exists(captions_file):
        print(f" Captions file not found: {captions_file}")
        return False
    
    valid_lines = 0
    invalid_lines = 0
    missing_images = []
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check format
            if '|' not in line:
                print(f"  Line {line_num}: Invalid format (missing |)")
                invalid_lines += 1
                continue
            
            parts = line.split('|', 1)
            if len(parts) != 2:
                print(f"  Line {line_num}: Invalid format")
                invalid_lines += 1
                continue
            
            image_name, caption = parts
            
            # Check if image exists
            image_path = os.path.join(images_dir, image_name.strip())
            if not os.path.exists(image_path):
                if image_name not in missing_images:
                    missing_images.append(image_name)
                invalid_lines += 1
                continue
            
            # Check if caption is placeholder
            if '[Write caption' in caption or caption.strip() == '':
                print(f"  Line {line_num}: Placeholder caption not replaced")
                invalid_lines += 1
                continue
            
            valid_lines += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Valid captions: {valid_lines}")
    print(f"Invalid captions: {invalid_lines}")
    
    if missing_images:
        print(f"\n  Missing images ({len(missing_images)}):")
        for img in missing_images[:10]:  # Show first 10
            print(f"   • {img}")
        if len(missing_images) > 10:
            print(f"   ... and {len(missing_images) - 10} more")
    
    print("=" * 60)
    
    if valid_lines == 0:
        print("\n No valid captions found!")
        return False
    elif invalid_lines > 0:
        print(f"\n  Found {invalid_lines} issues. Please fix before training.")
        return False
    else:
        print("\n All captions valid! Ready for training.")
        return True

def prepare_csv_captions(csv_file_path, output_path='data/captions.txt'):
    """
    Converts a Kaggle-style Flickr8k CSV file (image,caption) to our internal 
    format (image.jpg|Caption text). This skips the header line.
    """
    print(" Preparing CSV captions...")
    
    captions = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        # Skip the header line ("image,caption")
        next(f)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse CSV format: "image.jpg,Caption"
            parts = line.split(',', 1) # Split only on the first comma
            
            if len(parts) != 2:
                # This should only happen if a line is malformed
                continue
            
            image_name = parts[0].strip()
            caption = parts[1].strip()
            
            # Convert to our format: image_name.jpg|Caption text
            captions.append(f"{image_name}|{caption}")
    
    # Write to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(captions))
    
    print(f" Prepared {len(captions)} captions")
    print(f" Saved to: {output_path}")    

def main():
    """Main function"""
    
    print("\n" + "=" * 60)
    print(" CAPTION FILE PREPARATION TOOL")
    print("=" * 60 + "\n")
    
    print("Choose an option:")
    print("1. Prepare Flickr8k captions")
    print("2. Create custom captions template")
    print("3. Validate existing captions file")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        token_file = input("Enter path to Flickr8k captions.txt (CSV format): ").strip()
        if os.path.exists(token_file):
            # Now calling the function that handles CSV files:
            prepare_csv_captions(token_file) 
        else:
            print(f" File not found: {token_file}")
    
    elif choice == '2':
        prepare_custom_captions()
    
    elif choice == '3':
        validate_captions_file()
    
    elif choice == '4':
        print(" Goodbye!")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
