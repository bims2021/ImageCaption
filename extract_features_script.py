"""
Pre-extract and cache image features for faster training
Run this once before training to save hours of processing time
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ImageCaptionConfig
from src.preprocess import image_preprocessor
from src.utils.logger import logger

def extract_and_save_features():
    """Extract features from all images and save to disk"""
    
    print("\n" + "=" * 60)
    print("🚀 IMAGE FEATURE EXTRACTION")
    print("=" * 60 + "\n")
    
    # Load image paths from captions file
    captions_file = ImageCaptionConfig.CAPTIONS_FILE
    images_dir = ImageCaptionConfig.IMAGES_DIR
    
    if not os.path.exists(captions_file):
        print(f"❌ Captions file not found: {captions_file}")
        return
    
    # Get unique image paths
    print("📝 Reading image list from captions file...")
    unique_images = set()
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|', 1)
            if len(parts) == 2:
                image_name = parts[0].strip()
                image_path = os.path.join(images_dir, image_name)
                if os.path.exists(image_path):
                    unique_images.add(image_path)
    
    image_paths = sorted(list(unique_images))
    print(f"✅ Found {len(image_paths)} unique images\n")
    
    # Create feature extraction model
    print("🏗️  Building InceptionV3 feature extractor...")
    feature_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(*ImageCaptionConfig.IMAGE_SIZE, 3)
    )
    feature_model = tf.keras.Model(
        inputs=feature_model.input,
        outputs=feature_model.layers[-1].output
    )
    print("✅ Feature extractor ready\n")
    
    # Create features directory
    features_dir = os.path.join(ImageCaptionConfig.DATA_DIR, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    # Extract features with progress bar
    print("🔍 Extracting image features...")
    print("⏱️  This will take ~20-30 minutes for 8,000 images")
    print("💾 Features will be cached for instant loading later\n")
    
    features_dict = {}
    failed_images = []
    
    # Use tqdm for progress bar
    for img_path in tqdm(image_paths, desc="Extracting", unit="img"):
        try:
            features = image_preprocessor.extract_features(img_path, feature_model)
            features_dict[img_path] = features[0]  # Remove batch dimension
        except Exception as e:
            logger.warning(f"Failed to extract features from {img_path}: {e}")
            failed_images.append(img_path)
    
    # Save features
    features_file = os.path.join(features_dir, "image_features.pkl")
    print(f"\n💾 Saving features to {features_file}...")
    
    with open(features_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    # Save metadata
    metadata = {
        'num_images': len(features_dict),
        'failed_images': failed_images,
        'feature_shape': features_dict[list(features_dict.keys())[0]].shape if features_dict else None,
        'model': 'InceptionV3'
    }
    
    metadata_file = os.path.join(features_dir, "features_metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Summary
    file_size_mb = os.path.getsize(features_file) / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"📊 Extracted: {len(features_dict)} images")
    print(f"❌ Failed: {len(failed_images)} images")
    print(f"💾 File size: {file_size_mb:.2f} MB")
    print(f"📁 Location: {features_file}")
    print("\n💡 Next step: Run train_pipeline.py (will be MUCH faster!)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    extract_and_save_features()
