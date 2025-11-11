"""
Complete training pipeline for Image Caption Generator
Run this script to train a new model from scratch
"""

import os
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ImageCaptionConfig
from src.text_processor import text_processor
from src.model import ImageCaptioningModel
from src.train import TrainingManager
from src.utils.logger import logger

def load_dataset():
    """Load images and captions from data directory"""
    
    logger.info("=" * 60)
    logger.info("LOADING DATASET")
    logger.info("=" * 60)
    
    captions_file = ImageCaptionConfig.CAPTIONS_FILE
    images_dir = ImageCaptionConfig.IMAGES_DIR
    
    # Check if captions file exists
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    # Load captions
    image_captions = {}  # {image_path: [captions]}
    
    logger.info(f"Reading captions from: {captions_file}")
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                # Parse format: image_name.jpg|Caption text here
                parts = line.split('|', 1)
                if len(parts) != 2:
                    logger.warning(f"Skipping line {line_num}: Invalid format")
                    continue
                
                image_name, caption = parts
                image_name = image_name.strip()
                caption = caption.strip()
                
                # Build full image path
                image_path = os.path.join(images_dir, image_name)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Add caption
                if image_path not in image_captions:
                    image_captions[image_path] = []
                image_captions[image_path].append(caption)
                
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    # Create lists for training
    image_paths = []
    captions = []
    
    for img_path, img_captions in image_captions.items():
        for caption in img_captions:
            image_paths.append(img_path)
            captions.append(caption)
    
    logger.info(f" Loaded {len(image_captions)} unique images")
    logger.info(f" Total training samples: {len(captions)}")
    logger.info(f" Average captions per image: {len(captions)/len(image_captions):.2f}")
    
    return image_paths, captions

def prepare_data(image_paths, captions):
    """Prepare data for training"""
    
    logger.info("=" * 60)
    logger.info("PREPARING DATA")
    logger.info("=" * 60)
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    text_processor.build_vocabulary(captions)
    logger.info(f" Vocabulary size: {text_processor.vocab_size}")
    logger.info(f" Max sequence length: {text_processor.max_sequence_length}")
    
    # Convert captions to sequences
    logger.info("Converting captions to sequences...")
    caption_sequences = []
    
    for i, caption in enumerate(captions):
        if i % 1000 == 0:
            logger.info(f"  Processed {i}/{len(captions)} captions...")
        
        sequence = text_processor.text_to_sequence(caption)
        caption_sequences.append(sequence)
    
    logger.info(f" Converted {len(caption_sequences)} captions to sequences")
    
    return caption_sequences

def create_model():
    """Create and compile model"""
    
    logger.info("=" * 60)
    logger.info("CREATING MODEL")
    logger.info("=" * 60)
    
    # Create model manager
    model_manager = ImageCaptioningModel(
        vocab_size=text_processor.vocab_size,
        max_sequence_length=text_processor.max_sequence_length
    )
    
    # Build full model
    logger.info("Building model architecture...")
    model_manager.build_full_model()
    
    # Print model summary
    logger.info("\nModel Summary:")
    model_manager.full_model.summary(print_fn=logger.info)
    
    logger.info(f" Model created successfully")
    logger.info(f" Total parameters: {model_manager.full_model.count_params():,}")
    
    return model_manager

def train_model(model_manager, image_paths, caption_sequences):
    """Train the model"""
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    # Create training manager
    trainer = TrainingManager(model_manager.full_model, text_processor)
    
    logger.info(f"Training configuration:")
    logger.info(f"  Batch size: {ImageCaptionConfig.BATCH_SIZE}")
    logger.info(f"  Epochs: {ImageCaptionConfig.EPOCHS}")
    logger.info(f"  Learning rate: {ImageCaptionConfig.LEARNING_RATE}")
    logger.info(f"  Validation split: 0.2")
    
    # Train
    logger.info("\n Starting training...")
    history = trainer.train_model(
        image_paths=image_paths,
        captions=caption_sequences,
        validation_split=0.2
    )
    
    logger.info(" Training completed!")
    
    return history, trainer

def save_artifacts(trainer, history):
    """Save model and training artifacts"""
    
    logger.info("=" * 60)
    logger.info("SAVING ARTIFACTS")
    logger.info("=" * 60)
    
    # Model and vocabulary are already saved by TrainingManager
    
    # Save training history
    history_path = os.path.join(ImageCaptionConfig.MODELS_DIR, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    logger.info(f" Training history saved: {history_path}")
    
    # Save training info
    info_path = os.path.join(ImageCaptionConfig.MODELS_DIR, "training_info.json")
    trainer.save_training_info(history, info_path)
    logger.info(f" Training info saved: {info_path}")
    
    logger.info("\n Saved files:")
    logger.info(f"   • Model: {ImageCaptionConfig.MODELS_DIR}/final_model.h5")
    logger.info(f"   • Vocabulary: {ImageCaptionConfig.MODELS_DIR}/vocabulary.pkl")
    logger.info(f"   • History: {history_path}")
    logger.info(f"   • Info: {info_path}")

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 60)
    print(" IMAGE CAPTION GENERATOR - TRAINING PIPELINE")
    print("=" * 60 + "\n")
    
    try:
        # Step 1: Load dataset
        image_paths, captions = load_dataset()
        
        # Step 2: Prepare data
        caption_sequences = prepare_data(image_paths, captions)
        
        # Step 3: Create model
        model_manager = create_model()
        
        # Step 4: Train model
        history, trainer = train_model(model_manager, image_paths, caption_sequences)
        
        # Step 5: Save artifacts
        save_artifacts(trainer, history)
        
        # Success message
        print("\n" + "=" * 60)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n Your model is ready to use!")
        print(f" Model saved in: {ImageCaptionConfig.MODELS_DIR}/")
        print("\n Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Click 'Load Pretrained Model' in the sidebar")
        print("   3. Upload an image and generate captions!")
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        logger.warning("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n Training failed: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
