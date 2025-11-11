import tensorflow as tf
import numpy as np
import os
from typing import List, Tuple, Dict
import json
import pickle
import gc

from src.config import ImageCaptionConfig
from src.utils.logger import logger
from src.utils.error_handler import image_error_handler, ModelTrainingError

class TrainingManager:
    """Manages model training with callbacks and data generation - MEMORY EFFICIENT"""
    
    def __init__(self, model, text_processor):
        self.model = model
        self.text_processor = text_processor
        self.callbacks = []
        self.features_cache = None
        
        logger.info("Training manager initialized")
    
    def load_cached_features(self) -> Dict:
        """Load pre-extracted image features from cache"""
        features_file = os.path.join(ImageCaptionConfig.DATA_DIR, "features", "image_features.pkl")
        
        if os.path.exists(features_file):
            logger.info(f"📦 Loading cached features from {features_file}...")
            with open(features_file, 'rb') as f:
                features_dict = pickle.load(f)
            logger.info(f"✅ Loaded features for {len(features_dict)} images")
            return features_dict
        else:
            logger.warning("⚠️  No cached features found.")
            return None
    
    @image_error_handler
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks"""
        try:
            # Model checkpointing
            checkpoint_path = os.path.join(
                ImageCaptionConfig.MODELS_DIR, 
                "model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5"
            )
            
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=ImageCaptionConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
            
            # Reduce learning rate on plateau
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
            
            self.callbacks = [checkpoint_callback, early_stopping, reduce_lr]
            logger.info("Training callbacks setup completed")
            return self.callbacks
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to setup callbacks: {str(e)}")
    
    def prepare_data_arrays(self, image_paths: List[str], captions: List[List[int]]) -> Tuple:
        """
        Prepare data arrays from cached features - MEMORY EFFICIENT
        
        Returns:
            Tuple of (encoder_inputs, decoder_inputs, decoder_targets)
        """
        try:
            logger.info(f"Preparing data for {len(image_paths)} samples...")
            
            # Load features cache if not already loaded
            if self.features_cache is None:
                self.features_cache = self.load_cached_features()
            
            if self.features_cache is None:
                raise ModelTrainingError("No cached features available!")
            
            encoder_inputs = []
            decoder_inputs = []
            decoder_targets = []
            
            logger.info("✅ Using cached features (FAST)")
            
            for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
                if i % 5000 == 0 and i > 0:
                    logger.info(f"  Prepared {i}/{len(image_paths)} samples...")
                
                # Get cached features
                if img_path not in self.features_cache:
                    logger.warning(f"Image not in cache: {img_path}")
                    continue
                
                features = self.features_cache[img_path]
                
                # Prepare caption sequences
                caption_list = caption.tolist() if isinstance(caption, np.ndarray) else caption
                
                if len(caption_list) > ImageCaptionConfig.MAX_SEQUENCE_LENGTH:
                    caption_list = caption_list[:ImageCaptionConfig.MAX_SEQUENCE_LENGTH]
                
                padded_caption = caption_list + [0] * (ImageCaptionConfig.MAX_SEQUENCE_LENGTH - len(caption_list))
                
                encoder_inputs.append(features)
                decoder_inputs.append(padded_caption[:-1])
                decoder_targets.append(padded_caption[1:])
            
            logger.info(f"✅ Prepared {len(encoder_inputs)} samples")
            
            # Convert to numpy arrays
            encoder_inputs = np.array(encoder_inputs, dtype=np.float32)
            decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
            decoder_targets = np.array(decoder_targets, dtype=np.int32)
            
            logger.info(f"📊 Data shapes: encoder={encoder_inputs.shape}, decoder_in={decoder_inputs.shape}, decoder_out={decoder_targets.shape}")
            
            return encoder_inputs, decoder_inputs, decoder_targets
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to prepare data: {str(e)}")
    
    @image_error_handler
    def train_model(self, image_paths: List[str], captions: List[List[int]], 
                   validation_split: float = 0.05) -> tf.keras.callbacks.History:
        """
        Train the image captioning model - MEMORY EFFICIENT VERSION
        
        Args:
            image_paths: List of image paths
            captions: List of caption sequences
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        try:
            # Setup callbacks
            self.setup_callbacks()
            
            # Split indices (not data yet)
            split_index = int(len(image_paths) * (1 - validation_split))
            
            logger.info(f"Training on {split_index} samples, validating on {len(image_paths) - split_index} samples")
            
            # STEP 1: Prepare TRAINING data
            logger.info("=" * 60)
            logger.info("PREPARING TRAINING DATA")
            logger.info("=" * 60)
            
            train_images = image_paths[:split_index]
            train_captions = captions[:split_index]
            
            train_enc, train_dec_in, train_dec_out = self.prepare_data_arrays(
                train_images, 
                train_captions
            )
            
            # Create training dataset
            logger.info("Creating training dataset with cache and prefetch...")
            train_dataset = tf.data.Dataset.from_tensor_slices((
                (train_enc, train_dec_in),
                train_dec_out
            ))
            train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
            train_dataset = train_dataset.cache()  # Cache in memory
            train_dataset = train_dataset.batch(ImageCaptionConfig.BATCH_SIZE)
            train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            
            logger.info("✅ Training dataset ready")
            
            # Clear training arrays from memory
            del train_enc, train_dec_in, train_dec_out
            gc.collect()
            logger.info("🧹 Cleared training arrays from memory")
            
            # STEP 2: Prepare VALIDATION data
            logger.info("=" * 60)
            logger.info("PREPARING VALIDATION DATA")
            logger.info("=" * 60)
            
            val_images = image_paths[split_index:]
            val_captions = captions[split_index:]
            
            val_enc, val_dec_in, val_dec_out = self.prepare_data_arrays(
                val_images,
                val_captions
            )
            
            # Create validation dataset
            logger.info("Creating validation dataset with cache and prefetch...")
            val_dataset = tf.data.Dataset.from_tensor_slices((
                (val_enc, val_dec_in),
                val_dec_out
            ))
            val_dataset = val_dataset.cache()  # Cache in memory
            val_dataset = val_dataset.batch(ImageCaptionConfig.BATCH_SIZE)
            val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            
            logger.info("✅ Validation dataset ready")
            
            # Clear validation arrays from memory
            del val_enc, val_dec_in, val_dec_out
            gc.collect()
            logger.info("🧹 Cleared validation arrays from memory")
            
            # Clear features cache to free memory
            self.features_cache = None
            gc.collect()
            logger.info("🧹 Cleared features cache from memory")
            
            # STEP 3: Train model
            logger.info("=" * 60)
            logger.info("🚀 STARTING MODEL TRAINING")
            logger.info("=" * 60)
            
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=ImageCaptionConfig.EPOCHS,
                callbacks=self.callbacks,
                verbose=1
            )
            
            # Save final model
            final_model_path = os.path.join(ImageCaptionConfig.MODELS_DIR, "final_model.h5")
            self.model.save(final_model_path)
            logger.info(f"✅ Training completed. Model saved to {final_model_path}")
            
            # Save vocabulary
            vocab_path = os.path.join(ImageCaptionConfig.MODELS_DIR, "vocabulary.pkl")
            save_vocabulary(self.text_processor, vocab_path)
            
            return history
            
        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    @image_error_handler
    def save_training_info(self, history, filepath: str):
        """Save training history and configuration"""
        try:
            training_info = {
                'config': {
                    'image_size': ImageCaptionConfig.IMAGE_SIZE,
                    'embedding_dim': ImageCaptionConfig.EMBEDDING_DIM,
                    'lstm_units': ImageCaptionConfig.LSTM_UNITS,
                    'vocab_size': self.text_processor.vocab_size,
                    'max_sequence_length': self.text_processor.max_sequence_length
                },
                'history': {
                    'loss': [float(x) for x in history.history.get('loss', [])],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                    'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                    'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])]
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(training_info, f, indent=2)
                
            logger.info(f"Training info saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save training info: {e}")

# Global training manager
training_manager = None

def save_vocabulary(text_processor, filepath: str):
    """Save vocabulary for later use during inference"""
    try:
        vocab_data = {
            'vocab': text_processor.vocab,
            'reverse_vocab': text_processor.reverse_vocab,
            'vocab_size': text_processor.vocab_size,
            'max_sequence_length': text_processor.max_sequence_length
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        logger.info(f"Vocabulary saved to {filepath}")
        logger.info(f"Vocab size: {vocab_data['vocab_size']}, Max sequence length: {vocab_data['max_sequence_length']}")
        
    except Exception as e:
        logger.error(f"Failed to save vocabulary: {e}")
        raise ModelTrainingError(f"Failed to save vocabulary: {str(e)}")