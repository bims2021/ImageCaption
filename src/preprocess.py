import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from typing import List, Tuple, Optional

from src.config import ImageCaptionConfig
from src.utils.logger import logger
from src.utils.error_handler import image_error_handler, DataProcessingError

class ImagePreprocessor:
    """Handles image loading and preprocessing for InceptionV3"""
    
    def __init__(self):
        self.target_size = ImageCaptionConfig.IMAGE_SIZE
        logger.info(f"Image preprocessor initialized with target size: {self.target_size}")
    
    @image_error_handler
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for InceptionV3
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=self.target_size)
            else:
                # Assume it's already a PIL image or file-like object
                image = Image.open(image_path).resize(self.target_size)
            
            # Convert to array and preprocess for InceptionV3
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
            
            return np.expand_dims(image_array, axis=0)  # Add batch dimension
            
        except Exception as e:
            raise DataProcessingError(f"Failed to preprocess image {image_path}: {str(e)}")
    
    @image_error_handler
    def extract_features(self, image_path: str, model) -> np.ndarray:
        """
        Extract features using pre-trained CNN
        
        Args:
            image_path: Path to image file
            model: Pre-trained feature extraction model
            
        Returns:
            Image features array
        """
        try:
            # Preprocess image
            processed_image = self.load_and_preprocess_image(image_path)
            
            # Extract features
            features = model.predict(processed_image, verbose=0)
            features = features.reshape((features.shape[0], -1))
            
            return features
            
        except Exception as e:
            raise DataProcessingError(f"Failed to extract features from {image_path}: {str(e)}")
    
    @image_error_handler
    def validate_image(self, image_path: str) -> bool:
        """
        Validate if image can be processed
        
        Args:
            image_path: Path to image file
            
        Returns:
            bool: True if image is valid
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False

# Global instance
image_preprocessor = ImagePreprocessor()