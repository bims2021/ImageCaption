import os
import pickle
import tensorflow as tf
from typing import Dict, Tuple
import numpy as np

from src.config import ImageCaptionConfig
from src.utils.logger import logger

class ModelValidator:
    """Validates model files before loading"""
    
    @staticmethod
    def validate_model_file(model_path: str) -> Tuple[bool, str]:
        """
        Validate model file
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check file exists
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
            
            # Check file size
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if file_size_mb > ImageCaptionConfig.MAX_MODEL_SIZE_MB:
                return False, f"Model file too large: {file_size_mb:.2f}MB (max: {ImageCaptionConfig.MAX_MODEL_SIZE_MB}MB)"
            
            # Try to load model structure
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Validate input shapes
                if len(model.inputs) < 2:
                    return False, "Invalid model: Expected at least 2 inputs (image and sequence)"
                
                # Validate output shape
                if len(model.outputs) < 1:
                    return False, "Invalid model: No output layer found"
                
                logger.info(f"Model validation passed: {model_path}")
                logger.info(f"Model size: {file_size_mb:.2f}MB")
                logger.info(f"Input shapes: {[inp.shape for inp in model.inputs]}")
                logger.info(f"Output shape: {model.output.shape}")
                
                return True, "Model file is valid"
                
            except Exception as e:
                return False, f"Failed to load model: {str(e)}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_vocabulary_file(vocab_path: str) -> Tuple[bool, str, Dict]:
        """
        Validate vocabulary file
        
        Returns:
            Tuple of (is_valid, message, vocab_data)
        """
        try:
            # Check file exists
            if not os.path.exists(vocab_path):
                return False, f"Vocabulary file not found: {vocab_path}", {}
            
            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
            
            # Validate required keys
            required_keys = ['vocab', 'reverse_vocab', 'vocab_size', 'max_sequence_length']
            missing_keys = [key for key in required_keys if key not in vocab_data]
            
            if missing_keys:
                return False, f"Missing keys in vocabulary: {missing_keys}", {}
            
            # Validate vocab size
            if vocab_data['vocab_size'] < ImageCaptionConfig.MIN_VOCAB_SIZE:
                return False, f"Vocabulary too small: {vocab_data['vocab_size']} (min: {ImageCaptionConfig.MIN_VOCAB_SIZE})", {}
            
            # Validate special tokens
            vocab = vocab_data['vocab']
            required_tokens = [
                ImageCaptionConfig.START_TOKEN,
                ImageCaptionConfig.END_TOKEN,
                ImageCaptionConfig.UNKNOWN_TOKEN
            ]
            missing_tokens = [token for token in required_tokens if token not in vocab]
            
            if missing_tokens:
                return False, f"Missing special tokens: {missing_tokens}", {}
            
            logger.info(f"Vocabulary validation passed: {vocab_path}")
            logger.info(f"Vocab size: {vocab_data['vocab_size']}")
            logger.info(f"Max sequence length: {vocab_data['max_sequence_length']}")
            
            return True, "Vocabulary file is valid", vocab_data
            
        except Exception as e:
            return False, f"Failed to load vocabulary: {str(e)}", {}
    
    @staticmethod
    def validate_complete_model(model_dir: str = None) -> Dict:
        """
        Validate complete model package (model + vocabulary)
        
        Returns:
            Dictionary with validation results
        """
        model_dir = model_dir or ImageCaptionConfig.MODELS_DIR
        
        results = {
            'valid': True,
            'model_valid': False,
            'vocab_valid': False,
            'model_message': '',
            'vocab_message': '',
            'vocab_data': None,
            'errors': []
        }
        
        # Validate model file
        model_path = os.path.join(model_dir, "final_model.h5")
        model_valid, model_msg = ModelValidator.validate_model_file(model_path)
        results['model_valid'] = model_valid
        results['model_message'] = model_msg
        
        if not model_valid:
            results['valid'] = False
            results['errors'].append(model_msg)
        
        # Validate vocabulary file
        vocab_path = os.path.join(model_dir, "vocabulary.pkl")
        vocab_valid, vocab_msg, vocab_data = ModelValidator.validate_vocabulary_file(vocab_path)
        results['vocab_valid'] = vocab_valid
        results['vocab_message'] = vocab_msg
        results['vocab_data'] = vocab_data
        
        if not vocab_valid:
            results['valid'] = False
            results['errors'].append(vocab_msg)
        
        # Log summary
        if results['valid']:
            logger.info(" Complete model validation PASSED")
        else:
            logger.warning(" Complete model validation FAILED")
            for error in results['errors']:
                logger.warning(f"  - {error}")
        
        return results
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict:
        """Get detailed model information"""
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            
            return {
                'total_params': model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
                'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
                'layers': len(model.layers),
                'inputs': [str(inp.shape) for inp in model.inputs],
                'outputs': [str(out.shape) for out in model.outputs]
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

# Global validator instance
model_validator = ModelValidator()