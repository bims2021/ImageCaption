import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
import os

from src.config import ImageCaptionConfig
from src.utils.logger import logger
from src.utils.error_handler import image_error_handler, InferenceError

class CaptionGenerator:
    """Handles caption generation using trained model"""
    
    def __init__(self, model, text_processor):
        self.model = model
        self.text_processor = text_processor
        self.encoder = None
        self.decoder = None
        
        logger.info("Caption generator initialized")
    
    @image_error_handler
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            
            # Extract encoder and decoder if possible
            if hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    if 'encoder' in layer.name:
                        self.encoder = layer
                    elif 'decoder' in layer.name:
                        self.decoder = layer
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            raise InferenceError(f"Failed to load model from {model_path}: {str(e)}")
    
    @image_error_handler
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract image features using encoder"""
        from preprocess import image_preprocessor
        
        try:
            if self.encoder:
                # Use model's encoder
                processed_image = image_preprocessor.load_and_preprocess_image(image_path)
                features = self.encoder.predict(processed_image, verbose=0)
                return features[0]  # Remove batch dimension
            else:
                # Fallback: extract features directly
                feature_model = tf.keras.applications.InceptionV3(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(*ImageCaptionConfig.IMAGE_SIZE, 3)
                )
                features = image_preprocessor.extract_features(image_path, feature_model)
                return features[0]
                
        except Exception as e:
            raise InferenceError(f"Failed to extract features from {image_path}: {str(e)}")
    
    @image_error_handler
    def generate_caption_greedy(self, image_path: str, max_length: int = None) -> str:
        """
        Generate caption using greedy decoding
        
        Args:
            image_path: Path to image file
            max_length: Maximum caption length
            
        Returns:
            Generated caption
        """
        try:
            max_length = max_length or ImageCaptionConfig.MAX_SEQUENCE_LENGTH
            
            # Extract image features
            image_features = self.extract_features(image_path)
            image_features = np.expand_dims(image_features, axis=0)
            
            # Start with start token
            start_token = self.text_processor.vocab.get(ImageCaptionConfig.START_TOKEN, 0)
            sequence = [start_token]
            
            # Generate caption token by token
            for _ in range(max_length):
                # Prepare input
                sequence_input = np.array([sequence])
                
                # Predict next token
                predictions = self.model.predict([image_features, sequence_input], verbose=0)
                next_token_probs = predictions[0, -1, :]  # Last token probabilities
                
                # Apply temperature
                next_token_probs = self._apply_temperature(next_token_probs, ImageCaptionConfig.TEMPERATURE)
                
                # Greedy selection
                next_token = np.argmax(next_token_probs)
                
                # Stop if end token is generated
                if next_token == self.text_processor.vocab.get(ImageCaptionConfig.END_TOKEN, 2):
                    break
                
                sequence.append(next_token)
            
            # Convert to text
            caption = self.text_processor.sequence_to_text(sequence[1:])  # Remove start token
            logger.info(f"Generated caption: {caption}")
            return caption
            
        except Exception as e:
            raise InferenceError(f"Greedy caption generation failed: {str(e)}")
    
    @image_error_handler
    def generate_caption_beam_search(self, image_path: str, beam_size: int = None, 
                                   max_length: int = None) -> List[Dict[str, Any]]:
        """
        Generate captions using beam search
        
        Args:
            image_path: Path to image file
            beam_size: Number of beams to keep
            max_length: Maximum caption length
            
        Returns:
            List of candidate captions with scores
        """
        try:
            beam_size = beam_size or ImageCaptionConfig.BEAM_SIZE
            max_length = max_length or ImageCaptionConfig.MAX_SEQUENCE_LENGTH
            
            # Extract image features
            image_features = self.extract_features(image_path)
            image_features = np.expand_dims(image_features, axis=0)
            
            start_token = self.text_processor.vocab.get(ImageCaptionConfig.START_TOKEN, 0)
            
            # Initialize beams
            beams = [([start_token], 0.0)]  # (sequence, score)
            
            for _ in range(max_length):
                new_beams = []
                
                for sequence, score in beams:
                    # Skip if sequence ended
                    if sequence[-1] == self.text_processor.vocab.get(ImageCaptionConfig.END_TOKEN, 2):
                        new_beams.append((sequence, score))
                        continue
                    
                    # Prepare input
                    sequence_input = np.array([sequence])
                    
                    # Predict next tokens
                    predictions = self.model.predict([image_features, sequence_input], verbose=0)
                    next_token_probs = predictions[0, -1, :]
                    
                    # Get top k tokens
                    top_k_tokens = np.argsort(next_token_probs)[-beam_size:][::-1]
                    
                    for token in top_k_tokens:
                        token_prob = next_token_probs[token]
                        new_sequence = sequence + [token]
                        new_score = score + np.log(token_prob + 1e-8)  # Log probability
                        new_beams.append((new_sequence, new_score))
                
                # Keep top beam_size beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Convert beams to captions
            candidates = []
            for sequence, score in beams:
                # Remove start token and any tokens after end token
                caption_sequence = sequence[1:]  # Remove start token
                if self.text_processor.vocab.get(ImageCaptionConfig.END_TOKEN, 2) in caption_sequence:
                    end_index = caption_sequence.index(self.text_processor.vocab.get(ImageCaptionConfig.END_TOKEN, 2))
                    caption_sequence = caption_sequence[:end_index]
                
                caption = self.text_processor.sequence_to_text(caption_sequence)
                candidates.append({
                    'caption': caption,
                    'score': float(score),
                    'sequence': caption_sequence
                })
            
            logger.info(f"Beam search generated {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            raise InferenceError(f"Beam search failed: {str(e)}")
    
    def _apply_temperature(self, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature to probabilities"""
        if temperature == 0:
            return probabilities
        
        probabilities = np.log(probabilities + 1e-8) / temperature
        probabilities = np.exp(probabilities)
        return probabilities / np.sum(probabilities)

# Global caption generator
caption_generator = None  # Will be initialized after model loading