import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter

from src.config import ImageCaptionConfig
from src.utils.logger import logger
from src.utils.error_handler import image_error_handler, DataProcessingError

class TextProcessor:
    """Handles text processing, tokenization, and vocabulary management"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.max_sequence_length = 0
        self.START_TOKEN = ImageCaptionConfig.START_TOKEN
        self.END_TOKEN = ImageCaptionConfig.END_TOKEN
        self.UNKNOWN_TOKEN = ImageCaptionConfig.UNKNOWN_TOKEN
        
        logger.info("Text processor initialized")
    
    @image_error_handler
    def build_vocabulary(self, captions: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from captions
        
        Args:
            captions: List of caption strings
            
        Returns:
            Vocabulary dictionary
        """
        try:
            logger.info(f"Building vocabulary from {len(captions)} captions")
            
            # Preprocess captions
            processed_captions = [self._preprocess_text(caption) for caption in captions]
            
            # Tokenize and count words
            word_counts = Counter()
            for caption in processed_captions:
                tokens = caption.split()
                word_counts.update(tokens)
            
            # Create vocabulary (most common words)
            vocab_words = [self.START_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]
            vocab_words.extend([word for word, count in word_counts.most_common(
                ImageCaptionConfig.MAX_VOCAB_SIZE - len(vocab_words)
            )])
            
            # Create word to index mapping
            self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
            self.reverse_vocab = {idx: word for idx, word in enumerate(vocab_words)}
            self.vocab_size = len(vocab_words)
            
            logger.info(f"Vocabulary built with {self.vocab_size} words")
            return self.vocab
            
        except Exception as e:
            raise DataProcessingError(f"Failed to build vocabulary: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text: lowercase, remove special chars, etc."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @image_error_handler
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of indices
        
        Args:
            text: Input text
            
        Returns:
            List of word indices
        """
        try:
            processed_text = self._preprocess_text(text)
            tokens = processed_text.split()
            
            sequence = [self.vocab.get(self.START_TOKEN, 0)]  # Start token
            
            for token in tokens:
                sequence.append(self.vocab.get(token, self.vocab.get(self.UNKNOWN_TOKEN, 1)))
            
            sequence.append(self.vocab.get(self.END_TOKEN, 2))  # End token
            
            # Update max sequence length
            self.max_sequence_length = max(self.max_sequence_length, len(sequence))
            
            return sequence
            
        except Exception as e:
            raise DataProcessingError(f"Failed to convert text to sequence: {str(e)}")
    
    @image_error_handler
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert sequence of indices back to text
        
        Args:
            sequence: List of word indices
            
        Returns:
            Generated text
        """
        try:
            tokens = []
            for idx in sequence:
                if idx in self.reverse_vocab:
                    word = self.reverse_vocab[idx]
                    if word not in [self.START_TOKEN, self.END_TOKEN]:
                        tokens.append(word)
                else:
                    tokens.append(self.UNKNOWN_TOKEN)
            
            return ' '.join(tokens)
            
        except Exception as e:
            raise DataProcessingError(f"Failed to convert sequence to text: {str(e)}")
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information"""
        return {
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'start_token': self.START_TOKEN,
            'end_token': self.END_TOKEN
        }

# Global instance
text_processor = TextProcessor()