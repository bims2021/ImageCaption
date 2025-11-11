import traceback
import sys
from typing import Callable, Any

class ImageCaptionError(Exception):
    """Base exception for image caption generator"""
    pass

class ModelTrainingError(ImageCaptionError):
    """Raised when model training fails"""
    pass

class InferenceError(ImageCaptionError):
    """Raised when caption generation fails"""
    pass

class DataProcessingError(ImageCaptionError):
    """Raised when data processing fails"""
    pass

def handle_image_error(error: Exception, context: str = "") -> str:
    """Handle errors for image captioning system"""
    from .logger import logger
    
    logger.error(f"Error in {context}: {str(error)}")
    logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # User-friendly messages
    if isinstance(error, ModelTrainingError):
        return "Model training failed. Please check your data and try again."
    elif isinstance(error, InferenceError):
        return "Caption generation failed. Please try with a different image."
    elif isinstance(error, DataProcessingError):
        return "Data processing error. Please check your image files."
    else:
        return "An unexpected error occurred. Please try again."

def image_error_handler(func: Callable) -> Callable:
    """Decorator for image captioning functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = f"{func.__module__}.{func.__name__}"
            user_message = handle_image_error(e, context)
            raise type(e)(user_message) from e
    return wrapper