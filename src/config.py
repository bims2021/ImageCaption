import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler # Import for setup_logging to work

load_dotenv()

class ImageCaptionConfig:
    """Configuration for Image Caption Generator"""
    
    # Model Architecture
    IMAGE_SIZE = (299, 299)  # InceptionV3 input size
    EMBEDDING_DIM = 256
    LSTM_UNITS = 512
    DENSE_UNITS = 512
    DROPOUT_RATE = 0.3
    
    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5
    
    # Vocabulary & Text
    MAX_VOCAB_SIZE = 10000
    MAX_SEQUENCE_LENGTH = 40
    UNKNOWN_TOKEN = "<unk>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    
    # Paths (***MODIFIED: Use local /content/ paths to avoid Google Drive quota errors***)
    DATA_DIR = "./data"
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
    MODELS_DIR = "/content/models" # <-- CHANGED
    LOGS_DIR = "/content/logs"   # <-- CHANGED
    EXAMPLES_DIR = os.path.join(DATA_DIR, "examples")
    
    # Inference
    BEAM_SIZE = 3
    TEMPERATURE = 0.7
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5
    
    # Model Validation
    MIN_VOCAB_SIZE = 100
    MAX_MODEL_SIZE_MB = 500
    REQUIRED_MODEL_FILES = ['final_model.h5', 'vocabulary.pkl']
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.IMAGES_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.EXAMPLES_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logging.info("All directories created successfully")
    
    @classmethod
    def setup_logging(cls):
        """Configure logging with rotation and proper formatting"""
        
        # Create logs directory
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, cls.LOG_LEVEL.upper()))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(cls.LOG_FORMAT, cls.LOG_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = os.path.join(
            cls.LOGS_DIR, 
            f"image_caption_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=cls.LOG_FILE_MAX_BYTES,
            backupCount=cls.LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(cls.LOG_FORMAT, cls.LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = os.path.join(
            cls.LOGS_DIR,
            f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=cls.LOG_FILE_MAX_BYTES,
            backupCount=cls.LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        logger.info("Logging configured successfully")
        logger.info(f"Main log file: {log_file}")
        logger.info(f"Error log file: {error_log_file}")
    
    @classmethod
    def validate_model_files(cls) -> dict:
        """Validate required model files exist"""
        validation_results = {
            'valid': True,
            'missing_files': [],
            'file_sizes': {},
            'total_size_mb': 0
        }
        
        for filename in cls.REQUIRED_MODEL_FILES:
            filepath = os.path.join(cls.MODELS_DIR, filename)
            
            if not os.path.exists(filepath):
                validation_results['valid'] = False
                validation_results['missing_files'].append(filename)
            else:
                size_bytes = os.path.getsize(filepath)
                size_mb = size_bytes / (1024 * 1024)
                validation_results['file_sizes'][filename] = size_mb
                validation_results['total_size_mb'] += size_mb
        
        return validation_results
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get configuration summary"""
        return {
            'model': {
                'image_size': cls.IMAGE_SIZE,
                'embedding_dim': cls.EMBEDDING_DIM,
                'lstm_units': cls.LSTM_UNITS,
                'dropout_rate': cls.DROPOUT_RATE
            },
            'training': {
                'batch_size': cls.BATCH_SIZE,
                'epochs': cls.EPOCHS,
                'learning_rate': cls.LEARNING_RATE,
                'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE
            },
            'vocabulary': {
                'max_vocab_size': cls.MAX_VOCAB_SIZE,
                'max_sequence_length': cls.MAX_SEQUENCE_LENGTH
            },
            'inference': {
                'beam_size': cls.BEAM_SIZE,
                'temperature': cls.TEMPERATURE
            },
            'paths': {
                'data_dir': cls.DATA_DIR,
                'models_dir': cls.MODELS_DIR,
                'logs_dir': cls.LOGS_DIR
            }
        }

# Setup on import
ImageCaptionConfig.setup_directories()
ImageCaptionConfig.setup_logging()