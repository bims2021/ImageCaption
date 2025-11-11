import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

from src.config import ImageCaptionConfig
from src.utils.logger import logger
from src.utils.error_handler import image_error_handler, ModelTrainingError

class ImageCaptioningModel:
    """CNN Encoder + LSTM Decoder model for image captioning"""
    
    def __init__(self, vocab_size: int, max_sequence_length: int):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.encoder = None
        self.decoder = None
        self.full_model = None
        
        logger.info(f"Initializing model with vocab_size={vocab_size}, max_seq_len={max_sequence_length}")
    
    @image_error_handler
    def build_encoder(self) -> Model:
        """Build CNN encoder using InceptionV3"""
        try:
            # Load pre-trained InceptionV3
            inception = tf.keras.applications.InceptionV3(
                include_top=False,
                weights='imagenet',
                input_shape=(*ImageCaptionConfig.IMAGE_SIZE, 3)
            )
            
            # Freeze initial layers
            for layer in inception.layers[:-20]:
                layer.trainable = False
            
            # Build encoder
            inputs = inception.input
            features = inception.output
            features = layers.GlobalAveragePooling2D()(features)
            features = layers.Dense(ImageCaptionConfig.EMBEDDING_DIM, activation='relu')(features)
            features = layers.Dropout(ImageCaptionConfig.DROPOUT_RATE)(features)
            
            self.encoder = Model(inputs, features, name="encoder")
            logger.info("CNN encoder built successfully")
            return self.encoder
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to build encoder: {str(e)}")
    
    @image_error_handler
    def build_decoder(self) -> Model:
        """Build LSTM decoder for caption generation"""
        try:
            # Image features input
            image_features_input = layers.Input(
                shape=(ImageCaptionConfig.EMBEDDING_DIM,),
                name="image_features_input"
            )
            
            # Sequence input
            sequence_input = layers.Input(
                shape=(self.max_sequence_length,),
                name="sequence_input"
            )
            
            # Image feature processing
            image_dense = layers.Dense(ImageCaptionConfig.EMBEDDING_DIM, activation='relu')(
                image_features_input
            )
            image_dense = layers.RepeatVector(self.max_sequence_length)(image_dense)
            
            # Text embedding
            text_embedding = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=ImageCaptionConfig.EMBEDDING_DIM,
                input_length=self.max_sequence_length,
                name="text_embedding"
            )(sequence_input)
            
            # Combine image and text features
            combined = layers.Concatenate()([image_dense, text_embedding])
            
            # LSTM layers
            lstm_1 = layers.LSTM(
                ImageCaptionConfig.LSTM_UNITS,
                return_sequences=True,
                dropout=ImageCaptionConfig.DROPOUT_RATE
            )(combined)
            
            lstm_2 = layers.LSTM(
                ImageCaptionConfig.LSTM_UNITS,
                return_sequences=True,
                dropout=ImageCaptionConfig.DROPOUT_RATE
            )(lstm_1)
            
            # Output layer
            output = layers.TimeDistributed(
                layers.Dense(self.vocab_size, activation='softmax')
            )(lstm_2)
            
            self.decoder = Model(
                inputs=[image_features_input, sequence_input],
                outputs=output,
                name="decoder"
            )
            
            logger.info("LSTM decoder built successfully")
            return self.decoder
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to build decoder: {str(e)}")
    
    @image_error_handler
    def build_full_model(self) -> Model:
        """Build complete encoder-decoder model"""
        try:
            if not self.encoder:
                self.build_encoder()
            if not self.decoder:
                self.build_decoder()
            
            # Image input
            image_input = layers.Input(
                shape=(*ImageCaptionConfig.IMAGE_SIZE, 3),
                name="image_input"
            )
            
            # Sequence input
            sequence_input = layers.Input(
                shape=(self.max_sequence_length,),
                name="sequence_input"
            )
            
            # Get image features
            image_features = self.encoder(image_input)
            
            # Generate predictions
            predictions = self.decoder([image_features, sequence_input])
            
            self.full_model = Model(
                inputs=[image_input, sequence_input],
                outputs=predictions,
                name="image_captioning_model"
            )
            
            # Compile model
            self.full_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=ImageCaptionConfig.LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Full model built and compiled successfully")
            return self.full_model
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to build full model: {str(e)}")
    
    def get_model_summary(self):
        """Print model summary"""
        if self.full_model:
            return self.full_model.summary()
        return "Model not built yet"

# create a factory function:
def create_model_manager(vocab_size: int, max_sequence_length: int):
    """Create model manager with proper parameters"""
    return ImageCaptioningModel(vocab_size, max_sequence_length)