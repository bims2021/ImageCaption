# 🖼️ Image Caption Generator

An AI-powered image captioning system using CNN (InceptionV3) + LSTM architecture. Generate descriptive captions for your images using deep learning.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Training Your Own Model](#training-your-own-model)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- 🤖 **Deep Learning Model**: CNN encoder (InceptionV3) + LSTM decoder
- 🎯 **Beam Search**: Generate multiple caption candidates
- 📊 **Interactive UI**: User-friendly Streamlit interface
- ⚙️ **Configurable**: Adjustable beam size and temperature
- 📈 **Progress Tracking**: Real-time progress bars for long operations
- ✅ **Model Validation**: Automatic validation before loading
- 📝 **Comprehensive Logging**: Detailed logs for debugging
- 🎨 **Example Images**: Try with pre-loaded examples

## 📁 Project Structure

```
image_caption/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env                        # Environment variables (optional)
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── model.py               # Model architecture
│   ├── text_processor.py      # Text processing & tokenization
│   ├── preprocess.py          # Image preprocessing
│   ├── inference.py           # Caption generation
│   ├── train.py               # Training manager
│   ├── model_validator.py     # Model validation utilities
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── logger.py          # Logging configuration
│       └── error_handler.py   # Error handling
│
├── data/                       # Data directory
│   ├── images/                # Training images
│   ├── examples/              # Example images for demo
│   └── captions.txt           # Image-caption pairs
│
├── models/                     # Saved models
│   ├── final_model.h5         # Trained model weights
│   └── vocabulary.pkl         # Vocabulary mapping
│
└── logs/                       # Log files
    ├── image_caption_YYYYMMDD.log
    └── errors_YYYYMMDD.log
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Required Directories

The directories will be created automatically when you first run the app, but you can create them manually:

```bash
mkdir -p data/images data/examples models logs
```

## 🎯 Quick Start

### Option 1: Use Pre-trained Model

1. Download a pre-trained model (if available):
   ```bash
   # Place final_model.h5 and vocabulary.pkl in the models/ directory
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`

4. Click "Load Pretrained Model" in the sidebar

5. Upload an image and click "Generate Captions"

### Option 2: Start from Scratch

If you want to train your own model, see [Training Your Own Model](#training-your-own-model).

## 💻 Usage

### Web Interface

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Load the model:**
   - Click "🔄 Load Pretrained Model" in the sidebar
   - Wait for validation and loading to complete

3. **Generate captions:**
   - Upload an image (JPG, PNG, BMP)
   - Adjust beam size and temperature (optional)
   - Click "🎯 Generate Captions"
   - View multiple caption candidates with confidence scores

4. **Try examples:**
   - Scroll down to see example images
   - Click on any example to use it

### Configuration Options

- **Beam Size (1-5)**: Number of caption candidates to generate
  - Higher = more options, slower generation
  - Lower = fewer options, faster generation

- **Temperature (0.1-1.0)**: Controls creativity
  - Higher = more creative/diverse captions
  - Lower = more conservative/safe captions

## 🎓 Training Your Own Model

### Step 1: Prepare Dataset

1. **Collect images:**
   - Place images in `data/images/`
   - Supported formats: JPG, PNG, BMP
   - Recommended: 5,000+ images

2. **Create captions file:**
   - Create `data/captions.txt`
   - Format: `image_filename.jpg|Caption text here`
   - Example:
     ```
     dog.jpg|A brown dog running in the park
     cat.jpg|A black cat sitting on a windowsill
     beach.jpg|People walking on a sandy beach at sunset
     ```

### Step 2: Configure Training

Edit `src/config.py` to adjust training parameters:

```python
# Training Parameters
BATCH_SIZE = 32          # Adjust based on GPU memory
EPOCHS = 50              # Number of training epochs
LEARNING_RATE = 0.001    # Learning rate
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience

# Vocabulary & Text
MAX_VOCAB_SIZE = 10000   # Maximum vocabulary size
MAX_SEQUENCE_LENGTH = 40 # Maximum caption length
```

### Step 3: Create Training Script

Create `train_pipeline.py`:

```python
import os
from src.config import ImageCaptionConfig
from src.text_processor import text_processor
from src.model import create_model_manager
from src.train import TrainingManager

# Load your data
# ... (load image paths and captions)

# Build vocabulary
text_processor.build_vocabulary(captions)

# Create model
model_manager = create_model_manager(
    text_processor.vocab_size,
    text_processor.max_sequence_length
)
model_manager.build_full_model()

# Train
trainer = TrainingManager(model_manager.full_model, text_processor)
history = trainer.train_model(image_paths, caption_sequences)
```

### Step 4: Run Training

```bash
python train_pipeline.py
```

Training will:
- ✅ Build vocabulary from captions
- ✅ Extract image features using InceptionV3
- ✅ Train the LSTM decoder
- ✅ Save checkpoints during training
- ✅ Save final model and vocabulary
- ✅ Generate training logs

### Step 5: Monitor Training

View logs in real-time:

```bash
tail -f logs/image_caption_YYYYMMDD.log
```

Or use TensorBoard:

```bash
tensorboard --logdir=logs/tensorboard
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# Logging
LOG_LEVEL=INFO

# Model paths (optional)
MODEL_PATH=./models/final_model.h5
VOCAB_PATH=./models/vocabulary.pkl
```

### Advanced Configuration

Edit `src/config.py` for fine-tuning:

```python
# Model Architecture
IMAGE_SIZE = (299, 299)      # Input image size
EMBEDDING_DIM = 256          # Embedding dimension
LSTM_UNITS = 512             # LSTM hidden units
DROPOUT_RATE = 0.3           # Dropout rate

# Inference
BEAM_SIZE = 3                # Default beam size
TEMPERATURE = 0.7            # Default temperature
```

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Solution: Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

**2. TensorFlow GPU Issues**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA and cuDNN if needed
```

**3. Memory Errors**
- Reduce `BATCH_SIZE` in config.py
- Use smaller images
- Close other applications

**4. Model Loading Fails**
- Check model file exists: `ls models/`
- Validate model integrity
- Re-download or re-train model

**5. Slow Generation**
- Reduce `BEAM_SIZE`
- Use GPU if available
- Optimize model architecture

### Debug Mode

Enable debug logging:

```python
# In .env
LOG_LEVEL=DEBUG
```

View detailed logs:
```bash
cat logs/image_caption_YYYYMMDD.log
```

## 📊 Model Performance

Typical performance metrics:

- **Training Time**: 2-5 hours (5K images, GPU)
- **Inference Time**: 1-3 seconds per image (CPU)
- **Memory Usage**: 16 GB RAM
- **Model Size**: ~100-200 MB

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- InceptionV3 architecture by Google
- TensorFlow and Keras teams
- Streamlit for the amazing web framework
- The open-source community
