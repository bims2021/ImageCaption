import streamlit as st
import os
import sys
from PIL import Image
import tempfile
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ImageCaptionConfig
from src.utils.logger import setup_image_caption_logger
from src.utils.error_handler import handle_image_error
from src.text_processor import text_processor
from src.model import model_manager
from src.inference import CaptionGenerator
from src.model_validator import model_validator

# Setup logging
logger = setup_image_caption_logger()

def initialize_app():
    """Initialize the application"""
    try:
        st.set_page_config(
            page_title="Image Caption Generator",
            page_icon="🖼️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'caption_generator' not in st.session_state:
            st.session_state.caption_generator = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = None
        
        logger.info("Image Caption Generator app initialized")
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize app: {str(e)}")
        return False

def load_pretrained_model():
    """Load pretrained model with validation and progress tracking"""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Validate model files (20%)
        status_text.text(" Validating model files...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        validation_results = model_validator.validate_complete_model()
        
        if not validation_results['valid']:
            st.error(" Model validation failed:")
            for error in validation_results['errors']:
                st.error(f"  • {error}")
            progress_bar.empty()
            status_text.empty()
            return False
        
        # Step 2: Load vocabulary (40%)
        status_text.text(" Loading vocabulary...")
        progress_bar.progress(40)
        time.sleep(0.3)
        
        vocab_data = validation_results['vocab_data']
        text_processor.vocab = vocab_data['vocab']
        text_processor.reverse_vocab = vocab_data['reverse_vocab']
        text_processor.vocab_size = vocab_data['vocab_size']
        text_processor.max_sequence_length = vocab_data['max_sequence_length']
        
        # Step 3: Build model architecture (60%)
        status_text.text(" Building model architecture...")
        progress_bar.progress(60)
        time.sleep(0.3)
        
        model_manager.vocab_size = text_processor.vocab_size
        model_manager.max_sequence_length = text_processor.max_sequence_length
        model_manager.build_full_model()
        
        # Step 4: Load model weights (80%)
        status_text.text(" Loading model weights...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        model_path = os.path.join(ImageCaptionConfig.MODELS_DIR, "final_model.h5")
        model_manager.full_model.load_weights(model_path)
        
        # Step 5: Initialize caption generator (100%)
        status_text.text(" Initializing caption generator...")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        caption_generator = CaptionGenerator(model_manager.full_model, text_processor)
        
        st.session_state.caption_generator = caption_generator
        st.session_state.model_loaded = True
        st.session_state.model_info = model_validator.get_model_info(model_path)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        logger.info(" Pretrained model loaded successfully")
        st.success(" Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        st.error(f" Error loading model: {str(e)}")
        return False

def main():
    """Main application function"""
    if not initialize_app():
        return
    
    # Application header
    st.title(" Image Caption Generator")
    st.markdown("Upload an image and generate descriptive captions using AI")
    
    # Sidebar
    with st.sidebar:
        st.header(" Configuration")
        
        # Model status
        st.subheader(" Model Status")
        if st.session_state.model_loaded:
            st.success(" Model Loaded")
            
            # Show model info
            if st.session_state.model_info:
                with st.expander("Model Details"):
                    info = st.session_state.model_info
                    st.write(f"**Total Parameters:** {info.get('total_params', 'N/A'):,}")
                    st.write(f"**Trainable Parameters:** {info.get('trainable_params', 'N/A'):,}")
                    st.write(f"**Layers:** {info.get('layers', 'N/A')}")
                    st.write(f"**Vocab Size:** {text_processor.vocab_size:,}")
        else:
            st.warning(" No Model Found")
            if st.button(" Load Pretrained Model"):
                with st.spinner("Loading model..."):
                    if load_pretrained_model():
                        st.rerun()
        
        st.divider()
        
        # Generation settings
        st.subheader(" Generation Settings")
        beam_size = st.slider("Beam Size", 1, 5, 3, 
                            help="Number of caption candidates to generate")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1,
                              help="Higher values = more creative, Lower values = more conservative")
        
        st.divider()
        
        st.info(" **Tip:** For best results, use clear images with distinct subjects")
    
    # Main content
    st.subheader(" Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Image info
            with st.expander("Image Information"):
                st.write(f"**Format:** {image.format}")
                st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            st.subheader(" Generated Captions")
            
            if not st.session_state.model_loaded:
                st.error(" No model available. Please load a pretrained model from the sidebar.")
            else:
                if st.button(" Generate Captions", type="primary", use_container_width=True):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image.save(tmp_file.name, format='JPEG')
                            
                            # Create progress tracking
                            progress_container = st.container()
                            with progress_container:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Step 1: Processing image
                                status_text.text(" Processing image...")
                                progress_bar.progress(30)
                                time.sleep(0.2)
                                
                                # Step 2: Extracting features
                                status_text.text(" Extracting image features...")
                                progress_bar.progress(50)
                                time.sleep(0.2)
                                
                                # Step 3: Generating captions
                                status_text.text(" Generating captions...")
                                progress_bar.progress(70)
                                
                                candidates = st.session_state.caption_generator.generate_caption_beam_search(
                                    tmp_file.name, 
                                    beam_size=beam_size
                                )
                                
                                # Step 4: Complete
                                progress_bar.progress(100)
                                status_text.text(" Complete!")
                                time.sleep(0.3)
                                
                                # Clear progress indicators
                                progress_bar.empty()
                                status_text.empty()
                            
                            # Display results
                            st.success(" Captions generated successfully!")
                            
                            for i, candidate in enumerate(candidates):
                                with st.expander(f"Caption {i+1} (Score: {candidate['score']:.2f})", 
                                               expanded=i==0):
                                    st.markdown(f"### {candidate['caption']}")
                                    
                                    # Show confidence bar
                                    confidence = min(max((candidate['score'] + 10) / 10, 0), 1)
                                    st.progress(confidence)
                                    st.caption(f"Confidence: {confidence:.1%}")
                                    
                                    # Copy button
                                    st.code(candidate['caption'], language=None)
                            
                            # Clean up temp file
                            os.unlink(tmp_file.name)
                            
                    except Exception as e:
                        error_message = handle_image_error(e, "caption_generation")
                        st.error(f" {error_message}")
    
    # Example images section
    st.markdown("---")
    st.subheader(" Try Example Images")
    
    examples_dir = ImageCaptionConfig.EXAMPLES_DIR
    if os.path.exists(examples_dir) and os.listdir(examples_dir):
        example_images = [f for f in os.listdir(examples_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if example_images:
            cols = st.columns(min(len(example_images), 4))
            for idx, img_name in enumerate(example_images[:4]):
                with cols[idx]:
                    img_path = os.path.join(examples_dir, img_name)
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True)
                    st.caption(img_name)
        else:
            st.info("No example images available. Add images to `data/examples/` folder.")
    else:
        st.info(" Add example images to the `data/examples/` folder to see them here!")
    
    # Training section (simplified)
    st.markdown("---")
    st.subheader(" Model Training")
    
    with st.expander("Train New Model (Advanced)"):
        st.warning(" Training requires a dataset with images and captions")
        st.info("""
        **To train a model:**
        1. Prepare your dataset in the `data/images` folder
        2. Create a `captions.txt` file with image-caption pairs
        3. Run the training script separately
        4. The trained model will be saved in the `models` folder
        """)
        
        st.code("""
# Run training from command line:
python train_pipeline.py
        """, language="bash")

if __name__ == "__main__":
    # Try to load pretrained model on startup
    if not st.session_state.get('model_loaded', False):
        with st.spinner("Checking for pretrained model..."):
            load_pretrained_model()
    
    main()