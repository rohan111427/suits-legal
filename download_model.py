#!/usr/bin/env python3
"""
Script to download and cache the BART model locally
"""

import os
from pathlib import Path
from transformers import pipeline

def download_model():
    """Download and cache the BART model"""
    print("🔄 Downloading BART model for local caching...")
    
    # Create model cache directory
    model_cache_dir = Path("./model_cache")
    model_cache_dir.mkdir(exist_ok=True)
    
    # Set environment variable for HuggingFace cache
    os.environ['HF_HOME'] = str(model_cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(model_cache_dir)
    
    try:
        # Initialize pipeline with local cache
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn"
        )
        
        print("✅ BART model downloaded and cached successfully!")
        print(f"📁 Model cached in: {model_cache_dir.absolute()}")
        
        # Test the model with a simple example
        test_text = "This is a test document to verify the model is working correctly."
        result = summarizer(test_text, max_length=50, min_length=10, do_sample=False)
        print(f"🧪 Test summary: {result[0]['summary_text']}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_model()
    if success:
        print("\n🎉 Model setup complete! Your application will now load faster.")
    else:
        print("\n⚠️  Model download failed. Please check your internet connection.")
