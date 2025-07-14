#!/usr/bin/env python3
"""
Clean startup script for Suits AI that suppresses warnings
"""

import os
import sys
import warnings

# Suppress all warnings for a clean startup
warnings.filterwarnings("ignore")

# Set environment variables to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import and run the main application
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🚀 Starting Suits AI (Clean Mode)...")
    print("📱 Open your browser to: http://localhost:8000")
    print("⚡ All warnings suppressed for clean output")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )
