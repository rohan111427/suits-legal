#!/usr/bin/env python3
"""
Suits AI - Legal Assistant Application
Run script to start the FastAPI server
"""

import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", ".")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )
