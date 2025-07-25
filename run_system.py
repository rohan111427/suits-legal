#!/usr/bin/env python3
"""
Suits AI - Legal Assistant Application
Run script to start the FastAPI server using system-installed packages
"""

import os
import sys
import subprocess

# Add user's Python bin directory to PATH
user_bin_path = os.path.expanduser("~/Library/Python/3.13/bin")
current_path = os.environ.get("PATH", "")
if user_bin_path not in current_path:
    os.environ["PATH"] = f"{user_bin_path}:{current_path}"

# Set PYTHONPATH to include current directory
os.environ["PYTHONPATH"] = "."

# Run the application
if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "run.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped.")
