#!/bin/bash

# Suits AI - Quick Start Script
echo "🔧 Starting Suits AI Legal Assistant..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before running the application"
fi

# Create static directory if it doesn't exist
mkdir -p static

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit .env file with your Gemini API key"
echo "2. Run: python3 run.py"
echo "3. Open browser to: http://localhost:8000"
echo ""
echo "🎯 For more information, see README.md"
