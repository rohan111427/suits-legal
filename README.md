# Suits AI - Legal Assistant

A full-stack AI-powered legal document analysis and chat application built with FastAPI, HTML/CSS frontend, and SQLite database.

## Features

- **User Authentication**: Email/password-based registration and login
- **Document Upload**: Support for PDF, images (JPG, PNG), and text files
- **Text Extraction**: Automatic text extraction using PyMuPDF (PDF) and Tesseract OCR (images)
- **AI Analysis**: 
  - Document summarization using Facebook BART
  - Legal document structuring using Gemini API
- **Interactive Chat**: Chat with uploaded documents using contextual AI responses
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS

## Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: Jinja2 Templates + HTML/CSS/JavaScript
- **Database**: SQLite with SQLAlchemy ORM
- **AI Models**: 
  - Facebook BART (`facebook/bart-large-cnn`) for summarization
  - Google Gemini 1.5 Flash for document analysis and chat
- **Authentication**: Session-based with JWT tokens

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd suits-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies**:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # For macOS
   brew install tesseract
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Download and cache the BART model** (optional, for faster startup):
   ```bash
   python3 download_model.py
   ```

6. **Required Environment Variables**:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   SECRET_KEY=your_secret_key_for_sessions
   DATABASE_URL=sqlite:///./suits_ai.db
   ```

## Running the Application

1. **Start the server**:
   ```bash
   python run.py
   ```

2. **Access the application**:
   - Open your browser to `http://localhost:8000`
   - Create an account or sign in
   - Upload legal documents and start chatting!

## API Endpoints

- `GET /` - Landing page
- `GET /register` - Registration page
- `POST /register` - Create new user account
- `GET /login` - Login page
- `POST /login` - User authentication
- `GET /dashboard` - Main chat interface
- `POST /upload` - Upload and analyze documents
- `POST /chat` - Send chat messages
- `POST /new-chat` - Start new chat session
- `GET /logout` - User logout

## Database Schema

### Users
- `id` - Primary key
- `email` - User email (unique)
- `password` - Hashed password

### Documents
- `id` - Primary key
- `user_id` - Foreign key to users
- `filename` - Original filename
- `raw_text` - Extracted text content
- `summary` - BART-generated summary
- `structured_json` - Gemini-analyzed structure
- `created_at` - Upload timestamp

### Chat Sessions
- `id` - Primary key
- `user_id` - Foreign key to users
- `document_id` - Foreign key to documents (nullable)
- `created_at` - Session creation timestamp

### Chat History
- `id` - Primary key
- `session_id` - Foreign key to chat_sessions
- `user_input` - User message
- `assistant_reply` - AI response
- `timestamp` - Message timestamp

## Deployment

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload all files to the Space
3. Set environment variables in Space settings
4. The app will automatically deploy

### Local Production

1. **Install production server**:
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**:
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

### Docker Deployment

1. **Build Docker image**:
   ```bash
   docker build -t suits-ai .
   ```

2. **Run container**:
   ```bash
   docker run -p 8000:8000 --env-file .env suits-ai
   ```

## Configuration

### Gemini API Setup
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env` file: `GEMINI_API_KEY=your_key_here`

### Tesseract OCR Setup
- The application uses Tesseract for OCR text extraction from images
- Install using system package manager (see installation steps above)

## Usage

1. **Registration**: Create account with email and password
2. **Login**: Sign in to access dashboard
3. **Upload Document**: Click the "+" button to upload PDF, image, or text files
4. **AI Analysis**: Automatic analysis provides:
   - Document summary
   - Case type identification
   - Legal issue breakdown
   - Procedural history
   - Final decision analysis
5. **Chat**: Ask questions about uploaded documents
6. **New Chat**: Start fresh conversations for new documents

## File Structure

```
suits-ai/
├── main.py                 # FastAPI application
├── run.py                  # Startup script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── models/
│   └── database.py         # Database models
├── utils/
│   ├── auth.py            # Authentication utilities
│   ├── text_extraction.py # File processing
│   └── ai_services.py     # AI model integration
├── frontend/
│   └── templates/         # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── login.html
│       ├── register.html
│       └── dashboard.html
└── static/                # CSS/JS files (auto-created)
```

## Security Features

- Password hashing with bcrypt
- Session-based authentication
- CSRF protection
- File type validation
- Input sanitization

## Troubleshooting

### Common Issues

1. **BART Model Loading**: 
   - First run may take time to download model
   - Ensure sufficient disk space (~1.5GB)

2. **Tesseract OCR**:
   - Ensure tesseract is installed and in PATH
   - Check OCR language packs if needed

3. **Gemini API**:
   - Verify API key is correct
   - Check API quota and usage limits

4. **Database**:
   - SQLite file is auto-created
   - Ensure write permissions in app directory

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Support

For issues and questions, please create an issue in the repository.
