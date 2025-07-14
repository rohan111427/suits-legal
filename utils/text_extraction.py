import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return clean_text(text)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_image(file_bytes):
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return clean_text(text)
    except Exception as e:
        raise Exception(f"Error extracting text from image: {str(e)}")

def extract_text_from_text_file(file_bytes):
    """Extract text from plain text file"""
    try:
        text = file_bytes.decode('utf-8')
        return clean_text(text)
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode('latin-1')
            return clean_text(text)
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

def clean_text(text):
    """Clean extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    return text.strip()

def extract_text_from_file(file_bytes, filename):
    """Main function to extract text based on file type"""
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
        return extract_text_from_image(file_bytes)
    elif file_ext in ['txt', 'text']:
        return extract_text_from_text_file(file_bytes)
    else:
        raise Exception(f"Unsupported file type: {file_ext}")
