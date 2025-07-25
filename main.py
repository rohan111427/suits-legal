import os
from datetime import timedelta
from fastapi import FastAPI, Request, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.database import get_db, User, Document, ChatSession, ChatHistory
from utils.auth import verify_password, get_password_hash, create_access_token, verify_token
from utils.text_extraction import extract_text_from_file
from utils.ai_services import summarize_text, structure_legal_document, chat_with_document, chat_with_rag

app = FastAPI(title="Suits AI", version="1.0.0")
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY', 'your-secret-key'))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="frontend/templates")

# Helper function to get current user
def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.session.get("access_token")
    if not token:
        return None
    
    user_id = verify_token(token)
    if not user_id:
        return None
    
    user = db.query(User).filter(User.id == user_id).first()
    return user

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(request: Request, db: Session = Depends(get_db)):
    form_data = await request.form()
    name = form_data["name"]
    email = form_data["email"]
    password = form_data["password"]
    
    user = db.query(User).filter(User.email == email).first()
    if user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered"})

    hashed_password = get_password_hash(password)
    new_user = User(name=name, email=email, password=hashed_password)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, db: Session = Depends(get_db)):
    form_data = await request.form()
    email = form_data["email"]
    password = form_data["password"]

    user = db.query(User).filter(User.email == email).first()
    if not user:
        return RedirectResponse(url="/register", status_code=status.HTTP_302_FOUND)
    
    if not verify_password(password, user.password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid password"})

    # Create session
    token_expires = timedelta(minutes=30)
    token = create_access_token(data={"sub": str(user.id)}, expires_delta=token_expires)
    request.session["access_token"] = token

    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login")

    # Get all chat sessions for the user ordered by creation time (newest first)
    chat_sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user.id
    ).order_by(ChatSession.created_at.desc()).all()

    # Determine active session
    active_session = chat_sessions[0] if chat_sessions else None

    # Get chat history for active session
    chat_history = []
    if active_session:
        chat_history = db.query(ChatHistory).filter(
            ChatHistory.session_id == active_session.id
        ).order_by(ChatHistory.timestamp).all()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "chat_sessions": chat_sessions,
        "chat_history": chat_history,
        "active_session_id": active_session.id if active_session else None
    })

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text
        extracted_text = extract_text_from_file(file_content, file.filename)
        
        # Save document (without processing)
        document = Document(
            user_id=user.id,
            filename=file.filename,
            raw_text=extracted_text,
            summary=None,  # Will be processed on first user input
            structured_json=None  # Will be processed on first user input
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Create new chat session for this document
        chat_session = ChatSession(user_id=user.id, document_id=document.id)
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        
        # Add upload confirmation to chat history (no analysis yet)
        chat_entry = ChatHistory(
            session_id=chat_session.id,
            user_input=None,  # No user input yet
            assistant_reply=f"📄 Document '{file.filename}' has been uploaded successfully!\n\nI'm ready to help you with any questions about this document. What would you like to know?"
        )
        db.add(chat_entry)
        db.commit()
        
        return JSONResponse({
            "success": True,
            "document_id": document.id,
            "session_id": chat_session.id,
            "filename": file.filename
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/chat")
async def chat(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    form_data = await request.form()
    user_input = form_data["message"]
    try:
        session_id = int(form_data["session_id"])
    except (ValueError, TypeError):
        return JSONResponse({"error": "Missing or invalid session_id"}, status_code=400)
    
    # Get session
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    # Get document context if exists
    document = None
    if session.document_id:
        document = db.query(Document).filter(Document.id == session.document_id).first()
        
        # Check if this is the first user input and document needs processing
        if document and document.summary is None and document.structured_json is None:
            # This is the first user input - process the document now
            try:
                # Process the document
                summary = summarize_text(document.raw_text)
                structured_data = structure_legal_document(document.raw_text, summary)
                
                # Update the document with processed data
                document.summary = summary
                document.structured_json = structured_data
                db.commit()
                
                # Add processing notification to chat
                processing_message = f"🔍 Processing your document '{document.filename}'...\n\n"
                analysis_text = format_analysis(structured_data)
                full_analysis = processing_message + analysis_text + "\n\n" + "Now I'm ready to answer your question!"
                
                # Add analysis to chat history
                analysis_entry = ChatHistory(
                    session_id=session_id,
                    user_input=None,
                    assistant_reply=full_analysis
                )
                db.add(analysis_entry)
                db.commit()
                
            except Exception as e:
                # If processing fails, continue with the chat but note the error
                error_message = f"⚠️ There was an issue processing your document, but I can still help with general questions about it.\n\nError: {str(e)}"
                error_entry = ChatHistory(
                    session_id=session_id,
                    user_input=None,
                    assistant_reply=error_message
                )
                db.add(error_entry)
                db.commit()
    
    # Generate response using RAG
    assistant_reply = chat_with_rag(user_input, document, session_id, db)
    
    # Save to history
    chat_entry = ChatHistory(
        session_id=session_id,
        user_input=user_input,
        assistant_reply=assistant_reply
    )
    db.add(chat_entry)
    db.commit()
    
    return JSONResponse({
        "success": True,
        "reply": assistant_reply
    })

@app.post("/summarize-text")
async def summarize_text_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    summary = summarize_text(text)

    return JSONResponse({
        "success": True,
        "summary": summary
    })

@app.post("/analyze-text")
async def analyze_text_endpoint(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    data = await request.json()
    text = data.get("text", "")
    
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    try:
        # Generate summary
        summary = summarize_text(text)
        
        # Structure with Gemini
        structured_data = structure_legal_document(text, summary)
        
        # Save as document
        document = Document(
            user_id=user.id,
            filename="Text Analysis",
            raw_text=text,
            summary=summary,
            structured_json=structured_data
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Create new chat session
        chat_session = ChatSession(user_id=user.id, document_id=document.id)
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        
        # Format analysis
        analysis_text = format_analysis(structured_data)
        
        # Add to chat history
        chat_entry = ChatHistory(
            session_id=chat_session.id,
            user_input="Text analysis request",
            assistant_reply=analysis_text
        )
        db.add(chat_entry)
        db.commit()
        
        return JSONResponse({
            "success": True,
            "session_id": chat_session.id,
            "analysis": analysis_text
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/new-chat")
async def new_chat(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    # Create new empty chat session
    new_session = ChatSession(user_id=user.id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    return JSONResponse({
        "success": True,
        "session_id": new_session.id
    })

@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: int, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    # Get chat history
    chat_history = db.query(ChatHistory).filter(
        ChatHistory.session_id == session_id
    ).order_by(ChatHistory.timestamp).all()
    
    return JSONResponse({
        "success": True,
        "history": [
            {
                "user_input": chat.user_input,
                "assistant_reply": chat.assistant_reply,
                "timestamp": chat.timestamp.isoformat()
            }
            for chat in chat_history
        ]
    })

@app.delete("/chat-session/{session_id}")
async def delete_chat_session(session_id: int, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    # Delete chat history first (due to foreign key constraint)
    db.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
    
    # Delete the session
    db.delete(session)
    db.commit()
    
    return JSONResponse({
        "success": True,
        "message": "Chat session deleted successfully"
    })

@app.get("/get-chat-sessions")
async def get_chat_sessions(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return JSONResponse({"success": False, "error": "Unauthorized"}, status_code=401)
    
    # Get all chat sessions for the user ordered by creation time (newest first)
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user.id
    ).order_by(ChatSession.created_at.desc()).all()
    
    # Build response list
    sessions_list = []
    for session in sessions:
        # Get first message from chat history
        first_message = ""
        first_chat = db.query(ChatHistory).filter(
            ChatHistory.session_id == session.id
        ).order_by(ChatHistory.timestamp).first()
        
        if first_chat:
            first_message = first_chat.user_input
        
        # Get document info if exists
        document_filename = None
        if session.document_id:
            document = db.query(Document).filter(Document.id == session.document_id).first()
            if document:
                document_filename = document.filename
        
        sessions_list.append({
            "id": session.id,
            "document_id": session.document_id,
            "document_filename": document_filename,
            "first_message": first_message,
            "created_at": session.created_at.isoformat()
        })
    
    return JSONResponse({
        "success": True,
        "sessions": sessions_list
    })

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse("test_dashboard.html", {"request": request})

def format_analysis(structured_data):
    """Format the structured analysis for display with clean presentation"""
    if not structured_data:
        return "Analysis not available"
    
    output = []
    
    if structured_data.get('case_type') and structured_data['case_type'] != "Not specified":
        output.append("Case Type:")
        output.append(structured_data['case_type'])
        output.append("")
    
    if structured_data.get('summary') and structured_data['summary'] != "Not specified":
        output.append("Summary of Case:")
        output.append(structured_data['summary'])
        output.append("")
    
    if structured_data.get('background') and structured_data['background'] != "Not specified":
        output.append("Background of the Case:")
        output.append(structured_data['background'])
        output.append("")
    
    if structured_data.get('procedural_history') and structured_data['procedural_history'] != "Not specified":
        output.append("Procedural History:")
        output.append(structured_data['procedural_history'])
        output.append("")
    
    if structured_data.get('legal_issue') and structured_data['legal_issue'] != "Not specified":
        output.append("Core Legal Issue:")
        output.append(structured_data['legal_issue'])
        output.append("")
    
    if structured_data.get('judgment') and structured_data['judgment'] != "Not specified":
        output.append("Supreme Court's Judgment and Legal Reasoning:")
        output.append(structured_data['judgment'])
        output.append("")
    
    if structured_data.get('final_decision') and structured_data['final_decision'] != "Not specified":
        output.append("Final Decision (Holding):")
        output.append(structured_data['final_decision'])
    
    return "\n".join(output)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
