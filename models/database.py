from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
import json

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./suits_ai.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    
    documents = relationship("Document", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    raw_text = Column(Text)
    summary = Column(Text)
    structured_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="documents")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="chat_sessions")
    document = relationship("Document")
    chat_history = relationship("ChatHistory", back_populates="session")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    user_input = Column(Text)
    assistant_reply = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ChatSession", back_populates="chat_history")
    # Relationship with RAG message storage
    message_embeddings = relationship("MessageEmbedding", back_populates="chat_history")

class MessageEmbedding(Base):
    """Store message embeddings and keywords for RAG retrieval"""
    __tablename__ = "message_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    
    # Message content and metadata
    message_content = Column(Text)  # The actual message content
    message_type = Column(String)  # 'user' or 'assistant'
    keywords = Column(JSON)  # Extracted keywords as JSON array
    
    # Embedding vectors (stored as JSON for simplicity)
    embedding = Column(JSON)  # 768-dimensional vector from sentence-transformers
    
    # Legal context categorization
    legal_category = Column(String)  # 'judgment', 'arguments', 'background', 'general'
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chat_history = relationship("ChatHistory", back_populates="message_embeddings")
    user = relationship("User")
    session = relationship("ChatSession")
    
    # Index for faster similarity searches
    __table_args__ = (
        Index('idx_user_session', 'user_id', 'session_id'),
        Index('idx_legal_category', 'legal_category'),
        Index('idx_message_type', 'message_type'),
    )

class DocumentSegment(Base):
    """Store document segments with embeddings for enhanced RAG"""
    __tablename__ = "document_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Segment content and metadata
    segment_text = Column(Text)  # Text segment
    segment_type = Column(String)  # 'judgment', 'arguments', 'background', 'summary'
    keywords = Column(JSON)  # Extracted keywords
    embedding = Column(JSON)  # Embedding vector
    
    # Position in document
    segment_order = Column(Integer)  # Order within document
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_user', 'document_id', 'user_id'),
        Index('idx_segment_type', 'segment_type'),
    )

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
