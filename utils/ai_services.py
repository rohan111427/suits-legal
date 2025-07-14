import os
import json
import warnings
import re
from typing import List, Dict, Tuple, Optional
from transformers import pipeline
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from models.database import MessageEmbedding, ChatHistory

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, message=".*NotOpenSSLWarning.*")

load_dotenv()

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model configuration
GEMINI_MODEL_NAME = 'gemini-2.5-pro'  # Upgrade to Gemini 2.5 Pro model

# Initialize BART summarizer
try:
    from pathlib import Path
    import os
    
    # Set up model cache directory
    model_cache_dir = Path("./model_cache")
    model_cache_dir.mkdir(exist_ok=True)
    
    # Set environment variable for HuggingFace cache
    os.environ['HF_HOME'] = str(model_cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(model_cache_dir)
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"Warning: Could not load BART model: {e}")
    summarizer = None

def summarize_text(text):
    """Summarize text using Facebook BART model"""
    if not summarizer:
        return "Summarization not available - model not loaded"
    
    try:
        # Clean and validate input text
        text = text.strip()
        if not text:
            return "No text provided for summarization"
        
        # Calculate appropriate parameters based on input length
        input_words = len(text.split())
        
        # Skip summarization for very short texts
        if input_words < 10:
            return f"Text too short for summarization (only {input_words} words). Original text: {text}"
        
        # BART tokenizer max length is 1024 tokens (roughly 750-800 words)
        max_words_per_chunk = 750
        
        # Dynamic parameter calculation
        if input_words <= max_words_per_chunk:
            # Single chunk processing
            max_length = min(150, max(30, input_words // 3))  # Summary should be 1/3 of input
            min_length = max(10, input_words // 10)  # Minimum 1/10 of input
            
            result = summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']
        else:
            # Multi-chunk processing
            words = text.split()
            chunks = []
            
            # Split into word-based chunks to avoid token limit issues
            for i in range(0, len(words), max_words_per_chunk):
                chunk_words = words[i:i + max_words_per_chunk]
                chunks.append(" ".join(chunk_words))
            
            summaries = []
            
            for chunk in chunks:
                chunk_words = len(chunk.split())
                max_length = min(100, max(20, chunk_words // 3))
                min_length = max(10, chunk_words // 10)
                
                result = summarizer(
                    chunk, 
                    max_length=max_length, 
                    min_length=min_length, 
                    do_sample=False,
                    truncation=True
                )
                summaries.append(result[0]['summary_text'])
            
            # Combine summaries
            combined_summary = " ".join(summaries)
            combined_words = len(combined_summary.split())
            
            # Final summarization if combined summary is still long
            if combined_words > max_words_per_chunk:
                final_max_length = min(200, max(50, combined_words // 2))
                final_min_length = max(20, combined_words // 8)
                
                result = summarizer(
                    combined_summary, 
                    max_length=final_max_length, 
                    min_length=final_min_length, 
                    do_sample=False,
                    truncation=True
                )
                return result[0]['summary_text']
            else:
                return combined_summary
                
    except Exception as e:
        return f"Error generating summary: {str(e)}"

class RAGSystem:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Enhanced keyword extraction using legal-specific filtering"""
        # Remove common stop words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Clean text and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and count frequency
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top_k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]

    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generates embeddings for the provided text"""
        embedding = self.model.encode(text)
        return embedding

    def retrieve_relevant_messages(self, session: Session, new_message_embedding: np.ndarray, top_n: int = 5) -> List[ChatHistory]:
        """Retrieves top-N relevant messages based on cosine similarity of embeddings"""
        messages = session.query(MessageEmbedding).all()
        if not messages:
            return []

        message_embeddings = np.array([msg.embedding for msg in messages])
        similarities = cosine_similarity([new_message_embedding], message_embeddings)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]  # Top-N scores
        
        relevant_messages = []
        for i in top_indices:
            chat_msg = session.query(ChatHistory).get(messages[i].message_id)
            if chat_msg:
                relevant_messages.append(chat_msg)
        
        return relevant_messages

    def construct_prompt(self, context_messages: List[ChatHistory], user_query: str) -> str:
        """Constructs a prompt with retrieved context and new user query"""
        context_text = "\n".join([f"Past message: {msg.content}" for msg in context_messages])
        prompt = f"You are Suits AI, a legal assistant. Using the context, answer the user query.\n\nContext:\n{context_text}\n\nUser Query: {user_query}\n\nProvide a contextual and professional response."
        return prompt

def structure_legal_document(full_text, summary):
    """Structure legal document using Gemini API with improved prompt"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        prompt = f"""
        You are a legal analyst AI. Based on the Indian court judgment and its summary below, extract structured legal insights.
        
        BART Summary:
        {summary}
        
        Full Case Text:
        {full_text[:8000]}  # Limit to avoid token limits
        
        IMPORTANT: Format your response with proper spacing and structure. Use line breaks between sections for readability.
        
        Respond in this exact structure:
        
        ---
        Type of the case:
        <give the case type>
        
        Summary of Case:
        <One-paragraph high-level summary>
        
        Background of the Case:
        <Who filed, against whom, what for?>
        
        Procedural History:
        <Lower courts and rulings>
        
        Core Legal Issue:
        <Main legal question>
        
        Supreme Court's Judgment and Reasoning:
        <How the court interpreted law, key reasoning>
        
        Appellants' Arguments:
        - ...
        - ...
        
        Respondents' Arguments:
        - ...
        - ...
        
        Final Decision (Holding):
        <Who won, what was ordered?>
        ---
        """
        
        response = model.generate_content(prompt)
        
        # Parse the structured response and convert to JSON format
        try:
            response_text = response.text
            
            # Extract sections using text parsing
            sections = {}
            
            # Extract case type
            if "Type of the case:" in response_text:
                case_type_start = response_text.find("Type of the case:") + len("Type of the case:")
                case_type_end = response_text.find("Summary of Case:", case_type_start)
                sections["case_type"] = response_text[case_type_start:case_type_end].strip()
            else:
                sections["case_type"] = "Not specified"
            
            # Extract summary
            if "Summary of Case:" in response_text:
                summary_start = response_text.find("Summary of Case:") + len("Summary of Case:")
                summary_end = response_text.find("Background of the Case:", summary_start)
                sections["summary"] = response_text[summary_start:summary_end].strip()
            else:
                sections["summary"] = summary
            
            # Extract background
            if "Background of the Case:" in response_text:
                bg_start = response_text.find("Background of the Case:") + len("Background of the Case:")
                bg_end = response_text.find("Procedural History:", bg_start)
                sections["background"] = response_text[bg_start:bg_end].strip()
            else:
                sections["background"] = "Not specified"
            
            # Extract procedural history
            if "Procedural History:" in response_text:
                proc_start = response_text.find("Procedural History:") + len("Procedural History:")
                proc_end = response_text.find("Core Legal Issue:", proc_start)
                sections["procedural_history"] = response_text[proc_start:proc_end].strip()
            else:
                sections["procedural_history"] = "Not specified"
            
            # Extract legal issue
            if "Core Legal Issue:" in response_text:
                issue_start = response_text.find("Core Legal Issue:") + len("Core Legal Issue:")
                issue_end = response_text.find("Supreme Court's Judgment and Reasoning:", issue_start)
                sections["legal_issue"] = response_text[issue_start:issue_end].strip()
            else:
                sections["legal_issue"] = "Not specified"
            
            # Extract judgment
            if "Supreme Court's Judgment and Reasoning:" in response_text:
                judgment_start = response_text.find("Supreme Court's Judgment and Reasoning:") + len("Supreme Court's Judgment and Reasoning:")
                judgment_end = response_text.find("Appellants' Arguments:", judgment_start)
                sections["judgment"] = response_text[judgment_start:judgment_end].strip()
            else:
                sections["judgment"] = "Not specified"
            
            # Extract final decision
            if "Final Decision (Holding):" in response_text:
                decision_start = response_text.find("Final Decision (Holding):") + len("Final Decision (Holding):")
                sections["final_decision"] = response_text[decision_start:].strip()
            else:
                sections["final_decision"] = "Not specified"
            
            return sections
            
        except Exception as parse_error:
            print(f"Error parsing response: {parse_error}")
            # Fallback to basic structure
            return {
                "case_type": "Document Analysis",
                "summary": summary,
                "background": "Not specified",
                "procedural_history": "Not specified",
                "legal_issue": "Not specified",
                "judgment": "Not specified",
                "final_decision": "Not specified"
            }
            
    except Exception as e:
        return {
            "case_type": "Analysis Error",
            "summary": f"Error analyzing document: {str(e)}",
            "background": "Not available",
            "procedural_history": "Not available",
            "legal_issue": "Not available", 
            "judgment": "Not available",
            "final_decision": "Not available"
        }

def chat_with_document(user_input, document=None):
    """Chat with document using Gemini API"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        if document:
            context = f"""
            You are Suits AI, a legal assistant helping analyze this document:
            
            DOCUMENT: {document.filename}
            
            FULL TEXT:
            {document.raw_text[:6000]}  # Limit context
            
            SUMMARY:
            {document.summary}
            
            STRUCTURED ANALYSIS:
            {json.dumps(document.structured_json, indent=2)}
            
            USER QUESTION: {user_input}
            
            Provide a helpful, accurate response based on the document content. Be professional and concise.
            """
        else:
            context = f"""
            You are Suits AI, a legal assistant. The user has asked: {user_input}
            
            Since no document has been uploaded, you can surf the web to provide an informative answer.
            Be professional and helpful.
            """
        
        response = model.generate_content(context)
        return response.text
        
    except Exception as e:
        return f"I apologize, but I encountered an error processing your request: {str(e)}"

def chat_with_rag(user_input: str, document=None, session_id: str = None, db_session: Session = None) -> str:
    """Enhanced chat function with RAG capabilities and web search fallback"""
    try:
        # Initialize RAG system
        rag_system = RAGSystem()
        
        # Generate embeddings for the user input
        user_embedding = rag_system.generate_embeddings(user_input)
        
        # Extract keywords from user input
        keywords = rag_system.extract_keywords(user_input)
        
        # Initialize context from retrieved messages
        context_messages = []
        
        # If database session and session_id are provided, retrieve relevant messages
        if db_session and session_id:
            context_messages = rag_system.retrieve_relevant_messages(db_session, user_embedding, top_n=5)
        
        # Initialize Gemini model with upgraded model name
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Build enhanced context
        if document:
            # Document-based context with RAG
            context = f"""
            You are Suits AI, a helpful and professional legal assistant. You have access to a user's legal document and prior conversation history.
            
            DOCUMENT: {document.filename}
            
            FULL TEXT:
            {document.raw_text[:4000]}  # Limit context to manage token size
            
            SUMMARY:
            {document.summary}
            
            STRUCTURED ANALYSIS:
            {json.dumps(document.structured_json, indent=2)}
            
            CONVERSATION HISTORY (most relevant messages):
{'\n'.join([f"- User: {msg.user_input}\n  Assistant: {msg.assistant_reply}" for msg in context_messages[:3] if msg.user_input and msg.assistant_reply])}
            
            USER QUESTION: {user_input}
            
            EXTRACTED KEYWORDS: {', '.join(keywords)}
            
            Provide a detailed, polite, and context-aware response, referencing the document and conversation history when applicable. If the query is unrelated to the document or context, please provide a helpful legal answer based on your knowledge.
            """
        else:
            # General context with RAG and web search fallback
            context = f"""
            You are Suits AI, a helpful and professional legal assistant with extensive legal knowledge.
            
            CONVERSATION HISTORY (most relevant messages):
{'\n'.join([f"- User: {msg.user_input}\n  Assistant: {msg.assistant_reply}" for msg in context_messages[:5] if msg.user_input and msg.assistant_reply])}
            
            USER QUESTION: {user_input}
            
            EXTRACTED KEYWORDS: {', '.join(keywords)}
            
            Provide a detailed, polite, and context-aware response. Use the conversation history to provide better context and continuity. If you cannot answer based on the conversation history, use your legal knowledge to assist the user.
            """
        
        # Generate response
        response = model.generate_content(context)
        response_text = response.text
        
        # Store embeddings and keywords in database if session is provided
        if db_session and session_id:
            try:
                # Find the latest user message in chat history
                latest_user_message = db_session.query(ChatHistory).filter_by(
                    session_id=session_id,
                    user_input=user_input
                ).order_by(ChatHistory.timestamp.desc()).first()
                
                if latest_user_message:
                    # Store embedding and keywords for the user message
                    user_msg_embedding = MessageEmbedding(
                        message_id=latest_user_message.id,
                        user_id=latest_user_message.session.user_id if latest_user_message.session else None,
                        session_id=session_id,
                        message_content=user_input,
                        message_type='user',
                        keywords=keywords,
                        embedding=user_embedding.tolist(),
                        legal_category='general',
                        created_at=latest_user_message.timestamp
                    )
                    db_session.add(user_msg_embedding)
                    db_session.commit()
            except Exception as db_error:
                print(f"Database error storing embeddings: {db_error}")
                db_session.rollback()
        
        return response_text
    except Exception as e:
        return f"I apologize, but I encountered an error processing your request: {str(e)}"
