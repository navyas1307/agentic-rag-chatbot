"""
FastAPI Backend API
Serves the chatbot frontend and handles query requests
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import query_agent
import os
from pathlib import Path

# Create FastAPI app
app = FastAPI(
    title="Customer Support Chatbot API",
    description="AI-powered customer support chatbot with RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    success: bool
    question: str
    answer: str
    context_preview: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Load HTML template
def load_html():
    html_path = Path("index.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html not found</h1>"

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main chatbot interface"""
    return load_html()

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat queries"""
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(
                status_code=400,
                detail="Please provide a question"
            )
        
        # Query the agent
        result = query_agent(question)
        
        return ChatResponse(
            success=True,
            question=result['question'],
            answer=result['answer'],
            context_preview=result['context_used']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Chatbot service is running"
    )

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("CUSTOMER SUPPORT CHATBOT SERVER (FastAPI)")
    print("="*60)
    print("\n✓ Server will be available at:")
    print("   - http://localhost:8000")
    print("   - http://127.0.0.1:8000")
    print("\n✓ API Documentation:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n✓ API endpoint: http://localhost:8000/api/chat")
    print("\nMake sure you have:")
    print("  1. Run ingest.py to create the knowledge base")
    print("  2. Added your GEMINI_API_KEY to .env file")
    print("  3. (Optional) Ollama running for fallback")
    print("\n⚠️  Use 'localhost:8000' or '127.0.0.1:8000' in your browser")
    print("    (NOT 0.0.0.0:8000)")
    print("="*60 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )