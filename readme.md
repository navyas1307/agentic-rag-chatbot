# Company Chatbot - Complete Setup Guide

## Project Overview

This is a production-ready **Agentic AI Customer Support Chatbot** that demonstrates:
- **RAG (Retrieval Augmented Generation)** - Retrieves relevant info from company documents
- **Vector Database** - ChromaDB for efficient semantic search
- **LangGraph** - Agentic workflow with decision-making
- **Dual LLM Support** - Gemini (primary, free tier) + Ollama (fallback)
- **FastAPI Backend** - Modern async Python web framework
- **Modern Frontend** - Clean HTML/CSS/JS interface
- **Multi-format Support** - Handles both PDF and TXT files

## Project Structure

```
project/
├── app.py                      # FastAPI backend server
├── graph.py                    # LangGraph agentic workflow
├── ingest.py                   # Data ingestion & vectorization
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── index.html                  # Frontend interface (root level)
├── data/                       # Company knowledge base
│   ├── shipping_policy.txt
│   ├── returns_policy.txt
│   ├── payment_methods.txt
│   ├── product_catalog.txt
│   ├── customer_support.txt
│   └── [any PDFs you add]
└── chroma_db/                  # Vector database (auto-created)
```

## Setup Instructions

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Get Gemini API Key (FREE)

1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

### Step 3: Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Your Gemini API Key (REQUIRED for free tier)
GEMINI_API_KEY=your_actual_api_key_here

# Ollama Configuration (OPTIONAL - only if you want fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest
```

### Step 4: Add Your Company Data

Place your company documents in the `data/` folder:
- `.txt` files for text data
- `.pdf` files for PDF documents

The provided sample files include:
- Shipping policy
- Returns policy
- Payment methods
- Product catalog
- Customer support info

### Step 5: Ingest Data (Create Vector Database)

```bash
python ingest.py
```

This will:
- Read all TXT and PDF files from `data/` folder
- Extract and chunk the text
- Generate embeddings using sentence-transformers
- Store in ChromaDB vector database

You should see output like:
```
Loading embedding model...
Initializing ChromaDB...
Found 5 TXT files and 0 PDF files
Processing: shipping_policy.txt
  Created 15 chunks
...
✓ Successfully ingested 87 chunks from 5 files
✓ Vector database saved to ./chroma_db
```

### Step 6: Run the Application

```bash
python app.py
```

The server will start on http://localhost:8000

Open your browser and visit: **http://localhost:8000** or **http://127.0.0.1:8000**

**Important**: Use `localhost:8000` or `127.0.0.1:8000` in your browser, NOT `0.0.0.0:8000`

## How to Use

1. **Ask Questions**: Type any question about your company's policies, products, shipping, returns, etc.

2. **Try Sample Questions**:
   - "What are your shipping options?"
   - "How do I return a product?"
   - "What payment methods do you accept?"
   - "Do you offer cash on delivery?"
   - "What is your refund policy?"

3. **The chatbot will**:
   - Greet you warmly
   - Search the vector database for relevant info
   - Use RAG to generate accurate answers
   - Respond in a natural, conversational way

## Architecture Explained

### 1. **Data Ingestion (ingest.py)**
- Reads TXT and PDF files
- Chunks text into 500-word segments with 50-word overlap
- Generates embeddings using `all-MiniLM-L6-v2` model
- Stores in ChromaDB vector database

### 2. **LangGraph Workflow (graph.py)**
```
User Question → Retrieve Context → Agent Decision → Generate Answer
                      ↓                    ↓
                 Vector DB          Use Tools? (optional)
```

**Nodes**:
- `retrieve_node`: Gets relevant context from vector DB
- `decide_node`: Agent decides how to proceed
- `tool_node`: Optional tool execution (can be extended)
- `answer_node`: Generates final conversational response

### 3. **Backend API (app.py)**
FastAPI endpoints:
- `GET /` - Serves the chatbot frontend
- `POST /api/chat` - Chat query endpoint
- `GET /api/health` - Health check endpoint
- `GET /docs` - Swagger UI API documentation
- `GET /redoc` - ReDoc API documentation

### 4. **Frontend (index.html)**
- Modern, responsive UI with gradient design
- Animated typing indicators
- Message history with avatars
- Quick suggestion chips
- Error handling with user feedback
- Smooth animations and transitions

## LLM Configuration

### Primary: Gemini 2.5 Flash (FREE)
- Model: `gemini-2.5-flash`
- Free tier: 15 requests/minute, 1 million tokens/day
- Fast, accurate, and cost-effective
- No installation needed

### Fallback: Ollama (OPTIONAL)
If Gemini fails, the system automatically falls back to Ollama:

1. **Install Ollama**: https://ollama.ai/download
2. **Pull a model**:
   ```bash
   ollama pull llama3
   ```
3. **Start Ollama**:
   ```bash
   ollama serve
   ```

The system will automatically use Ollama if Gemini is unavailable.

## API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints directly from the browser!

### Example API Usage

**Chat Request**:
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are your shipping options?"}'
```

**Response**:
```json
{
  "success": true,
  "question": "What are your shipping options?",
  "answer": "We offer several shipping options...",
  "context_preview": "Shipping Policy: We provide..."
}
```

## Technical Features

### FastAPI Benefits
- **Async Support**: High performance with async/await
- **Type Safety**: Pydantic models for request/response validation
- **Auto Documentation**: Swagger UI and ReDoc generated automatically
- **Fast**: One of the fastest Python frameworks
- **Modern**: Built on Python 3.6+ type hints

### RAG Implementation
- **Retrieval**: Semantic search using embeddings
- **Augmentation**: Injects relevant context into prompts
- **Generation**: LLM generates answers using retrieved context

### Vector Database (ChromaDB)
- Local, persistent storage
- Fast similarity search
- Metadata support for source tracking
- No external dependencies

### LangGraph Agentic Workflow
- **State Management**: TypedDict for type safety
- **Conditional Routing**: Agent decides next steps
- **Tool Integration**: Extensible for API calls, databases, etc.
- **Error Handling**: Graceful fallbacks

### Prompt Engineering
The chatbot uses carefully crafted prompts:
- **System context** with company knowledge
- **Role definition** as customer service agent
- **Output formatting** for conversational responses
- **Guardrails** to stay on topic

## Customization

### Add More Data
Simply add TXT or PDF files to `data/` folder and run:
```bash
python ingest.py
```

### Modify Prompts
Edit prompts in `graph.py`:
- `answer_prompt`: Controls response style
- `decision_prompt`: Controls agent behavior

### Add Tools
Extend `tool_node` in `graph.py` to:
- Check order status from database
- Call external APIs
- Retrieve real-time information
- Perform calculations

### Styling
Customize the UI in `index.html`:
- Colors, fonts, layout in `<style>` section
- Add company logo
- Change messaging and text

### CORS Configuration
Modify CORS settings in `app.py` if needed:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```


## Troubleshooting

### "Vector database not found"
**Solution**: Run `python ingest.py` first to create the knowledge base

### "index.html not found"
**Solution**: Ensure `index.html` is in the same directory as `app.py` (root level, not in templates folder)

### "Gemini error"
**Solution**: Check your API key in `.env` file is correct and active

### "Connection timeout" with Ollama
**Solution**: If using Ollama fallback, ensure it's running:
```bash
ollama serve
```

### Empty responses
**Solution**: Check that data files exist in `data/` folder and run `ingest.py`

### Port already in use
**Solution**: Change port in `app.py`:
```python
uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
```

### Browser shows "Can't connect"
**Solutions**:
- Use `http://localhost:8000` NOT `http://0.0.0.0:8000`
- Check if another service is using port 8000
- Check firewall settings
- Restart the server

## Deployment Options

### Local Development
```bash
python app.py
```

### Production with Uvicorn
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python ingest.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Google Cloud Run
```bash
gcloud run deploy chatbot \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS EC2/ECS
Use the same Docker container or install directly on the instance

## Next Steps for Production

1. **Add Authentication**: JWT tokens, OAuth, or API keys
2. **Database Integration**: Connect to PostgreSQL/MongoDB for user data
3. **Analytics**: Track user queries, response times, satisfaction
4. **Caching**: Use Redis for frequently asked questions
5. **Rate Limiting**: Prevent API abuse
6. **Monitoring**: Add logging with Sentry or CloudWatch
7. **Testing**: Unit tests with pytest, integration tests
8. **CI/CD**: GitHub Actions or GitLab CI
9. **Load Balancing**: Nginx or cloud load balancers
10. **Backup**: Automated backups of vector database

## Performance Optimization

- **Async Operations**: FastAPI handles concurrent requests efficiently
- **Vector Search**: ChromaDB uses HNSW for fast similarity search
- **Caching**: Cache frequent queries in Redis
- **Connection Pooling**: Reuse LLM connections
- **CDN**: Serve static files from CDN

## Support & Documentation

For questions or issues:
- Check code comments in each file
- FastAPI docs: https://fastapi.tiangolo.com
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- ChromaDB docs: https://docs.trychroma.com
- Gemini API docs: https://ai.google.dev/docs

---

