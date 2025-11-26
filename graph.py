"""
LangGraph Agentic Workflow
Implements RAG with tool-calling agent pattern using Gemini (primary) and Ollama (fallback)
"""

import os
import json
import requests
from typing import TypedDict, Literal
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
CHROMA_DIR = "./chroma_db"

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    USE_GEMINI = True
    print("✓ Using Gemini AI (Primary)")
else:
    USE_GEMINI = False
    print("⚠ Gemini API key not found, using Ollama fallback")

# Initialize embedding model and ChromaDB
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    collection = chroma_client.get_collection("company_docs")
    print(f"✓ Loaded vector database with {collection.count()} documents")
except:
    print("⚠ Vector database not found. Run ingest.py first!")
    collection = None

# State Definition
class AgentState(TypedDict):
    question: str
    context: str
    llm_decision: str
    tool_output: str
    answer: str

# RAG Retrieval Tool
def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant context from vector database"""
    if not collection:
        return "No knowledge base available. Please run ingest.py first."
    
    try:
        query_embedding = embedding_model.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        if results['documents']:
            context = "\n\n".join(results['documents'][0])
            return context
        return "No relevant information found."
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

# LLM Functions
def call_gemini(prompt: str, temperature: float = 0.7) -> str:
    """Call Gemini API"""
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
            )
        )
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

def call_ollama(prompt: str, temperature: float = 0.7) -> str:
    """Call Ollama API (fallback)"""
    try:
        # Fixed: Increased timeout and added connection timeout
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1024  
                }
            },
            timeout=(10, 180)  
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Ollama returned status code: {response.status_code}")
        return None
    except requests.exceptions.Timeout:
        print("Ollama error: Request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("Ollama error: Could not connect to Ollama server. Is it running?")
        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """Call LLM with Gemini primary and Ollama fallback"""
    if USE_GEMINI:
        result = call_gemini(prompt, temperature)
        if result:
            return result
        print("Gemini failed, trying Ollama fallback...")
    
    result = call_ollama(prompt, temperature)
    if result:
        return result
    
    return "I apologize, but I'm having trouble connecting to the AI service. Please try again."

# Graph Nodes
def retrieve_node(state: AgentState) -> AgentState:
    """Node 1: Retrieve relevant context from vector DB"""
    print(f"\n[RETRIEVE] Searching knowledge base for: {state['question']}")
    context = retrieve_context(state['question'])
    state['context'] = context
    print(f"[RETRIEVE] Found {len(context)} characters of context")
    return state

def decide_node(state: AgentState) -> AgentState:
    """Node 2: Agent decides whether to use tools or answer directly"""
    
    decision_prompt = f"""You are a helpful customer service agent for our company.

Context from knowledge base:
{state['context']}

Customer question: {state['question']}

Based on the context provided, can you answer this question directly?

Respond with ONLY a JSON object in this exact format:
{{"action": "ANSWER", "reasoning": "brief reason"}}

If you can answer based on the context, use action "ANSWER".
Make sure your response is valid JSON only, no extra text."""

    print("[DECIDE] Agent analyzing question...")
    decision = call_llm(decision_prompt, temperature=0.3)
    state['llm_decision'] = decision
    
    try:
        decision_json = json.loads(decision)
        action = decision_json.get("action", "ANSWER")
        print(f"[DECIDE] Decision: {action}")
    except:
        print("[DECIDE] Could not parse decision, defaulting to ANSWER")
    
    return state

def tool_node(state: AgentState) -> AgentState:
    """Node 3: Execute tools (currently just retrieval)"""
    print("[TOOL] Executing tool...")
    state['tool_output'] = "Tool execution completed"
    return state

def answer_node(state: AgentState) -> AgentState:
    """Node 4: Generate final answer"""
    
    answer_prompt = f"""You are a friendly and professional customer service representative for our company.

Knowledge Base Information:
{state['context']}

Customer Question: {state['question']}

Instructions:
1. Greet the customer warmly if this is the first interaction
2. Answer their question accurately using ONLY the information from the knowledge base above
3. If the information isn't in the knowledge base, politely say you don't have that specific information and offer to help with something else
4. Be conversational, helpful, and professional
5. Keep your answer concise but complete
6. End with an offer to help with anything else

Provide your response as a natural conversation, NOT as JSON."""

    print("[ANSWER] Generating response...")
    answer = call_llm(answer_prompt, temperature=0.7)
    state['answer'] = answer
    print(f"[ANSWER] Response generated ({len(answer)} characters)")
    return state

# Conditional Edge
def route_decision(state: AgentState) -> Literal["answer", "tool"]:
    """Route based on agent decision"""
    try:
        decision = json.loads(state['llm_decision'])
        action = decision.get("action", "ANSWER")
        if action == "TOOL":
            return "tool"
    except:
        pass
    return "answer"

# Build Graph
def create_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide", decide_node)
    graph.add_node("tool", tool_node)
    graph.add_node("final_answer", answer_node)  # renamed

    # Entry
    graph.set_entry_point("retrieve")
    
    # Edges
    graph.add_edge("retrieve", "decide")

    # Conditional routing
    graph.add_conditional_edges(
        "decide",
        route_decision,
        {
            "tool": "tool",
            "answer": "final_answer"   # renamed
        }
    )

    graph.add_edge("tool", "final_answer")  # renamed
    graph.add_edge("final_answer", END)     # renamed

    return graph.compile()


# Create the graph
workflow = create_graph()

def query_agent(question: str) -> dict:
    """Run a query through the agent workflow"""
    
    initial_state = AgentState(
        question=question,
        context="",
        llm_decision="",
        tool_output="",
        answer=""
    )
    
    print(f"\n{'='*50}")
    print(f"PROCESSING QUERY: {question}")
    print(f"{'='*50}")
    
    try:
        result = workflow.invoke(initial_state)
        
        print(f"\n{'='*50}")
        print("WORKFLOW COMPLETED")
        print(f"{'='*50}\n")
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context_used": result["context"][:500] + "..." if len(result["context"]) > 500 else result["context"]
        }
    except Exception as e:
        print(f"Error in workflow: {e}")
        return {
            "question": question,
            "answer": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
            "context_used": ""
        }

if __name__ == "__main__":
    # Test the agent
    test_questions = [
        "What are your shipping options?",
        "How do I return a product?",
        "What payment methods do you accept?"
    ]
    
    for q in test_questions:
        result = query_agent(q)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}\n")
        print("-" * 80)
