"""
Data Ingestion Script
Processes TXT and PDF files from the data/ folder and creates vector embeddings
Uses ChromaDB for vector storage with sentence-transformers embeddings
"""

import os
import glob
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader

# Paths
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"

# Initialize embedding model 
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create or get collection
collection = client.get_or_create_collection(
    name="company_docs",
    metadata={"description": "Company knowledge base"}
)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {txt_path}: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def ingest_documents():
    """Process all documents in data folder"""
    
    # Get all files
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    all_files = txt_files + pdf_files
    
    if not all_files:
        print("No TXT or PDF files found in data/ folder!")
        return
    
    print(f"Found {len(txt_files)} TXT files and {len(pdf_files)} PDF files")
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    chunk_id = 0
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract text based on file type
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_txt(file_path)
        
        if not text.strip():
            print(f"Skipping empty file: {filename}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")
        
        # Prepare for vector DB
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "source": filename,
                "chunk_id": chunk_id
            })
            all_ids.append(f"doc_{chunk_id}")
            chunk_id += 1
    
    if not all_chunks:
        print("No content to ingest!")
        return
    
    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    print("Generating embeddings...")
    
    # Generate embeddings
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    
    # Clear existing collection
    try:
        client.delete_collection("company_docs")
        collection = client.create_collection(name="company_docs")
    except:
        pass
    
    # Add to ChromaDB
    print("Adding to vector database...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadata,
        ids=all_ids
    )
    
    print(f"✓ Successfully ingested {len(all_chunks)} chunks from {len(all_files)} files")
    print(f"✓ Vector database saved to {CHROMA_DIR}")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    ingest_documents()