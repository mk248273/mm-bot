import os
import faiss
import pickle
import numpy as np
from groq import Groq
from google import genai

# Initialize models and API clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai_client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))  # Use environment variable

def get_embedding(text):
    """Get embedding for text using Google GenAI."""
    if isinstance(text, list):
        # Handle list of texts
        embeddings = []
        for t in text:
            result = genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=t
            )
            embeddings.append(result.embeddings[0].values)
        return np.array(embeddings)
    else:
        # Handle single text
        result = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return np.array([result.embeddings[0].values])

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# PDF Functions
def save_pdf_embeddings(text, index_path, chunk_path):
    """Save PDF text embeddings to FAISS index and chunks to pickle file."""
    chunks = chunk_text(text)
    embeddings = get_embedding(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "PDF embeddings saved."

def load_pdf_embeddings(index_path, chunk_path):
    """Load PDF embeddings and chunks from saved files."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def answer_from_pdf(question, index_path, chunk_path, top_k=5):
    """Answer question using PDF embeddings."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    # Get question embedding
    q_embed = get_embedding(question)
    
    # Search for similar chunks
    _, I = index.search(q_embed, k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    # Create context and prompt
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Get response from Groq
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You're a helpful assistant for PDF content analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# Website Functions
def save_website_embeddings(text, index_path, chunk_path):
    """Save website text embeddings to FAISS index and chunks to pickle file."""
    chunks = chunk_text(text)
    embeddings = get_embedding(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "Website embeddings saved."

def load_website_embeddings(index_path, chunk_path):
    """Load website embeddings and chunks from saved files."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def answer_from_website(question, index_path, chunk_path, top_k=5):
    """Answer question using website embeddings."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    # Get question embedding
    q_embed = get_embedding(question)
    
    # Search for similar chunks
    _, I = index.search(q_embed, k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    # Create context and prompt
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Get response from Groq
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You're a helpful assistant for website content analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# YouTube Functions
def save_youtube_embeddings(text, index_path="youtube_index.faiss", chunk_path="youtube_chunks.pkl"):
    """Save YouTube transcript embeddings to FAISS index and chunks to pickle file."""
    chunks = chunk_text(text, chunk_size=500)  # Smaller chunks for YouTube
    embeddings = get_embedding(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "YouTube embeddings saved."

def load_youtube_embeddings(index_path="youtube_index.faiss", chunk_path="youtube_chunks.pkl"):
    """Load YouTube embeddings and chunks from saved files."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def answer_from_youtube(question, index_path="youtube_index.faiss", chunk_path="youtube_chunks.pkl", top_k=5):
    """Answer question using YouTube transcript embeddings."""
    index, chunks = load_youtube_embeddings(index_path, chunk_path)

    # Get question embedding
    q_embed = get_embedding(question)
    
    # Search for similar chunks
    _, I = index.search(q_embed, k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    # Create context and prompt
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Get response from Groq
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions based on YouTube transcript."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# Text Functions
def save_text_embeddings(text, index_path, chunk_path):
    """Save plain text embeddings to FAISS index and chunks to pickle file."""
    chunks = chunk_text(text)
    embeddings = get_embedding(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "Text embeddings saved."

def load_text_embeddings(index_path, chunk_path):
    """Load text embeddings and chunks from saved files."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def answer_from_text(question, index_path, chunk_path, top_k=5):
    """Answer question using text embeddings."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    # Get question embedding
    q_embed = get_embedding(question)
    
    # Search for similar chunks
    _, I = index.search(q_embed, k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    # Create context and prompt
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Get response from Groq
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You're a helpful assistant for analyzing plain text."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# General utility functions
def save_embeddings(text, index_path, chunk_path):
    """General function to save embeddings for any text."""
    chunks = chunk_text(text)
    embeddings = get_embedding(chunks)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "Embeddings saved."

def load_embeddings(index_path, chunk_path):
    """General function to load embeddings and chunks."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def answer_question(index_path, chunk_path, question, top_k=5):
    """Answer question using vector search (returns raw chunks)."""
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    # Get question embedding
    q_embedding = get_embedding(question)
    
    # Search for similar chunks
    _, I = index.search(q_embedding, top_k)
    
    # Return top chunks separated by dividers
    return "\n---\n".join([chunks[i] for i in I[0]])