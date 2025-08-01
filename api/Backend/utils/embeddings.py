import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Initialize model and API client
model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Chunk text form PDF
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

#  Save embeddings from pdf
def save_pdf_embeddings(text, index_path, chunk_path):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "Embeddings saved."
#load Pdf saved embedding
def load_pdf_embeddings(index_path, chunk_path):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

#  Answer from LLM
def answer_from_pdf(question, index_path, chunk_path, top_k=5):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    q_embed = model.encode([question])
    _, I = index.search(np.array(q_embed), k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You're a helpful assistant for PDF content analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

#Save embedding from website 
def save_website_embeddings(text, index_path, chunk_path):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "Website embeddings saved."

#Load website saved embedding 
def load_website_embeddings(index_path, chunk_path):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

#  Answer from LLM
def answer_from_website(question, index_path, chunk_path, top_k=5):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    q_embed = model.encode([question])
    _, I = index.search(np.array(q_embed), k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You're a helpful assistant for website content analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# Answer question using vector search
def answer_question(index_path, chunk_path, question, top_k=5):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    q_embedding = model.encode([question])
    _, I = index.search(np.array(q_embedding), top_k)
    return "\n---\n".join([chunks[i] for i in I[0]])


def save_embeddings(text, index_path, chunk_path):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "Embeddings saved."


def load_embeddings(index_path, chunk_path):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# Load youtube embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=500):
    """Split text into chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
# Save youtube embeddings 
def save_youtube_embeddings(text, index_path="youtube_index.faiss", chunk_path="youtube_chunks.pkl"):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "YouTube embeddings saved."

def load_youtube_embeddings(index_path="youtube_index.faiss", chunk_path="youtube_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

#Load YouTube saved embeddings
def load_youtube_embeddings(index_path, chunk_path):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Answer from YouTube Embeddings
def answer_from_youtube(question, index_path="youtube_index.faiss", chunk_path="youtube_chunks.pkl", top_k=5):
    index, chunks = load_youtube_embeddings(index_path, chunk_path)

    # Embed question
    q_embed = model.encode([question])
    _, I = index.search(np.array(q_embed), k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    # Build prompt
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions based on YouTube transcript."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

#Save Text embeddings 
def save_text_embeddings(text, index_path, chunk_path):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return "Text embeddings saved."

#Load Saved Text Embedding
def load_text_embeddings(index_path, chunk_path):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Answer From Text Embeddings
def answer_from_text(question, index_path, chunk_path, top_k=5):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    q_embed = model.encode([question])
    _, I = index.search(np.array(q_embed), k=top_k)
    top_chunks = [chunks[i] for i in I[0]]

    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You're a helpful assistant for analyzing plain text."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

