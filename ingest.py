import faiss
import numpy as np
import pymupdf
import pickle
import os
from sentence_transformers import SentenceTransformer

def load_pdf(path):
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text,chunk_size = 300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(chunks,model):
    embeddings = model.encode(chunks).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index,embeddings

def save(index,chunks,path = "store"):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index,f"{path}/index.faiss")
    with open(f"{path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {path}/")

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text = load_pdf("document.pdf")
    chunks = chunk_text(text,chunk_size=300,overlap=50)
    print(f"Total chunks:{len(chunks)}")
    index, _ = build_index(chunks,model)
    save(index,chunks)
