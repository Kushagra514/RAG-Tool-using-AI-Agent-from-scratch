import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_store(path = "store"):
    index = faiss.read_index(f"{path}/index.faiss")
    with open(f"{path}/chunks.pkl","rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query, index, chunks,model,k=3,threshold = 0.8):
    query_vec = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec,k)

    results = []
    for dist, idx in zip(distances[0],indices[0]):
        confident = dist < threshold
        results.append({
            "chunk": chunks[idx],
            "distance": round(float(dist),3),
            "confident": confident
        })
    return results

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, chunks = load_store()

    query = "Which college does Kushagra study at?"
    results = retrieve(query,index,chunks,model)

    for i,r in enumerate(results):
        print(f"\n--- chunk {i+1} | distance: {r['distance']} | confident: {r['confident']} ---")
        print(r['chunk'])