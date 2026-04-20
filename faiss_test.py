import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "The capital of France is Paris.",
    "Paris is a city in Europe.",
    "Python is a programming language",
    "Machine learning requires data",
    "The eifel tower is in paris.",
]

embeddings = model.encode(docs).astype('float32')
dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"Index contains {index.ntotal} vectors\n")

query = "Tell me about Paris"
query_vec = model.encode([query]).astype('float32')

k=3
distances,indices = index.search(query_vec,k)


print(f"Top {k} results for: '{query}'\n")
for rank, (dist,idx) in enumerate(zip(distances[0],indices[0])):
    print(f" {rank+1}.[dist={dist:.3f}] {docs[idx]}")