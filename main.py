# from sentence_transformers import SentenceTransformer
# import numpy as np
#
# model = SentenceTransformer('all-MiniLM-L6-v2')
#
# sentences = [
#     "The capital of France is Paris.",
#     "Paris is a city in Europe.",
#     "Python is a programming language.",
#     "Machine learning requires data.",
# ]
#
# embeddings = model.encode(sentences)
# print(f"Shape: {embeddings.shape}")
# print(f"Each sentence -> vector of {embeddings.shape[1]} numbers\n")
#
# def cosine_similarity(a,b):
#     return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# query = "What is the capital of France?"
# query_vec = model.encode([query])[0]
#
# print(f"Query: '{query}\n")
# for i, sentence in enumerate(sentences):
#     score = cosine_similarity(query_vec,embeddings[i])
#     print(f" {score:.3f} {sentence}")

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print(client.models.list())