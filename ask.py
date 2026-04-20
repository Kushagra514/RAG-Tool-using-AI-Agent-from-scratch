import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from retrieve import load_store,retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')
index, chunks = load_store()

def ask(query):
    results = retrieve(query,index,chunks,model,k=3)

    low_confidence = all(not r["confident"] for r in results)
    if low_confidence:
        print("[warning] retrived chunks have low confidence - answers maybe unreliable\n")

    context = "\n\n".join([r["chunk"] for r in results])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document assistant. "
                "Answer only using the context below. "
                "If the answer is not in the context, say 'I dont have enough info'. "
                "Do not make up facts.\n\n"
                f"CONTEXT:\n{context}"
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]
    response = client.chat.completions.create(
        model = "mixtral-8x7b-32768",
        messages = messages,
    )
    answer = response.choices[0].message.content
    return answer,results

if __name__ == "__main__":
    while True:
        query = input("\n Ask a question (or 'quit'):").strip()
        if query.lower() == 'quit':
            break
        answer,results = ask(query)
        print(f"\nAnswer: {answer}")
        print(f"\nSources used: {len(results)} chunks,distances: {[r['distance'] for r in results]}")