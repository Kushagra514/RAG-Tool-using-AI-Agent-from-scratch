import pickle
import faiss
from duckduckgo_search import DDGS
from retrieve import load_store, retrieve
from sentence_transformers import SentenceTransformer

index, chunks = load_store()
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculator(expression: str) -> str:
    try:
        result = eval(expression,{"__builtins__":{}})
        return str(result)
    except:
        return "Invalid expression"

def web_search(query: str) -> str:
    with DDGS() as ddgs:
        result = list(ddgs.text(query,max_results=3))

def doc_search(query: str) -> str:
    results = retrieve(query,index,chunks,model,k=3)
    return "\n\n".join(r['chunk'] for r in results)

def save_to_memory(text: str):
    vec = model.encode([text]).astype('float32')
    index.add(vec)
    chunks.append(text)
    faiss.write_index(index, "store/index.faiss")
    with open("store/chunks.pkl","wb") as f:
        pickle.dump(chunks,f)
    return "Saved to memory"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "evaluate a math expression",
            "parameters" : {
                "type": "object",
                "properties": {
                    "expression": {"type":"string","description" : "e.g. 2+2 or 15*12"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query" : {"type":"string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "doc_search",
            "description": "Search the uploaded document for relevant information",
            "parameter": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]