
# 🚀 RAG + Tool-Using AI Agent

An end-to-end **agentic AI system** that combines Retrieval-Augmented Generation (RAG), tool use, and memory to answer questions from documents and external sources with grounded, reliable responses.

---

## 🧠 Problem

LLMs are powerful but fundamentally limited:

- ❌ Hallucinate when lacking context
- ❌ Cannot access private documents
- ❌ Lack real-time information
- ❌ Forget past interactions

This project addresses these gaps by building a **stateful AI agent** that follows:

```
Retrieve → Reason → Act → Observe → Respond
```

---

## ⚡ Solution

A modular system integrating:

- 📄 **RAG** → grounded answers from documents
- 🛠️ **Tool use** → web search, calculator, document retrieval
- 🧠 **Memory** → short-term + long-term persistence
- 🤖 **Agent loop** → decision-making using function calling

---

## 🏗️ Architecture

```
User Query
↓
LLM (decision step)
↓
Select Tool (if needed)
├── doc_search (FAISS)
├── web_search (DuckDuckGo)
└── calculator
↓
Tool Execution
↓
LLM (final answer generation)
↓
Response + Memory Update
```

---

## 🔑 Core Components

### 📄 Document Ingestion

**PDF → Text Extraction → Chunking → Embeddings → FAISS Index**

- PyMuPDF for text extraction
- Chunking with overlap
- SentenceTransformers for embeddings (384-dim)
- FAISS (IndexFlatL2) for vector storage

---

### 🔍 Retrieval Engine

**Query → Embedding → Similarity Search → Top-k Chunks**

- Semantic similarity search
- Maps FAISS indices → original text
- Provides context to LLM

---

### 🛠️ Tool Layer

- 🧮 **Calculator** → evaluates math expressions
- 🌐 **Web Search** → DuckDuckGo for real-time info
- 📄 **Document Search** → FAISS-based retrieval

---

### 🤖 Agent Loop

Two-step reasoning:

1. **Decision**  
   *Do I need a tool? Which one?*

2. **Execution + Response**  
   - Execute tool
   - Feed result to LLM
   - Generate final answer

---

### 🧠 Memory System

#### Short-Term Memory (Conversation Buffer)

```python
if len(conversation) > 20:
    conversation[1:] = conversation[-18:]
```

- Keeps recent context
- Prevents token overflow

#### Long-Term Memory (Semantic Memory)

```python
if len(answer) > 100:
    save_to_memory(f"Q: {user_input}\nA: {answer}")
```

- Stores Q&A as embeddings
- Persisted in FAISS
- Enables cross-session recall

---

## 🧪 Example Queries

| Query | Behavior |
|-------|----------|
| What is the document about? | Uses `doc_search` |
| What is 25 * 67? | Uses `calculator` |
| Who is the CEO of Google? | Uses `web_search` |

---

## 📂 Project Structure

```
.
├── ingest.py
├── retrieve.py
├── ask.py
├── tools.py
├── agent.py
├── store/
│   ├── index.faiss
│   └── chunks.pkl
└── document.pdf
```

---

## 🚀 Setup & Run

### Install Dependencies

```bash
pip install faiss-cpu sentence-transformers pymupdf duckduckgo-search python-dotenv groq
```

### Create `.env`

```
GROQ_API_KEY=your_api_key
```

### Run

```bash
python ingest.py
python agent.py
```

---

## 📈 Key Design Decisions

- **FAISS** → fast local vector search
- **Chunking** → better retrieval granularity
- **Tool use** → extends LLM capability
- **Two-step agent loop** → reliable reasoning

---

## ⚠️ Limitations

- Naive memory filtering (length-based)
- Uses L2 instead of cosine similarity
- No reranking layer
- CLI-only interface

---

## 🚀 Future Improvements

- Cosine similarity + normalization
- LLM-based memory filtering
- Reranking (cross-encoder)
- Web UI (Streamlit/React)
- Deployment (FastAPI + Docker)

---

## 🎯 Summary

**RAG + Tool Use + Memory + Agent Reasoning**

A complete agentic AI pipeline, similar to systems used in:

- ChatGPT (tool use, memory)
- Perplexity (retrieval + web search)
- Notion AI (document grounding)

---

## 👤 Author

**Kushagra Tewari**

- GitHub: [https://github.com/Kushagra514](https://github.com/Kushagra514)
- LinkedIn: [https://www.linkedin.com/in/kushagra-tewari-536a92332](https://www.linkedin.com/in/kushagra-tewari-536a92332)

---

## ⭐ Final Note

This project is not just a chatbot—it is a modular AI agent system capable of reasoning, acting, and learning over time.
