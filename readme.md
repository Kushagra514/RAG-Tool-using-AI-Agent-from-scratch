🚀 RAG + Tool-Using AI Agent

An end-to-end agentic AI system that combines Retrieval-Augmented Generation (RAG), tool use, and memory to answer questions from documents and external sources with grounded, reliable responses.

🧠 Problem

LLMs are powerful but fundamentally limited:

❌ Hallucinate when lacking context
❌ Cannot access private documents
❌ Lack real-time information
❌ Forget past interactions

This project addresses these gaps by building a stateful AI agent that can:

Retrieve → Reason → Act → Observe → Respond
⚡ Solution

A modular system that integrates:

📄 RAG → grounded answers from documents
🛠️ Tool use → external capabilities (web, math, DB)
🧠 Memory → short-term + long-term persistence
🤖 Agent loop → decision-making with function calling
🏗️ Architecture
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
🔑 Core Components
1. Document Ingestion Pipeline
PDF → Text Extraction → Chunking → Embeddings → FAISS Index
Extract text using PyMuPDF
Chunk with overlap for context preservation
Encode using SentenceTransformers (384-dim embeddings)
Store vectors in FAISS (IndexFlatL2)
2. Retrieval Engine
Query → Embedding → Similarity Search → Top-k Chunks
Semantic search over vector space
Maps FAISS indices → original text chunks
Provides context for LLM grounding
3. Tool Layer

The agent is equipped with 3 tools:

🧮 Calculator
Evaluates arithmetic expressions safely
🌐 Web Search
Uses DuckDuckGo for real-time information
📄 Document Search
Queries internal knowledge base (FAISS)
4. Agent (Core Intelligence)

Implements a two-step reasoning loop:

Step 1 — Decision

LLM determines:

Do I need a tool? Which one?
Step 2 — Execution + Response
Tool is executed in Python
Result fed back to LLM
Final grounded answer generated
5. Memory System
🟡 Short-Term Memory (Conversation Buffer)
if len(conversation) > 20:
    conversation[1:] = conversation[-18:]
Maintains recent context
Prevents token overflow
🔵 Long-Term Memory (Semantic Memory)
if len(answer) > 100:
    save_to_memory(f"Q: {user_input}\nA: {answer}")
Stores Q&A pairs as embeddings
Persisted in FAISS + disk
Enables cross-session recall
🧪 Example Interactions
Query	Behavior
What is the document about?	Uses doc_search
What is 25 * 67?	Uses calculator
Who is the CEO of Google?	Uses web_search
📂 Project Structure
.
├── ingest.py        # Builds FAISS index from documents
├── retrieve.py      # Semantic retrieval logic
├── ask.py           # Basic RAG pipeline
├── tools.py         # Tool definitions
├── agent.py         # Agent loop (decision + execution)
├── store/
│   ├── index.faiss
│   └── chunks.pkl
└── document.pdf
🚀 Setup & Run
1. Install Dependencies
pip install faiss-cpu sentence-transformers pymupdf duckduckgo-search python-dotenv groq
2. Configure Environment
GROQ_API_KEY=your_api_key
3. Ingest Document
python ingest.py
4. Run Agent
python agent.py
📈 Key Design Decisions
Why FAISS?
Fast approximate nearest neighbor search
Efficient for local vector storage
Why Chunking?
Improves retrieval granularity
Avoids context overflow
Why Tool Use?
Extends LLM beyond static knowledge
Enables real-world applicability
Why Two-Step Agent Loop?
Separates reasoning from execution
Improves reliability and interpretability
⚠️ Limitations
Memory filtering is naive (length-based)
Uses L2 distance instead of cosine similarity
No ranking/reranking layer
CLI-only interface
🚀 Future Improvements
🔥 Cosine similarity + normalization
🧠 LLM-based memory filtering
📊 Reranking (cross-encoder)
🌐 Web UI (Streamlit / React)
⚙️ Deployment (FastAPI + Docker)
🎯 What This Demonstrates
RAG + Tool Use + Memory + Agent Reasoning

A complete agentic AI pipeline, similar to systems used in:

ChatGPT (tool use, memory)
Perplexity (retrieval + web search)
Notion AI (document grounding)
👤 Author

Kushagra Tewari

GitHub: https://github.com/Kushagra514
LinkedIn: https://www.linkedin.com/in/kushagra-tewari-536a92332
⭐ Final Note

This project is not just a chatbot—it is a modular AI agent system capable of reasoning, acting, and learning over time.