# Chatbot API Service

# Overview
This project is a chatbot API service designed to handle and respond to user queries. It processes incoming messages and returns appropriate responses through a simple REST API.

The chatbot is being developed as part of a larger system and is intended to be integrated with a frontend application for real-time conversational interaction.

---

# Features
- Simple REST API for chatbot communication
- Accepts user messages via POST requests
- Returns structured JSON responses
- Loads configured backend documents and chunks them for retrieval
- Stores embeddings in a persistent ChromaDB vector database
- Uses a local deterministic embedding function for vector search, so startup does not need an external embedding-model download
- Uses Ollama generation with `deepseek-r1:1.5b`
- Easily extendable for AI integration or advanced NLP features
- Lightweight and easy to deploy

---

 # How It Works
1. User sends a message from the frontend
2. Message is sent to the chatbot API endpoint
3. Documents listed in `DOCUMENT_PATHS` inside `app.py` are chunked, embedded, and stored in ChromaDB
4. User questions are embedded and compared with stored document chunks
5. The retrieved context is passed to Ollama in `rag_service.py`
6. API returns a response in JSON format

---

# Tech Stack
- Python
- Flask
- Flask-CORS
- ChromaDB
- Local embeddings stored in ChromaDB

---

# RAG Endpoints
- `GET /documents/status`: returns the number of stored chunks
- `POST /chat`: retrieves matching chunks and passes them to the LLM placeholder

The current Ollama model is configured in `RAGService` as `deepseek-r1:1.5b`. Make sure Ollama is running locally and the model is pulled before asking document questions.

To add a document from code, put its path in `DOCUMENT_PATHS` in `app.py`:

```python
DOCUMENT_PATHS = [
    "documents/Admission Procedure_undergraduate 2024.pdf",
]
```

Project documents live in the `documents/` folder.
