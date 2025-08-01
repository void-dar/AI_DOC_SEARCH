# Document Intelligence API

A FastAPI-based backend that ingests documents, stores vector embeddings in Qdrant, and uses OpenAI LLMs to answer questions based on user-uploaded content.

## Features
- Upload & parse HTML, PDF, or plain text
- Embeds document chunks using OpenAI
- Stores in Qdrant (per-user collections)
- Answers questions via LangChain + OpenAI
- Source snippets returned with every answer

## Tech Stack
- FastAPI
- LangChain
- Qdrant
- OpenAI
- BeautifulSoup / PyMuPDF

## Running the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
Docker:
docker build -t doc-api .
docker run -p 8000:8000 doc-api
Run qdrant application:
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
Run FastAPI:
uvicorn app.main:app --reload / fastapi run app/

