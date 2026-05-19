# Document Intelligence API

A RAG (Retrieval-Augmented Generation) backend that ingests documents,
stores vector embeddings, and answers natural language queries using LLMs
— with source snippets returned alongside every answer.

## The Problem

Businesses and researchers drowning in documents (PDFs, reports, contracts)
need to query their own content without reading everything manually.
Standard search returns keywords. This returns answers, with citations.

## What I Built

- **Document ingestion** — accepts HTML, PDF, and plain text
- **Chunked embedding** — splits documents into semantic chunks,
  embeds via OpenAI, stores per-user in Qdrant vector store
- **LLM-powered Q&A** — LangChain orchestrates retrieval + generation
- **Source attribution** — every answer includes the source snippets
  it was generated from
- **Per-user isolation** — each user gets their own Qdrant collection,
  ensuring data separation

## Architecture

Upload (PDF/HTML/text)
→ Parse (BeautifulSoup / PyMuPDF)
→ Chunk + Embed (OpenAI Embeddings)
→ Store (Qdrant — per-user collection)
Query
→ Embed query
→ Similarity search (Qdrant)
→ LLM generation with retrieved context (LangChain + OpenAI)
→ Response + source snippets

## Key Technical Decisions

**Why Qdrant over pgvector:** Per-user collection isolation is cleaner
in Qdrant than partitioned pgvector tables. Also allows independent
scaling of the vector store from the main database.

**Why LangChain:** Handles the retrieval-generation pipeline cleanly
including context window management for large document sets.

## Stack

- Python · FastAPI · LangChain · Qdrant · OpenAI API
- BeautifulSoup · PyMuPDF · Docker · Docker Compose

## Running Locally

```bash
# Start Qdrant
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Start API
docker build -t doc-api .
docker run -p 8000:8000 doc-api

# Or with compose
docker-compose up
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload and ingest a document |
| POST | `/query` | Ask a question against your documents |
| GET | `/documents` | List ingested documents |
| DELETE | `/documents/{id}` | Remove a document |
