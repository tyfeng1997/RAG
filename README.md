# TiDB RAG Agent Backend

This project is a Retrieval-Augmented Generation (RAG) backend designed for agent applications. It leverages TiDB as a vector database and integrates Mistral OCR for PDF document processing. The system is modular, supporting document chunking, vector search, reranking, and generative models. Future enhancements will include full-text search and advanced retrieval strategies to improve accuracy.

## Project Structure

- **core/**
  - `core.py`: Core data structures (Chunk, Document, etc.) and abstract interfaces (EmbeddingModel, VectorStore, etc.).
  - `rag.py`: Main RAG pipeline, including chunking, retrieval, reranking, and answer generation.
- **embedding/**
  - `cohere_embedding.py`: Cohere embedding model implementation.
- **generate/**
  - `anthropic_genearte.py`: Anthropic Claude generative model implementation.
- **rerank/**
  - `cohere_rerank.py`: Cohere rerank model implementation.
- **vector/**
  - `TiDB_vector.py`: TiDB vector store and retrieval implementation.
- `utils.py`: Utility functions, including OCR result conversion to pipeline data.
- `api.py`: FastAPI entry point, exposing main APIs.

## Chunk Concept

A Chunk is the smallest unit of document processing. Each chunk contains content, document ID, type (e.g., text), original index, metadata, and embedding vector. All retrieval, reranking, and generation operations are based on chunks.

## How to Run

```sh
uvicorn api:app --reload
```

## API Endpoints

### 1. Query Endpoint

```sh
curl -X POST "http://127.0.0.1:8000/query" \
	-H "Content-Type: application/json" \
	-d '{
		"text": "What is Crypto Payment Gateway?",
		"use_reranking": true
	}'
```

### 2. PDF Ingestion Endpoint

```sh
curl -X POST "http://127.0.0.1:8000/ingest_pdf" \
	-H "Content-Type: application/json" \
	-d '{
		"base64_pdf": "JVBERi0xLjUKJcTl8uXrp...",
		"doc_id": "doc_fake",
		"chunk_size": 512
	}'
```

## Future Plans

- Integrate full-text search
- Support advanced retrieval strategies (e.g., hybrid, multimodal)
- Improve retrieval accuracy and scalability

---

For more details, please refer to the source code or open an issue.
