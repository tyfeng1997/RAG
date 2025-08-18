# TiDB RAG System

A modular Retrieval-Augmented Generation (RAG) system built with TiDB as the vector and full-text search backend. This project provides a flexible architecture for building AI applications that combine vector similarity search, full-text search, and advanced reranking capabilities.

## 🚀 Features

- **Hybrid Search**: Combines vector similarity search and full-text search for comprehensive document retrieval
- **Multi-Provider Support**: Integrates with Anthropic Claude, Cohere, and OpenAI APIs
- **TiDB Integration**: Uses TiDB for both vector storage and full-text search capabilities
- **Modular Architecture**: Easy to extend with new embedding models, rerankers, and generative models
- **Flexible Configuration**: Customizable search strategies and parameters
- **Production Ready**: Includes logging, error handling, and batch processing capabilities

## 🏗️ Architecture

The system follows a modular, plugin-based architecture with clear abstractions:

```
├── core/
│   ├── core.py          # Abstract base classes and data models
│   └── rag.py           # Main RAG system orchestrator
├── embedding/
│   └── cohere_embedding.py    # Cohere embedding implementation
├── fulltext/
│   └── TiDB_fulltext.py       # TiDB full-text search implementation
├── generate/
│   └── anthropic_genearte.py  # Anthropic Claude generation implementation
├── rerank/
│   └── cohere_rerank.py       # Cohere reranking implementation
├── vector/
│   └── TiDB_vector.py         # TiDB vector store implementation
├── logger.py            # Logging utilities
└── example.py          # Usage example and factory function
```

### Core Abstractions

The system is built around these key abstract base classes:

- **`EmbeddingModel`**: Interface for text embedding models
- **`RerankModel`**: Interface for document reranking models
- **`GenerativeModel`**: Interface for text generation models
- **`VectorStore`**: Interface for vector similarity search
- **`TextSearchStore`**: Interface for full-text search capabilities

This design makes it easy to swap out different providers or add new implementations without changing the core RAG logic.

## 📦 Installation

### Prerequisites

- Python 3.11+
- uv (for environment management)
- TiDB database instance
- API keys for your chosen providers (Anthropic, Cohere, etc.)

### Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd tidb_rag
   ```

2. **Create and activate virtual environment**:

   ```bash
   uv venv rag
   source rag/bin/activate  # On Windows: rag\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   uv pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env.local` file in the project root:

   ```env
   # API Keys
   ANTHROPIC_API_KEY=your_anthropic_api_key
   COHERE_API_KEY=your_cohere_api_key

   # TiDB Configuration
   TIDB_DATABASE_URL=mysql+pymysql://username:password@host:port/database

   ```

## 🔧 Usage

### Basic Example

```python
from dotenv import load_dotenv
from example import create_rag_system
from core.core import SearchStrategy

load_dotenv()

# Create RAG system with default configuration
rag_system = create_rag_system({
    "vector_search_top_n": 10,
    "text_search_top_n": 10,
    "rerank_top_n": 5,
    "final_context_chunks": 3
})

# Ingest documents (pipeline JSON format)
pipeline_data = [
    {
        "doc_id": "doc_1",
        "original_uuid": "uuid_here",
        "content": "Document content...",
        "chunks": [
            {
                "type": "text",
                "chunk_id": "chunk_1",
                "original_index": 0,
                "content": "Chunk content..."
            }
        ]
    }
]

# Ingest documents
success = rag_system.ingest_from_pipeline_json(pipeline_data)
print(f"Ingestion successful: {success}")

# Query the system
result = rag_system.query(
    "What is machine learning?",
    strategy=SearchStrategy.HYBRID,
    use_reranking=True
)

print(f"Answer: {result['answer']}")
print(f"Found {len(result['search_results'])} search results")
```

### Advanced Usage

#### Custom Configuration

```python
from core.core import RAGConfig
from core.rag import RAGSystem
from embedding.cohere_embedding import CohereEmbeddingModel
# ... other imports

# Create custom configuration
config = RAGConfig(
    vector_search_top_n=15,
    text_search_top_n=15,
    rerank_top_n=8,
    final_context_chunks=4,
    max_tokens=1500
)

# Initialize components manually for more control
embedding_model = CohereEmbeddingModel(model_name="embed-v4.0")
# ... initialize other components

# Create RAG system with custom setup
rag_system = RAGSystem(
    embedding_model=embedding_model,
    rerank_model=rerank_model,
    generative_model=generative_model,
    vector_store=vector_store,
    text_search_store=text_search_store,
    config=config
)
```

#### Search Strategies

```python
from core.core import SearchStrategy

# Vector-only search
vector_result = rag_system.query(
    "query text",
    strategy=SearchStrategy.VECTOR_ONLY
)

# Full-text only search
text_result = rag_system.query(
    "query text",
    strategy=SearchStrategy.TEXT_ONLY
)

# Hybrid search (recommended)
hybrid_result = rag_system.query(
    "query text",
    strategy=SearchStrategy.HYBRID
)
```

## 🔌 Adding New Components

The modular architecture makes it easy to integrate new providers or capabilities:

### Adding a New Embedding Model

```python
from core.core import EmbeddingModel
from typing import List

class CustomEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, api_key: str):
        # Initialize your model
        pass

    def embed_text(self, text: str) -> List[float]:
        # Implement single text embedding
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Implement batch embedding
        pass

    def get_dimension(self) -> int:
        # Return embedding dimension
        return 1536  # Example dimension
```

### Adding a New Vector Store

```python
from core.core import VectorStore, Chunk, SearchResult
from typing import List, Optional, Dict, Any

class CustomVectorStore(VectorStore):
    def insert_chunks(self, chunks: List[Chunk]) -> bool:
        # Implement chunk insertion
        pass

    def search_similar(self, query_embedding: List[float],
                      top_n: int = 10,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        # Implement similarity search
        pass

    def delete_by_doc_id(self, doc_id: str) -> bool:
        # Implement document deletion
        pass
```

### Adding a New Generative Model

```python
from core.core import GenerativeModel, Chunk
from typing import List

class CustomGenerativeModel(GenerativeModel):
    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        # Implement text generation
        pass

    def generate_with_context(self, query: str, context_chunks: List[Chunk],
                            max_tokens: int = 1000, **kwargs) -> str:
        # Implement context-aware generation
        pass
```

## 🛠️ Development

### Project Structure

- **`core/`**: Abstract interfaces and core data models
- **`embedding/`**: Embedding model implementations
- **`fulltext/`**: Full-text search implementations
- **`generate/`**: Text generation model implementations
- **`rerank/`**: Document reranking implementations
- **`vector/`**: Vector store implementations
- **`logger.py`**: Logging utilities
- **`example.py`**: Example usage and factory functions

### Configuration Options

The `RAGConfig` class supports the following parameters:

```python
@dataclass
class RAGConfig:
    # Model configurations
    embedding_model_config: Dict[str, Any] = field(default_factory=dict)
    rerank_model_config: Dict[str, Any] = field(default_factory=dict)
    generative_model_config: Dict[str, Any] = field(default_factory=dict)
    vector_store_config: Dict[str, Any] = field(default_factory=dict)
    text_search_config: Dict[str, Any] = field(default_factory=dict)

    # Search parameters
    vector_search_top_n: int = 20          # Top results from vector search
    text_search_top_n: int = 20            # Top results from text search
    rerank_top_n: int = 10                 # Top results after reranking
    final_context_chunks: int = 5          # Final chunks sent to generator

    # Generation parameters
    max_tokens: int = 1000                 # Maximum tokens in generated response
```

## 📊 Data Models

### Document Structure

```python
@dataclass
class Document:
    doc_id: str                    # Unique document identifier
    original_uuid: str             # Original document UUID
    content: str                   # Full document content
    chunks: List[Chunk]            # Document chunks
    metadata: Dict[str, Any]       # Additional metadata
    created_at: Optional[datetime] # Creation timestamp
```

### Chunk Structure

```python
@dataclass
class Chunk:
    chunk_id: str                  # Unique chunk identifier
    doc_id: str                    # Parent document ID
    content: str                   # Chunk content
    chunk_type: ChunkType          # Chunk type (TEXT, etc.)
    original_index: int            # Original position in document
    metadata: Dict[str, Any]       # Additional metadata
    embedding: Optional[List[float]] # Vector embedding
    created_at: Optional[datetime] # Creation timestamp
```

## 🔍 Search Flow

1. **Query Processing**: Input query is processed for both vector and text search
2. **Embedding Generation**: Query is embedded using the configured embedding model
3. **Parallel Search**:
   - Vector search finds semantically similar chunks
   - Full-text search finds keyword-relevant chunks
4. **Result Fusion**: Results are combined and deduplicated
5. **Reranking** (optional): Results are reranked for relevance using the rerank model
6. **Context Selection**: Top chunks are selected as context
7. **Answer Generation**: Final answer is generated using the context chunks

## 📝 API Reference

### RAGSystem Methods

#### `ingest_documents(documents: List[Document]) -> bool`

Ingest a list of Document objects into the system.

#### `ingest_from_pipeline_json(pipeline_data: List[Dict[str, Any]]) -> bool`

Ingest documents from pipeline JSON format.

#### `query(query: str, strategy: SearchStrategy, use_reranking: bool = True, **kwargs) -> Dict[str, Any]`

Execute a complete RAG query pipeline.

#### `search(query: str, strategy: SearchStrategy, **kwargs) -> List[SearchResult]`

Perform search without generation.

#### `delete_document(doc_id: str) -> bool`

Delete a document and all its chunks from both stores.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:

- Check the example usage in `example.py`
- Review the abstract interfaces in `core/core.py`
- Examine existing implementations for guidance
- Open an issue for bugs or feature requests

## 🚧 Roadmap

- [ ] Add support for more embedding providers (OpenAI, Hugging Face)
- [ ] Implement document update capabilities
- [ ] Add support for multimodal content (images, tables)
- [ ] Enhance metadata filtering capabilities
- [ ] Add streaming response support
- [ ] Implement caching for embeddings and search results
- [ ] Add comprehensive test suite
- [ ] Performance optimization and benchmarking
