from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

@dataclass
class RAGConfig:
    """RAG system configuration"""
    embedding_model_config: Dict[str, Any] = field(default_factory=dict)
    rerank_model_config: Dict[str, Any] = field(default_factory=dict)
    generative_model_config: Dict[str, Any] = field(default_factory=dict)
    vector_store_config: Dict[str, Any] = field(default_factory=dict)
    
    # Search parameters
    vector_search_top_n: int = 20
    rerank_top_n: int = 10
    final_context_chunks: int = 5
    
    # Generation parameters
    max_tokens: int = 1000

class SearchStrategy(Enum):
    VECTOR_ONLY = "vector_only"
    
class ChunkType(Enum):
    TEXT = "text"
    # Future support: IMAGE = "image", TABLE = "table", etc.


@dataclass
class Chunk:
    """Basic chunk unit for RAG processing"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    original_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = field(default_factory=datetime.now)


@dataclass
class Document:
    """Document container with chunks"""
    doc_id: str
    original_uuid: str
    content: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = field(default_factory=datetime.now)

    @classmethod
    def from_pipeline_json(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from pipeline JSON format"""
        chunks = []
        for chunk_data in data.get('chunks', []):
            chunk = Chunk(
                chunk_id=chunk_data['chunk_id'],
                doc_id=data['doc_id'],
                content=chunk_data['content'],
                chunk_type=ChunkType(chunk_data['type']),
                original_index=chunk_data['original_index']
            )
            chunks.append(chunk)
        
        return cls(
            doc_id=data['doc_id'],
            original_uuid=data['original_uuid'],
            content=data['content'],
            chunks=chunks
        )


@dataclass
class SearchResult:
    """Search result container"""
    chunk: Chunk
    score: float
    search_type: str  # "vector", "text", "hybrid"
    rank: int = 0


@dataclass
class RerankResult:
    """Rerank result container"""
    chunk: Chunk
    relevance_score: float
    original_rank: int
    new_rank: int


class EmbeddingModel(ABC):
    """Abstract embedding model interface"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class RerankModel(ABC):
    """Abstract rerank model interface"""
    
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk], top_n: int = 10) -> List[RerankResult]:
        """Rerank chunks based on query relevance"""
        pass


class GenerativeModel(ABC):
    """Abstract generative model interface"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text based on prompt"""
        pass
    
    @abstractmethod
    def generate_with_context(self, query: str, context_chunks: List[Chunk], 
                            max_tokens: int = 1000, **kwargs) -> str:
        """Generate answer with context chunks"""
        pass


class VectorStore(ABC):
    """Abstract vector database interface"""
    
    @abstractmethod
    def insert_chunks(self, chunks: List[Chunk]) -> bool:
        """Insert chunks with embeddings"""
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: List[float], 
                      top_n: int = 10, 
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        pass
