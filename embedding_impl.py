import os
import cohere
import anthropic
from typing import List, Dict, Any, Optional
from tidb_vector.integrations import TiDBVectorClient
from pytidb import TiDBClient
from pytidb.schema import TableModel, Field
from dotenv import load_dotenv

from rag_core import (
    EmbeddingModel, RerankModel, GenerativeModel, VectorStore, TextSearchStore,
    Chunk, SearchResult, RerankResult, ChunkType
)

# Load environment variables
load_dotenv()

class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model implementation"""
    
    def __init__(self, model_name: str = "embed-v4.0", api_key: Optional[str] = None):
        self.client = cohere.ClientV2(api_key or os.getenv("COHERE_API_KEY"))
        self.model_name = model_name
        self._dimension = 1024
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        response = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_document",
            output_dimension=self._dimension,
            embedding_types=["float"],
        )
        return response.embeddings.float[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        # Cohere has batch size limits, process in chunks if needed
        batch_size = 96  # Cohere's typical batch limit
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type="search_document",
                output_dimension=self._dimension,
                embedding_types=["float"],
            )
            all_embeddings.extend(response.embeddings.float)
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension



class CohereRerankModel(RerankModel):
    """Cohere rerank model implementation"""
    
    def __init__(self, model_name: str = "rerank-v3.5", api_key: Optional[str] = None):
        self.client = cohere.ClientV2(api_key or os.getenv("COHERE_API_KEY"))
        self.model_name = model_name
    
    def rerank(self, query: str, chunks: List[Chunk], top_n: int = 10) -> List[RerankResult]:
        """Rerank chunks based on query relevance"""
        if not chunks:
            return []
        
        # Prepare documents for reranking
        documents = [chunk.content for chunk in chunks]
        
        # Cohere rerank API call
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model_name,
            top_n=min(top_n, len(documents))
        )
        
        # Convert to RerankResult objects
        results = []
        for i, result in enumerate(response.results):
            original_chunk = chunks[result.index]
            rerank_result = RerankResult(
                chunk=original_chunk,
                relevance_score=result.relevance_score,
                original_rank=result.index,
                new_rank=i
            )
            results.append(rerank_result)
        
        return results

if __name__ == "__main__":
    # Initialize Cohere embedding model
    cohere_rerank_model = CohereRerankModel()
    query = "What is the capital of the United States?"
    chunks = [
        Chunk(chunk_id=1, doc_id=1, content="The capital of the United States is Washington, D.C."),
        Chunk(chunk_id=2, doc_id=2, content="The capital of France is Paris."),
        Chunk(chunk_id=3, doc_id=3, content="The capital of Germany is Berlin.")
    ]
    result = cohere_rerank_model.rerank(query, chunks, top_n=2)
    for r in result:
        print(f"Chunk: {r.chunk.content}, Relevance Score: {r.relevance_score}, New Rank: {r.new_rank}")