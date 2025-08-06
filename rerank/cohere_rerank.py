import os
from typing import List, Optional

import cohere
from dotenv import load_dotenv

from core.core import Chunk, RerankModel, RerankResult

# Load environment variables
load_dotenv()

class CohereRerankModel(RerankModel):
    """Cohere rerank model implementation"""
    
    def __init__(self, model_name: str = "rerank-v3.5", api_key: Optional[str] = None):
        self.client = cohere.ClientV2(api_key or os.getenv("COHERE_API_KEY"))
        self.model_name = model_name
    
    def rerank(
        self, query: str, chunks: List[Chunk], top_n: int = 10
    ) -> List[RerankResult]:
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
