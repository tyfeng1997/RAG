import os
from typing import List, Optional

import cohere
from dotenv import load_dotenv

from core.core import EmbeddingModel

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