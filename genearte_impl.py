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

load_dotenv()

class AnthropicGenerativeModel(GenerativeModel):
    """Anthropic Claude implementation"""
    
    def __init__(self, model_name: str = "claude-opus-4-20250514", api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text based on prompt"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return message.content[0].text
    # def generate_with_streaming(self, prompt: str, max_tokens: int = 1000, **kwargs) :
    #     """Generate text with streaming response"""
        
    #     with self.client.stream(
    #         model=self.model_name,
    #         max_tokens=max_tokens,
    #         messages=[{"role": "user", "content": prompt}],
    #         **kwargs
    #     ) as stream:
    #         for text in stream.text_stream:
    #             print(text, end='', flush=True)
    #             yield text

    def generate_with_context(self, query: str, context_chunks: List[Chunk], 
                            max_tokens: int = 1000, **kwargs) -> str:
        """Generate answer with context chunks"""
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"Context {i}:\n{chunk.content}\n")
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""Based on the following context information, please answer the user's question.

                Context Information:
                {context_text}

                User Question: {query}

                Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""

        return self.generate(prompt, max_tokens, **kwargs)


if __name__ == "__main__":
    
    from embedding_impl import CohereEmbeddingModel, CohereRerankModel
    cohere_rerank_model = CohereRerankModel()
    query = "What is the capital of the United States?"
    chunks = [
        Chunk(chunk_id=1, doc_id=1, content="The capital of the United States is Washington, D.C."),
        Chunk(chunk_id=2, doc_id=2, content="The capital of France is Paris."),
        Chunk(chunk_id=3, doc_id=3, content="The capital of Germany is Berlin.")
    ]
    results = cohere_rerank_model.rerank(query, chunks, top_n=2)
    
    
    # Initialize Anthropic generative model
    anthropic_model = AnthropicGenerativeModel(api_key=os.getenv("ANTHROPIC_API_KEY"))
    


    response = anthropic_model.generate_with_context(query=query, context_chunks=[result.chunk for result in results], max_tokens=500)
    print(f"Generated response: {response}")
   