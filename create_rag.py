from embedding_impl import (
        CohereEmbeddingModel, CohereRerankModel,
    )
from genearte_impl import  AnthropicGenerativeModel
from vectordb_impl import  TiDBVectorStore
from full_text_search_impl import TiDBTextSearchStore
from rag_core import RAGConfig
from rag_system import RAGSystem
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rag_core import SearchStrategy
load_dotenv()
def create_rag_system(config_dict: Optional[Dict[str, Any]] = None) -> RAGSystem:
    """Factory function to create RAG system with default configuration"""
    
    # Create config
    config = RAGConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Initialize models and stores
    embedding_model = CohereEmbeddingModel()
    rerank_model = CohereRerankModel()
    generative_model = AnthropicGenerativeModel()
    
    # Initialize vector store
    vector_store = TiDBVectorStore(
        vector_dimension=embedding_model.get_dimension()
    )
    
    # Initialize text search store
    # Note: You'll need to provide actual TiDB connection details
    text_search_store = TiDBTextSearchStore()
    
    return RAGSystem(
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        generative_model=generative_model,
        vector_store=vector_store,
        text_search_store=text_search_store,
        config=config
    )


if __name__ == "__main__":
    pipeline_data = [
        {
            "doc_id": "filename_doc_1",
            "original_uuid": "5e4c01057a10732d34784af2a97bee9d173863f043b9901de8ef7f57bc590145",
            "content": "This is a comprehensive document about artificial intelligence and machine learning.",
            "chunks": [
                {
                    "type": "text",
                    "chunk_id": "doc_1_chunk_0",
                    "original_index": 0,
                    "content": "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines."
                },
                {
                    "type": "text", 
                    "chunk_id": "doc_1_chunk_1",
                    "original_index": 1,
                    "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience."
                }
            ]
        }
    ]
    rag_system = create_rag_system({
        "vector_search_top_n": 10,
        "text_search_top_n": 10,
        "rerank_top_n": 5,
        "final_context_chunks": 3
    })
    
    # Ingest documents
    print("Ingesting documents...")
    # success = rag_system.ingest_from_pipeline_json(pipeline_data)
    # print(f"Ingestion successful: {success}")
    
    print("\nQuerying the RAG system...")
    result = rag_system.query(
        "What is machine learning?",
        strategy=SearchStrategy.HYBRID,
        use_reranking=True
    )
    
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Strategy used: {result['strategy']}")
    print(f"Found {len(result['search_results'])} search results")
    print(f"Used {len(result['context_chunks'])} context chunks")
