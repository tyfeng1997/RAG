

from typing import Any, Dict, Optional
from dotenv import load_dotenv
from embedding.cohere_embedding import CohereEmbeddingModel
from rerank.cohere_rerank import CohereRerankModel
from generate.anthropic_genearte import AnthropicGenerativeModel
from core.core import RAGConfig, SearchStrategy
from core.rag import RAGSystem
from vector.TiDB_vector import TiDBVectorStore
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
    

    return RAGSystem(
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        generative_model=generative_model,
        vector_store=vector_store,
        config=config
    )




def ocr_pages_to_pipeline_data(pages,doc_id,chunk_size=1024):
    
    chunks = []
    chunk_index = 0
    
    for page in pages:
        # print(f"Page Index: {page.index}")
        # print(f"Markdown Content:\n{page.markdown}\n")
        markdown = page.markdown
        index = page.index

        for i in range(0, len(markdown), chunk_size):
            chunk_content = markdown[i:i+chunk_size]
            chunks.append({
                "type": "text",
                "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                "original_index": chunk_index,
                "content": chunk_content
            })
            chunk_index += 1
    pipeline_data = [{
        "doc_id": doc_id,
        "original_uuid": doc_id,
        "content": "", 
        "chunks": chunks
    }]
    return pipeline_data

