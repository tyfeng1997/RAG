from dataclasses import asdict
from typing import Any, Dict, List, Optional

from logger import setup_logger
from core.core import (
    Chunk,
    Document,
    EmbeddingModel,
    GenerativeModel,
    RAGConfig,
    RerankModel,
    RerankResult,
    SearchResult,
    SearchStrategy,
    VectorStore,
)

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        rerank_model: RerankModel,
        generative_model: GenerativeModel,
        vector_store: VectorStore,
        config: RAGConfig,
    ):
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.generative_model = generative_model
        self.vector_store = vector_store
        self.config = config
        self.logger = setup_logger(__name__)
    
        
    def ingest_documents(self, documents: List[Document]) -> bool:
        
        """Ingest documents into the RAG system"""
        try:
            all_chunks = []
            for doc in documents:
                all_chunks.extend(doc.chunks)
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_model.embed_batch(chunk_texts)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding
            
            vector_success = self.vector_store.insert_chunks(all_chunks)
            
            return vector_success
            
        except Exception as e:
            self.logger.error(f"Error ingesting documents: {e}")
            return False

    def ingest_from_pipeline_json(self, pipeline_data: List[Dict[str, Any]]) :
        """Ingest documents from pipeline JSON format"""
        try:
            documents = []
            for data in pipeline_data:
                doc = Document.from_pipeline_json(data)
                documents.append(doc)
            
            return self.ingest_documents(documents),len(documents)
            
        except Exception as e:
            self.logger.error(f"Error ingesting from pipeline JSON: {e}")
            return False
    
    
    def _vector_search(
        self, query: str, top_n: Optional[int] = None, **kwargs
    ) -> List[SearchResult]:
        """Perform vector search"""
        top_n = top_n or self.config.vector_search_top_n

        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search_similar(
            query_embedding, 
            top_n=top_n,
            filter_metadata=kwargs.get('filter_metadata')
        )
        
        return results

    def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.VECTOR_ONLY,
        **kwargs
    ) -> List[SearchResult]:
        """Search for relevant chunks"""
        
        if strategy == SearchStrategy.VECTOR_ONLY:
            return self._vector_search(query, **kwargs)
        else:
            self.logger.error(f"Unsupported search strategy: {strategy}")
            return []
        
    def rerank_results(
        self, query: str, search_results: List[SearchResult]
    ) -> List[RerankResult]:
        """Rerank search results for better relevance"""
        if not search_results:
            return []
        
        # Extract chunks for reranking
        chunks = [result.chunk for result in search_results]
        
        # Perform reranking
        rerank_results = self.rerank_model.rerank(
            query, 
            chunks, 
            top_n=self.config.rerank_top_n
        )
        
        return rerank_results
    def generate_answer(
        self,
        query: str,
        context_chunks: Optional[List[Chunk]] = None,
        **kwargs
    ) -> str:
        """Generate answer using context chunks"""
        if context_chunks is None:
            context_chunks = []
        
        # Limit context chunks to configured maximum
        context_chunks = context_chunks[:self.config.final_context_chunks]
        
        # Generate answer
        answer = self.generative_model.generate_with_context(
            query,
            context_chunks,
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
        )
        
        return answer
    
    def query(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.VECTOR_ONLY,
        use_reranking: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete RAG query pipeline"""
        
        # Step 1: Search for relevant chunks
        search_results = self.search(query, strategy, **kwargs)
        
        if not search_results:
            return {
                "query": query,
                "answer": "I couldn't find any relevant information to answer your question.",
                "search_results": [],
                "rerank_results": [],
                "context_chunks": []
            }
        
        # Step 2: Rerank results (optional)
        context_chunks = []
        rerank_results = []
        
        if use_reranking:
            rerank_results = self.rerank_results(query, search_results)
            context_chunks = [result.chunk for result in rerank_results]
        else:
            # Use top search results directly
            top_results = search_results[:self.config.final_context_chunks]
            context_chunks = [result.chunk for result in top_results]
        
        # Step 3: Generate answer
        # answer = self.generate_answer(query, context_chunks, **kwargs)
        
        return {
            # "query": query,
            # "answer": answer,
            # "search_results": [asdict(result) for result in search_results],
            # "rerank_results": [asdict(result) for result in rerank_results],
            "context_chunks": [asdict(chunk) for chunk in context_chunks],
            # "strategy": strategy.value
        }
    def delete_document(self, doc_id: str) -> bool:
        
        """Delete a document and all its chunks"""
        try:
            vector_success = self.vector_store.delete_by_doc_id(doc_id)
            text_success = self.text_search_store.delete_by_doc_id(doc_id)
            return vector_success and text_success
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False


