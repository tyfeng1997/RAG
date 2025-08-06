
from typing import List, Dict, Any, Optional, Union
import asyncio
from dataclasses import asdict

from rag_core import (
    Document, Chunk, SearchResult, RerankResult, RAGConfig, SearchStrategy,
    EmbeddingModel, RerankModel, GenerativeModel, VectorStore, TextSearchStore
)

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 rerank_model: RerankModel,
                 generative_model: GenerativeModel,
                 vector_store: VectorStore,
                 text_search_store: TextSearchStore,
                 config: RAGConfig):
        
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.generative_model = generative_model
        self.vector_store = vector_store
        self.text_search_store = text_search_store
        self.config = config
        
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
            
            # Store in both vector and text search stores
            vector_success = self.vector_store.insert_chunks(all_chunks)
            text_success = self.text_search_store.insert_chunks(all_chunks)
            
            return vector_success and text_success
            
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            return False
    
    def ingest_from_pipeline_json(self, pipeline_data: List[Dict[str, Any]]) -> bool:
        """Ingest documents from pipeline JSON format"""
        try:
            documents = []
            for data in pipeline_data:
                doc = Document.from_pipeline_json(data)
                documents.append(doc)
            
            return self.ingest_documents(documents)
            
        except Exception as e:
            print(f"Error ingesting from pipeline JSON: {e}")
            return False
    
    
    def _vector_search(self, query: str, top_n: Optional[int] = None, **kwargs) -> List[SearchResult]:
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
    def _text_search(self, query: str, top_n: Optional[int] = None, **kwargs) -> List[SearchResult]:
        """Perform full-text search"""
        top_n = top_n or self.config.text_search_top_n

        # Search text store
        results = self.text_search_store.search_text(
            query,
            top_n=top_n,
            filter_metadata=kwargs.get('filter_metadata')
        )
        
        return results
    
    def _hybrid_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Perform hybrid search (vector + text)"""
        # Get results from both search methods
        vector_results = self._vector_search(query, **kwargs)
        text_results = self._text_search(query, **kwargs)
        
        # Combine and deduplicate results
        seen_chunk_ids = set()
        combined_results = []
        
        # Add vector results first
        for result in vector_results:
            if result.chunk.chunk_id not in seen_chunk_ids:
                combined_results.append(result)
                seen_chunk_ids.add(result.chunk.chunk_id)
        
        # Add text results that weren't already found
        for result in text_results:
            if result.chunk.chunk_id not in seen_chunk_ids:
                result.search_type = "text"
                combined_results.append(result)
                seen_chunk_ids.add(result.chunk.chunk_id)
        
        # Re-rank combined results
        combined_results = sorted(combined_results, key=lambda x: x.score, reverse=True)
        
        return combined_results

    
    def search(self, query: str, 
               strategy: SearchStrategy = SearchStrategy.HYBRID,
               **kwargs) -> List[SearchResult]:
        """Search for relevant chunks"""
        
        if strategy == SearchStrategy.VECTOR_ONLY:
            return self._vector_search(query, **kwargs)
        elif strategy == SearchStrategy.TEXT_ONLY:
            return self._text_search(query, **kwargs)
        elif strategy == SearchStrategy.HYBRID:
            return self._hybrid_search(query, **kwargs)
        
    def rerank_results(self, query: str, search_results: List[SearchResult]) -> List[RerankResult]:
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
    def generate_answer(self, query: str, 
                       context_chunks: Optional[List[Chunk]] = None,
                       **kwargs) -> str:
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
    
    def query(self, query: str, 
              strategy: SearchStrategy = SearchStrategy.HYBRID,
              use_reranking: bool = True,
              **kwargs) -> Dict[str, Any]:
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
        answer = self.generate_answer(query, context_chunks, **kwargs)
        
        return {
            "query": query,
            "answer": answer,
            "search_results": [asdict(result) for result in search_results],
            "rerank_results": [asdict(result) for result in rerank_results],
            "context_chunks": [asdict(chunk) for chunk in context_chunks],
            "strategy": strategy.value
        }
    def delete_document(self, doc_id: str) -> bool:
        
        """Delete a document and all its chunks"""
        try:
            vector_success = self.vector_store.delete_by_doc_id(doc_id)
            text_success = self.text_search_store.delete_by_doc_id(doc_id)
            return vector_success and text_success
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False


