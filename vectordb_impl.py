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



class TiDBVectorStore(VectorStore):
    """TiDB vector store implementation"""
    
    def __init__(self, table_name: str = "rag_embeddings", 
                 connection_string: Optional[str] = None,
                 vector_dimension: int = 1024,
                 drop_existing: bool = False):
        self.table_name = table_name
        self.connection_string = connection_string or os.getenv("TIDB_DATABASE_URL")
        self.vector_dimension = vector_dimension
        
        self.client = TiDBVectorClient(
            table_name=self.table_name,
            connection_string=self.connection_string,
            vector_dimension=self.vector_dimension,
            drop_existing_table=drop_existing
        )
    
    def insert_chunks(self, chunks: List[Chunk]) -> bool:
        """Insert chunks with embeddings"""
        try:
            # Prepare data for insertion
            ids = []
            texts = []
            embeddings = []
            metadatas = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.chunk_id} missing embedding")
                
                ids.append(chunk.chunk_id)
                texts.append(chunk.content)
                embeddings.append(chunk.embedding)
                
                # Prepare metadata
                metadata = {
                    "doc_id": chunk.doc_id,
                    "chunk_type": chunk.chunk_type.value,
                    "original_index": chunk.original_index,
                    **chunk.metadata
                }
                metadatas.append(metadata)
            
            # Insert into TiDB
            self.client.insert(
                ids=ids,
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return True
            
        except Exception as e:
            print(f"Error inserting chunks: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], 
                      top_k: int = 10, 
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks"""
        try:
            # Perform vector search
            results = self.client.query(query_embedding, k=top_k)
            
            search_results = []
            for i, result in enumerate(results):
                # Reconstruct chunk from result
                metadata = result.metadata or {}
                chunk = Chunk(
                    chunk_id=result.id,
                    doc_id=metadata.get("doc_id", ""),
                    content=result.document,
                    chunk_type=ChunkType(metadata.get("chunk_type", "text")),
                    original_index=metadata.get("original_index", 0),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ["doc_id", "chunk_type", "original_index"]}
                )
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=1.0 - result.distance,  # Convert distance to similarity
                    search_type="vector",
                    rank=i
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete all chunks for a document"""
        pass


if __name__ == "__main__":
    # Example usage of TiDBVectorStore
    vector_store = TiDBVectorStore(
        table_name="rag_embeddings",
        connection_string=os.getenv("TIDB_DATABASE_URL"),
        vector_dimension=1024,
        drop_existing=True
    )
    
    # Create example chunks
    chunks = [
        Chunk(chunk_id=1, doc_id="doc1", content="This is a test chunk.", 
              chunk_type=ChunkType.TEXT, original_index=0, 
              embedding=[0.1] * 1024, metadata={"source": "test"}),
        Chunk(chunk_id=2, doc_id="doc2", content="Another test chunk.", 
              chunk_type=ChunkType.TEXT, original_index=1, 
              embedding=[0.2] * 1024, metadata={"source": "test"})
    ]
    
    # Insert chunks
    if vector_store.insert_chunks(chunks):
        print("Chunks inserted successfully")
    
    # Search similar chunks
    query_embedding = [0.15] * 1024
    results = vector_store.search_similar(query_embedding, top_k=5)
    for result in results:
        print(f"Found chunk: {result.chunk.content} with score {result.score}")