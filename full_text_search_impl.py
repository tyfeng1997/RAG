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



class TiDBChunkModel(TableModel, table=True):
    """TiDB model for full-text search"""
    __tablename__ = "rag_chunks"
    
    chunk_id: str = Field(primary_key=True)
    doc_id: str = Field()
    content: str = Field()
    chunk_type: str = Field(default="text")
    original_index: int = Field(default=0)
    metadata_json: str = Field(default="{}")  # JSON string for metadata


class TiDBTextSearchStore(TextSearchStore):
    """TiDB full-text search implementation"""
    
    def __init__(self, host: str, port: int = 4000, username: str = "", 
                 password: str = "", database: str = "test"):
        self.db = TiDBClient.connect(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            

        )
        self.table = self.db.create_table(schema=TiDBChunkModel,if_exists="skip")
        
        # Create full-text index if not exists
        if not self.table.has_fts_index("content"):
            self.table.create_fts_index("content")
    
    def insert_chunks(self, chunks: List[Chunk]) -> bool:
        """Insert chunks for text search"""
        try:
            import json
            
            chunk_models = []
            for chunk in chunks:
                chunk_model = TiDBChunkModel(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    chunk_type=chunk.chunk_type.value,
                    original_index=chunk.original_index,
                    metadata_json=json.dumps(chunk.metadata)
                )
                chunk_models.append(chunk_model)
            
            self.table.bulk_insert(chunk_models)
            return True
            
        except Exception as e:
            print(f"Error inserting chunks for text search: {e}")
            return False
    
    def search_text(self, query: str, 
                   top_n: int = 10,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Full-text search for chunks"""
        try:
            import json
            
            # Perform full-text search
            query_builder = self.table.search(query, search_type="fulltext").limit(top_n).text_column("content")
            
            # Add metadata filters if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key == "doc_id":
                        query_builder = query_builder.filter(TiDBChunkModel.doc_id == value)
            
            results = query_builder.to_pandas()
            
            search_results = []
            for i, row in results.iterrows():
                # Parse metadata
                try:
                    metadata = json.loads(row['metadata_json'])
                except:
                    metadata = {}
                
                chunk = Chunk(
                    chunk_id=row['chunk_id'],
                    doc_id=row['doc_id'],
                    content=row['content'],
                    chunk_type=ChunkType(row['chunk_type']),
                    original_index=row['original_index'],
                    metadata=metadata
                )
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=1.0,  # TiDB doesn't return BM25 scores directly
                    search_type="text",
                    rank=i
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Error in text search: {e}")
            return []
    
    def delete_by_doc_id(self, doc_id: str) -> bool:
        pass
    
    
if __name__ == "__main__":
    # Example usage
    text_search_store = TiDBTextSearchStore(
        host=os.getenv("TIDB_DATABASE_HOST"),
        port=int(os.getenv("TIDB_DATABASE_PORT")),
        username=os.getenv("TIDB_DATABASE_USERNAME"),
        password=os.getenv("TIDB_DATABASE_PASSWORD"),
        database=os.getenv("TIDB_DATABASE_NAME")
    )
    
    # Insert example chunks
    # chunks = [
    #     Chunk(chunk_id="3", doc_id="doc3", content="i love china", chunk_type=ChunkType.TEXT),
    # ]
    
    # text_search_store.insert_chunks(chunks)
    
    # Perform search
    results = text_search_store.search_text("love", top_n=5)
    for result in results:
        print(f"Found chunk: {result.chunk.content} with score {result.score}")