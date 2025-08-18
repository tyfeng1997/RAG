from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from mistralai import Mistral
from utils import create_rag_system,ocr_pages_to_pipeline_data
from core.core import SearchResult, SearchStrategy
from dotenv import load_dotenv
load_dotenv(".env.local")
app = FastAPI()
rag_system = create_rag_system({
    "vector_search_top_n": 10,
    "rerank_top_n": 5,
    "final_context_chunks": 5
})
class QueryRequest(BaseModel):
    text: str
    use_reranking: bool = True

class PDFIngestRequest(BaseModel):
    base64_pdf: str
    doc_id: str
    chunk_size: int = 1024 

@app.post("/query")
def query_endpoint(request: QueryRequest):
    result = rag_system.query(
        request.text,
        use_reranking=request.use_reranking
    )
    return result

@app.post("/ingest_pdf")
def ingest_pdf_endpoint(request: PDFIngestRequest):
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{request.base64_pdf}"
        },
        include_image_base64=False
    )
    pipeline_data = ocr_pages_to_pipeline_data(ocr_response.pages,doc_id=request.doc_id,chunk_size=request.chunk_size)
    
    success,num_docs = rag_system.ingest_from_pipeline_json(pipeline_data)
    return {"success": success, "num_docs": num_docs}

# 启动命令： uvicorn api:app --reload