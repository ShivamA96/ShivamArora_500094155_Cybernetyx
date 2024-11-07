# endpoints.py
from typing import List
from fastapi import APIRouter, UploadFile, BackgroundTasks, HTTPException, Depends
from utility.extract_text_util import extract_text
from chromadb import Client
from sentence_transformers import SentenceTransformer


def get_chroma_client(request):
    return request.app.state.chroma_client


def get_model(request):
    return request.app.state.model


router = APIRouter()


@router.post("/ingest")
async def ingest_document(files: List[UploadFile], background_tasks: BackgroundTasks, chroma_client: Client = Depends(get_chroma_client), model: SentenceTransformer = Depends(get_model)):
    async def process_file(file):
        text = await extract_text(file)
        embedding = model.encode(text, convert_to_tensor=True).tolist()
        chroma_client.add_document(file.filename, text, embedding)

    for file in files:
        background_tasks.add_task(process_file, file)

    return {"message": "Ingestion Started :)"}


@router.get("/query")
async def query_documents(query: str, chroma_client: Client = Depends(get_chroma_client), model: SentenceTransformer = Depends(get_model)):
    embedding = model.encode(query, convert_to_tensor=True).tolist()
    results = chroma_client.query(embedding, top_k=5)
    return {"Results are as follows ": results}
