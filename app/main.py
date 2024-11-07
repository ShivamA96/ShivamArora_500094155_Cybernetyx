# main.py
from fastapi import FastAPI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from api.endpoints import router
from contextlib import asynccontextmanager


chroma_client = chromadb.Client(Settings(persist_directory="chroma_db"))
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device="cpu")


@asynccontextmanager
async def lifetime(app: FastAPI):
    app.state.chroma_client = chroma_client
    app.state.model = model
    try:
        yield
    finally:
        pass

app = FastAPI(lifespan=lifetime)
app.router.include_router(router)
