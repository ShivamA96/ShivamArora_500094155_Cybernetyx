# utility.py
from fastapi import UploadFile, HTTPException
import aiofiles
from PyPDF2 import PdfReader
from docx import Document


async def extract_text(file: UploadFile) -> str:
    if file.filename.endswith(".pdf"):
        reader = PdfReader(await file.read())
        return " ".join(page.extract_text() for page in reader.pages)
    elif file.filename.endswith(".docx"):
        doc = Document(await file.read())
        return " ".join(para.text for para in doc.paragraphs)
    elif file.filename.endswith(".txt"):
        async with aiofiles.open(file.file, 'r') as f:
            return await f.read()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
