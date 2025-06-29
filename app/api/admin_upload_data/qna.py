from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json
import uuid
import os
import re
from typing import List, Optional
from dotenv import load_dotenv

from app.core.config import settings
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from urlextract import URLExtract

load_dotenv()

qna_router = APIRouter(prefix="/upload", tags=["post"])

# ---------- Appwrite Setup ----------
from app.core.config import settings

PROJECT_ID =settings.APPWRITE_PROJECT_ID
DATABASE_ID =settings.APPWRITE_DATABASE_ID
API_KEY = settings.APPWRITE_API_KEY
APPWRITE_ENDPOINT =settings.APPWRITE_ENDPOINT

client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)
databases = Databases(client)

# ---------- Pinecone Setup ----------
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host=settings.PINECONE_INDEX_URI)
except Exception as e:
    print(f"❌ Error initializing Pinecone: {e}")
    index = None

# ---------- Embedding + URL Extraction ----------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
url_extractor = URLExtract()
URL_REGEX = re.compile(r'''((?:http|ftp)s?://[^\s<>"'\]\)]{4,})''', flags=re.IGNORECASE)

def extract_links(text: str) -> list:
    raw_links = set(m.group(0).strip(').,]') for m in URL_REGEX.finditer(text))
    raw_links.update(url_extractor.find_urls(text))
    return sorted(link.replace("\n", "").strip() for link in raw_links)

# ---------- QnA Upload Route ----------
@qna_router.post("/qna")
async def upload_qna_file(
    file: UploadFile = File(...),
    filename: str = Form(...),
    drivelink: Optional[str] = Form(default=""),
    collection_id: str = Form(...)
):
    file_name = filename.strip()

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a JSON file.")

    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in uploaded file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON must be a list of Q&A objects.")
    if not data:
        return {"message": "No data provided in file.", "uploaded_count": 0}

    vectors_to_upsert = []
    vector_ids = []

    for idx, item in enumerate(data):
        question = item.get("question")
        answer = item.get("answer")

        if not question or not answer:
            print(f"⚠️ Skipping item {idx}: missing question or answer.")
            continue

        try:
            full_text = f"question: {question} + answer: {answer}".replace("\n", " ").strip()
            links = extract_links(full_text)
            embedded_vector = embedding_model.embed_query(full_text)

            vector_id = f"{file_name}_{idx}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedded_vector,
                "metadata": {
                    "text": full_text,
                    "links": links
                }
            })

            vector_ids.append(vector_id)
            print(f"✅ Vector ID: {vector_id}")
        except Exception as embed_err:
            print(f"❌ Embedding error at index {idx}: {embed_err}")
            continue

    if not vectors_to_upsert:
        return {"message": "No valid entries to upload.", "uploaded_count": 0}

    # ---------- Pinecone Upsert ----------
    try:
        index.upsert(vectors=vectors_to_upsert)
        print(f"✅ Upserted {len(vectors_to_upsert)} entries into Pinecone.")
    except Exception as e:
        print(f"❌ Pinecone upsert error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upsert data into Pinecone.")

    # ---------- Appwrite Metadata Store ----------
    try:
        document_data = {
            "NAME": file_name,
            "MAX_SIZE": len(vectors_to_upsert)
        }

        if drivelink:
            document_data["DRIVE_LINK"] = drivelink

        doc = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=collection_id,
            document_id=uuid.uuid4().hex,
            data=document_data
        )

        doc_id = doc["$id"] if isinstance(doc, dict) else getattr(doc, "$id", None)
        print(f"✅ Stored metadata in Appwrite: {doc_id}")
    except AppwriteException as e:
        print(f"❌ Appwrite error: {e.message}")
        try:
            index.delete(ids=vector_ids)
            print(f"🗑️ Rolled back {len(vector_ids)} vectors from Pinecone.")
        except Exception as pinecone_delete_error:
            print(f"⚠️ Pinecone rollback failed: {pinecone_delete_error}")
        raise HTTPException(status_code=500, detail="Failed to store metadata in Appwrite.")

    return {
        "message": f"{len(vectors_to_upsert)} entries uploaded to Pinecone and metadata stored in Appwrite.",
        "max_id": len(vectors_to_upsert),
        "file_name": file_name,
        "doc_id": doc_id
    }
