import json
import uuid
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from pinecone import Pinecone
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException
from app.core.config import settings
import os
import datetime

upload_json_router = APIRouter(prefix="/upload", tags=["post"])

APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT", "https://fra.cloud.appwrite.io/v1")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID", "6825c9130002bf2b1514")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY", "YOUR_APPWRITE_API_KEY")
DATABASE_ID = os.getenv("APPWRITE_DATABASE_ID", "6836c51200377ed9fbdd")
COLLECTION_ID = os.getenv("APPWRITE_COLLECTION_ID", "684829ba000794b13a92")

client = Client()
client.set_endpoint(APPWRITE_ENDPOINT).set_project(APPWRITE_PROJECT_ID).set_key(APPWRITE_API_KEY)
databases = Databases(client)

try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_URI.split('/')[-1] if '/' in settings.PINECONE_INDEX_URI else "your-pinecone-index-name"
    index = pc.Index(host=settings.PINECONE_INDEX_URI)
except Exception as e:
    print(f"Error initializing Pinecone client: {e}")



@upload_json_router.post("/json/qna/delete")
async def delete_qna_file(
    file_name: str = Form(...),
    max_id: int = Form(...),
    doc_id: str = Form(...)  # Accept doc_id directly
):
    vector_ids = [f"{file_name}_{i}" for i in range(max_id)]

    # Step 1: Delete vectors from Pinecone
    try:
        index.delete(ids=vector_ids, namespace="example-namespace")
        print(f"✅ Deleted {len(vector_ids)} vectors from Pinecone.")
    except Exception as e:
        print(f"❌ Pinecone deletion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete vectors from Pinecone.")

    # Step 2: Delete metadata from Appwrite
    try:
        databases.delete_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            document_id=doc_id
        )
        print(f"✅ Deleted metadata from Appwrite for doc_id: {doc_id}")
    except AppwriteException as e:
        print(f"❌ Appwrite deletion error: {e.message}")
        raise HTTPException(status_code=500, detail="Failed to delete metadata from Appwrite.")

    return {
        "message": f"Deleted {len(vector_ids)} vectors and metadata.",
        "file_name": file_name,
        "doc_id": doc_id
    }


@upload_json_router.post("/json/qna")
async def upload_qna_file(file: UploadFile = File(...), file_name: str = Form(...)):
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file.")
        contents = await file.read()
        data = json.loads(contents.decode('utf-8'))
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

    for i, item in enumerate(data):
        question = item.get("question")
        answer = item.get("answer")
        if not question or not answer:
            print(f"Skipping item {i} due to missing fields: {item}")
            continue

        vector_id = f"{file_name}_{i}"
       
        vectors_to_upsert.append({
            "_id": vector_id,
            "text": question,
            "answer": answer
        })
        vector_ids.append(vector_id)

    try:
        index.upsert_records("example-namespace", vectors_to_upsert)
        print(f"Upserted {len(vectors_to_upsert)} entries into Pinecone.")
    except Exception as e:
        print(f"Pinecone error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upsert data into Pinecone.")

    try:
        document_data = {
            "NAME": file_name,
            "MAX_SIZE": len(vectors_to_upsert)
        }
        databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            document_id="unique()",
            data=document_data
        )
        print(f"Stored metadata in Appwrite: {document_data}")
    except AppwriteException as e:
        print(f"Appwrite error: {e.message}")
        try:
            index.delete(ids=vector_ids, namespace="example-namespace")
            print(f"Rolled back {len(vector_ids)} vectors from Pinecone.")
        except Exception as pinecone_delete_error:
            print(f"Pinecone rollback failed: {pinecone_delete_error}")
        raise HTTPException(status_code=500, detail="Failed to store metadata in Appwrite.")

    return {
        "message": f"{len(vectors_to_upsert)} entries uploaded to Pinecone and metadata stored in Appwrite.",
        "uploaded_count": len(vectors_to_upsert),
        "file_name": file_name
    }