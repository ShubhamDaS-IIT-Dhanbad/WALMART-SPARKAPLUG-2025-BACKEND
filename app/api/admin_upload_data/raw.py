import json
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pinecone import Pinecone
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException
from app.core.config import settings
import os

raw_router = APIRouter(prefix="/upload", tags=["post"])

# Appwrite configuration
PROJECT_ID = "6825c9130002bf2b1514"
DATABASE_ID = "6836c51200377ed9fbdd"
API_KEY = "standard_92aaa34dd0375dc1bf9c36180dd91c3a923a2c8d6e92da38a609ce0d6d00734c62700cb1fe23218bd959e64552ff396f740faf1c3d0c2cb66cfc5f164e9ec845eb1750ebc8d4356e4d9c1a16a1f68bc446b6fa45dbebaee001ceb66a4447dfc4fff677b8125718833c4e5a099c450a97d875ed0b1d4eb115bbf3d06e09b7b039"
APPWRITE_ENDPOINT = "https://fra.cloud.appwrite.io/v1"

# Appwrite Client
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)
databases = Databases(client)

# Pinecone Client
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host=settings.PINECONE_INDEX_URI)

except Exception as e:
    print(f"Error initializing Pinecone client: {e}")
    index = None

@raw_router.post("/raw")
async def upload_qna_file(
    file: UploadFile = File(...),
    file_name: str = Form(...),
    collection_id: str = Form(...)
):
    file_name = file_name.strip()

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

    for i, item in enumerate(data):
        question = item.get("text", "").strip()
        answer = item.get("category", "").strip()
        if not question or not answer:
            print(f"Skipping item {i}: missing fields.")
            continue

        vector_id = f"{file_name}_{i}"
        vector_ids.append(vector_id)
        vectors_to_upsert.append({
            "id": vector_id,
            "text": question,
            "category": answer
            
        })

    try:
        index.upsert_records("example-namespace",vectors_to_upsert)
        print(f"✅ Upserted {len(vectors_to_upsert)} entries into Pinecone.")
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upsert data into Pinecone.")

    try:
        document_data = {
            "NAME": file_name,
            "MAX_SIZE": len(vectors_to_upsert)
        }
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
            index.delete(ids=vector_ids, namespace="example-namespace")
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
