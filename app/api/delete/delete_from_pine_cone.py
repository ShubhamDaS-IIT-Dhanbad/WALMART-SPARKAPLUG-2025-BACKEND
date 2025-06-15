import os
import logging
from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel
from pinecone import Pinecone
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException

# ================== ENV + CONFIG ==================

# Appwrite env variables
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT", "https://fra.cloud.appwrite.io/v1")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID", "6825c9130002bf2b1514")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY", "YOUR_APPWRITE_API_KEY")
DATABASE_ID = os.getenv("APPWRITE_DATABASE_ID", "6836c51200377ed9fbdd")

# Pinecone API Key from settings (Make sure settings.PINECONE_API_KEY is defined)
from app.core.config import settings

# ================== INIT SERVICES ==================

# Appwrite client setup
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT).set_project(APPWRITE_PROJECT_ID).set_key(APPWRITE_API_KEY)
databases = Databases(client)

# Pinecone client setup
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(host=settings.PINECONE_INDEX_URI)
# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# FastAPI router
delete_from_pine_cone_router = APIRouter(prefix="/delete", tags=["delete"])


# ================== SCHEMAS ==================

class DeleteRequest(BaseModel):
    name: str
    max_id: int


# ================== ROUTES ==================

@delete_from_pine_cone_router.post("/")
def delete_vectors(payload: DeleteRequest):
    try:
        logger.info(f"Received deletion request: {payload}")
        ids_to_delete = [f"{payload.name}_{i}" for i in range(payload.max_id + 1)]
        index.delete(ids=ids_to_delete)
        return {"status": "success", "deleted_count": len(ids_to_delete)}
    except Exception as e:
        logger.error(f"❌ Error deleting from Pinecone: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")





@delete_from_pine_cone_router.post("/document/delete")
async def delete_qna_file(
    file_name: str = Form(...),
    max_id: int = Form(...),
    doc_id: str = Form(...),
    collection_id: str = Form(...)  # ✅ collection_id now comes from frontend
):
    # Step 1: Delete vectors from Pinecone
    try:
        vector_ids = [f"{file_name}_{i}" for i in range(max_id + 1)]
        index.delete(ids=vector_ids)
        logger.info(f"✅ Deleted {len(vector_ids)} vectors from Pinecone.")
    except Exception as e:
        logger.error(f"❌ Pinecone deletion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete vectors from Pinecone.")

    # Step 2: Delete document from Appwrite
    try:
        databases.delete_document(
            database_id=DATABASE_ID,
            collection_id=collection_id,  # ✅ Use the passed collection_id
            document_id=doc_id
        )
        logger.info(f"✅ Deleted metadata from Appwrite for doc_id: {doc_id}")
    except AppwriteException as e:
        logger.error(f"❌ Appwrite deletion error: {e.message}")
        raise HTTPException(status_code=500, detail="Failed to delete metadata from Appwrite.")

    return {
        "message": f"Deleted {len(vector_ids)} vectors and metadata.",
        "file_name": file_name,
        "doc_id": doc_id,
        "collection_id": collection_id
    }
