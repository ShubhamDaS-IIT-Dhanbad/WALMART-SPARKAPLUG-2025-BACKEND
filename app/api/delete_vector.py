from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from app.core.config import settings

delete_router = APIRouter(prefix="/delete", tags=["delete"])

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("iit-ism-chat-bot-1536")

class DeleteRequest(BaseModel):
    name: str
    max_id: int

@delete_router.post("/")
def delete_vectors(payload: DeleteRequest):
    try:
        print(payload)
        ids_to_delete = [f"{payload.name}_{i}" for i in range(payload.max_id + 1)]
        index.delete(ids=ids_to_delete)
        return {"status": "success", "deleted_count": len(ids_to_delete)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
