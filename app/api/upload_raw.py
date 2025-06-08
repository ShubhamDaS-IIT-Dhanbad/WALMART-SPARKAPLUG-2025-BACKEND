import json
from fastapi import APIRouter, Request
from pinecone import Pinecone
from app.core.config import settings

upload_raw_router = APIRouter(prefix="/upload", tags=["upload"])

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(host=settings.PINECONE_INDEX_URI)

@upload_raw_router.post("/raw")
async def upload_raw_json(request: Request):
    body = await request.json()
    name = body.get("name")
    data = body.get("data", [])

    if not name or not isinstance(data, list):
        return {"error": "Invalid format. 'name' (str) and 'data' (list) required."}

    records = []

    for item in data:
        print(item)
        _id = item.get("_id")
        text = item.get("text", "")
        category = item.get("category", "")

        if not _id or not text:
            continue

       
        records.append({
            "_id": _id,
            "text": text,
            "category": category 
        })
    print(records)
    if records:
        index.upsert_records("example-namespace",records)

    return {
        "message": f"✅ {len(records)} entries embedded & uploaded to Pinecone.",
        "max_id": len(data)
    }
