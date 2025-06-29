from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.permission import Permission
from appwrite.role import Role
from appwrite.id import ID

document_create_router = APIRouter(prefix="/admin", tags=["document"])

# Appwrite configuration
from app.core.config import settings

PROJECT_ID =settings.APPWRITE_PROJECT_ID
DATABASE_ID =settings.APPWRITE_DATABASE_ID
API_KEY = settings.APPWRITE_API_KEY
APPWRITE_ENDPOINT =settings.APPWRITE_ENDPOINT

# Initialize Appwrite client
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)

databases = Databases(client)

# Route to create a document
@document_create_router.post("/create-document")
async def create_document(
    file_name: str = Form(...),
    max_id: int = Form(...),
    collection_id: str = Form(...)
):
    try:
        document = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=collection_id,
            data={
                "file_name": file_name,
                "max_id": max_id
            },
            permissions=[
                Permission.read(Role.any()),
                Permission.write(Role.any()),
                Permission.update(Role.any()),
                Permission.delete(Role.any()),
            ]
        )

        return {
            "message": "Document created successfully.",
            "document_id": document["$id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating document: {str(e)}")
