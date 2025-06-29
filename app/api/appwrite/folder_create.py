from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.permission import Permission
from appwrite.role import Role
from appwrite.id import ID
import time

folder_create_router = APIRouter(prefix="/admin", tags=["Folders"])

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

# Request body model
class FolderCreateRequest(BaseModel):
    parent_collection_id: str
    folder_name: str

@folder_create_router.post("/create-folder")
async def create_folder(payload: FolderCreateRequest):
    try:
        new_collection_id = ID.unique()

        # 1. Create new collection
        databases.create_collection(
            database_id=DATABASE_ID,
            collection_id=new_collection_id,
            name=payload.folder_name,
            permissions=[
                Permission.read(Role.any()),
                Permission.write(Role.any()),
                Permission.update(Role.any()),
                Permission.delete(Role.any()),
            ]
        )

        # 2. Add attributes
        databases.create_string_attribute(DATABASE_ID, new_collection_id, "folder_name", 255, True)
        databases.create_boolean_attribute(DATABASE_ID, new_collection_id, "is_folder", True)
        databases.create_string_attribute(DATABASE_ID, new_collection_id, "sub_folder_id", 255, True)

        # 3. Wait for attributes to be indexed
        time.sleep(2)

        # 4. Add folder reference to parent collection
        document = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=payload.parent_collection_id,
            document_id=ID.unique(),
            data={
                "folder_name": payload.folder_name,
                "is_folder": True,
                "sub_folder_id": new_collection_id,
            },
            permissions=[
                Permission.read(Role.any()),
                Permission.write(Role.any()),
                Permission.update(Role.any()),
                Permission.delete(Role.any()),
            ]
        )

        return {
            "message": "Folder created and added to parent collection.",
            "collection_id": new_collection_id,
            "document_id": document["$id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")
