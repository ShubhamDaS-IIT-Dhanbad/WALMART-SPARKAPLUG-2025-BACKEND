from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from appwrite.client import Client
from appwrite.query import Query
from appwrite.services.databases import Databases
from appwrite.permission import Permission
from appwrite.role import Role
from appwrite.id import ID
import time

document_folder_router = APIRouter(prefix="/admin", tags=["Folders"])

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

@document_folder_router.post("/create-document-folder")
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
        print(f"✅ Collection '{payload.folder_name}' created with ID: {new_collection_id}")

        # 2. Add attributes
        databases.create_string_attribute(
            database_id=DATABASE_ID,
            collection_id=new_collection_id,
            key="NAME",
            size=2550,  # ✅ FIXED: size parameter is required
            required=True
        )
        databases.create_string_attribute(
            database_id=DATABASE_ID,
            collection_id=new_collection_id,
            key="DRIVE_LINK",
            size=95500,  # ✅ FIXED: size parameter is required
            required=True
        )
        databases.create_integer_attribute(
            database_id=DATABASE_ID,
            collection_id=new_collection_id,
            key="MAX_SIZE",
            required=True
        )
        print("✅ Attributes added successfully")

        # 3. Wait for attributes to be indexed
        time.sleep(5)

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
        print("✅ Reference added to parent collection")

        return {
            "message": "Folder created and added to parent collection.",
            "collection_id": new_collection_id,
            "document_id": document["$id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")


class DeleteFolderRequest(BaseModel):
    parent_collection_id: str
    folder_collection_id: str

@document_folder_router.post("/delete-document-folder")
async def delete_folder(payload: DeleteFolderRequest):
    try:
        # 1. Check if folder collection has any documents
        folder_docs = databases.list_documents(
            database_id=DATABASE_ID,
            collection_id=payload.folder_collection_id
        )

        if folder_docs["total"] > 0:
            raise HTTPException(status_code=400, detail="Folder is not empty. Cannot delete.")

        # 2. Delete the folder collection
        databases.delete_collection(
            database_id=DATABASE_ID,
            collection_id=payload.folder_collection_id
        )
        print(f"🗑️ Collection {payload.folder_collection_id} deleted")

        # 3. Find document in parent collection where sub_folder_id == folder_collection_id
        parent_docs = databases.list_documents(
            database_id=DATABASE_ID,
            collection_id=payload.parent_collection_id,
            queries=[
                Query.equal("sub_folder_id", payload.folder_collection_id)
            ]
        )

        if parent_docs["total"] == 0:
            raise HTTPException(status_code=404, detail="Reference to folder not found in parent collection.")

        # 4. Delete the reference document from parent collection
        doc_id = parent_docs["documents"][0]["$id"]
        databases.delete_document(
            database_id=DATABASE_ID,
            collection_id=payload.parent_collection_id,
            document_id=doc_id
        )
        print("✅ Reference document deleted from parent collection")

        return {"message": "Folder and reference deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting folder: {str(e)}")
