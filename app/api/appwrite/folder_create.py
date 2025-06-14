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
PROJECT_ID = "6825c9130002bf2b1514"
DATABASE_ID = "6836c51200377ed9fbdd"
API_KEY = "standard_92aaa34dd0375dc1bf9c36180dd91c3a923a2c8d6e92da38a609ce0d6d00734c62700cb1fe23218bd959e64552ff396f740faf1c3d0c2cb66cfc5f164e9ec845eb1750ebc8d4356e4d9c1a16a1f68bc446b6fa45dbebaee001ceb66a4447dfc4fff677b8125718833c4e5a099c450a97d875ed0b1d4eb115bbf3d06e09b7b039"  # Replace with your actual Appwrite API key
APPWRITE_ENDPOINT = "https://fra.cloud.appwrite.io/v1"

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
