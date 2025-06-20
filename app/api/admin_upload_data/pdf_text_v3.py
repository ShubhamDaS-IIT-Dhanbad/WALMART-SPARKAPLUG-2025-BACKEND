from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import uuid
import tempfile
import json
import re
from typing import List, Optional
from dotenv import load_dotenv
import tiktoken
import openai
import shutil

from app.core.config import settings
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from urlextract import URLExtract
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# ------------------ Setup ------------------
pdf_text_router_v3 = APIRouter(prefix="/upload", tags=["post"])
load_dotenv()

PROJECT_ID = "6825c9130002bf2b1514"
DATABASE_ID = "6836c51200377ed9fbdd"
API_KEY = "standard_92aaa34dd0375dc1bf9c36180dd91c3a923a2c8d6e92da38a609ce0d6d00734c62700cb1fe23218bd959e64552ff396f740faf1c3d0c2cb66cfc5f164e9ec845eb1750ebc8d4356e4d9c1a16a1f68bc446b6fa45dbebaee001ceb66a4447dfc4fff677b8125718833c4e5a099c450a97d875ed0b1d4eb115bbf3d06e09b7b039"
APPWRITE_ENDPOINT = "https://fra.cloud.appwrite.io/v1"

client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)
databases = Databases(client)

PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_INDEX_NAME = "ism-buddy-dim-1536"
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

url_extractor = URLExtract()
URL_REGEX = re.compile(r'''((?:http|ftp)s?://[^\s<>"'\]\)]{4,})''', flags=re.IGNORECASE)

# ------------------ Utility Functions ------------------

def clean_and_tag_urls(text: str) -> tuple[str, list]:
    urls = set(m.group(0).strip(').,]') for m in URL_REGEX.finditer(text))
    urls.update(url_extractor.find_urls(text))
    urls = sorted(urls)

    if urls:
        tagged = text.rstrip() + "\n\n<<URLS>>\n" + "\n".join(urls)
    else:
        tagged = text
    return tagged, urls


async def process_chunks(chunks: List[Document], filename: str):
    all_vectors = []
    all_ids = []
    total_tokens = 0

    for counter, chunk in enumerate(chunks):
        vector_id = f"{filename}_{counter}"
        text_content = chunk.page_content

        merged_text, all_links = clean_and_tag_urls(text_content)
        embedding = embedding_model.embed_query(merged_text)
        token_count = len(encoding.encode(merged_text))
        total_tokens += token_count

        all_vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": merged_text,
                "links": all_links
            }
        })
        all_ids.append(vector_id)

    return all_vectors, all_ids, total_tokens

# ------------------ Routes ------------------

@pdf_text_router_v3.post("/text")
async def upload_text_file(
    filename: str = Form(...),
    drivelink: Optional[str] = Form(default=""),
    collection_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        if file.content_type != "text/plain":
            raise HTTPException(status_code=400, detail="Only .txt files are supported.")

        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.txt")
        with open(file_path, "wb") as f:
            f.write(await file.read())

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=900)
        chunks = splitter.split_documents(docs)

        all_vectors, all_ids, total_tokens = await process_chunks(chunks, filename)
        print(all_vectors)

        try:
            index.upsert(vectors=all_vectors)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {str(e)}")

        try:
            databases.create_document(
                database_id=DATABASE_ID,
                collection_id=collection_id,
                document_id=uuid.uuid4().hex,
                data={
                    "NAME": filename,
                    "MAX_SIZE": len(all_vectors),
                    "DRIVE_LINK": drivelink or ""
                }
            )
        except AppwriteException as e:
            index.delete(ids=all_ids)
            raise HTTPException(status_code=500, detail=f"Appwrite error: {e.message}")

        return {
            "message": f"{len(all_vectors)} chunks uploaded and metadata saved.",
            "total_tokens": total_tokens,
            "file_name": filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir)


@pdf_text_router_v3.post("/pdf")
async def upload_pdf(
    filename: str = Form(...),
    drivelink: Optional[str] = Form(default=""),
    collection_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=900)
        chunks = splitter.split_documents(docs)

        all_vectors, all_ids, total_tokens = await process_chunks(chunks, filename)
        print(all_vectors)

        try:
            index.upsert(vectors=all_vectors)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pinecone upsert failed: {str(e)}")

        try:
            data = {
                "NAME": filename,
                "MAX_SIZE": len(all_vectors)
            }

            if drivelink:
                data["DRIVE_LINK"] = drivelink

            databases.create_document(
                database_id=DATABASE_ID,
                collection_id=collection_id,
                document_id=uuid.uuid4().hex,
                data=data
            )
        except AppwriteException as e:
            index.delete(ids=all_ids)
            raise HTTPException(status_code=500, detail=f"Appwrite error: {e.message}")

        return {
            "message": f"{len(all_vectors)} chunks uploaded and metadata saved.",
            "total_tokens": total_tokens,
            "file_name": filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir)
