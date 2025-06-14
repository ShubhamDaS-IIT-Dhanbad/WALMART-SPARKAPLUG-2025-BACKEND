import json
import os
import re
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, Form
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
import pdfplumber

from app.core.config import settings

openai = OpenAI(api_key=settings.OPENAI_API_KEY)
pinecone = Pinecone(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)
index = pinecone.Index(settings.PINECONE_INDEX_NAME)

pdf_text_router = APIRouter(prefix="/upload", tags=["upload"])


def clean_gpt_response(text: str) -> str:
    text = re.sub(r"^```json?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def read_text_file(file: UploadFile) -> str:
    content = file.file.read()
    return content.decode("utf-8")


def read_pdf_file(file: UploadFile) -> str:
    pdf_bytes = file.file.read()
    pdf_stream = BytesIO(pdf_bytes)
    text = ""
    with pdfplumber.open(pdf_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, max_tokens: int = 3000):
    enc = tiktoken.get_encoding("cl100k_base")
    lines = text.split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(enc.encode(current + line)) > max_tokens:
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


async def generate_gist_pairs(chunk: str, model: str, prompt: str):
    filled_prompt= f"""
You are an intelligent system that extracts relevant information from the following text for use in a Retrieval-Augmented Generation (RAG) system using Pinecone.

Instructions:
1. Read the text carefully and split it into logical "gist" entries with corresponding "metadata".
2. Metadata should reflect any useful tags, titles, dates, or structural information (e.g., section names, topics).
3. All "gist" values must summarize or represent meaningful pieces of content for embedding.

Return ONLY a valid JSON array like this:

[
  {{
    "gist": "Short summary or key content",
    "metadata": {{
      "topic": "Optional tag",
      "section": "Optional section name"
    }}
  }},
  ...
]
must do instruction:
 {{prompt}}
 
Only include metadata fields if they are clearly identifiable in the text.
Text:
\"\"\"{chunk}\"\"\"
"""

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": filled_prompt}]
        )
        content = response.choices[0].message.content
        cleaned = clean_gpt_response(content)
        pairs = json.loads(cleaned)
        return pairs
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"GPT output was: {content}")
        return []
    except Exception as e:
        print(f"Error generating gist pairs: {e}")
        return []


async def get_embedding(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def sanitize_metadata(metadata):
    if isinstance(metadata, dict):
        return {k: sanitize_metadata(v) for k, v in metadata.items()}
    elif isinstance(metadata, list):
        if all(isinstance(i, str) for i in metadata):
            return metadata
        return json.dumps(metadata)
    elif isinstance(metadata, (str, int, float, bool)) or metadata is None:
        return metadata
    else:
        return json.dumps(metadata)


async def upsert_vector_batch(vectors):
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        for vec in batch:
            if "metadata" in vec:
                vec["metadata"] = sanitize_metadata(vec["metadata"])
        index.upsert(vectors=batch)


async def process_file(file: UploadFile, name: str, model: str, prompt: str, is_pdf: bool):
    print(f"Processing {'PDF' if is_pdf else 'Text'} file with model: {model}")
    text = read_pdf_file(file) if is_pdf else read_text_file(file)
    chunks = chunk_text(text)
    gist_chunk_data = []

    current_id = 0
    for i, chunk in enumerate(chunks):
        print(f"\n🧩 Chunk {i + 1}/{len(chunks)}:")
        print("-" * 60)
        print(chunk)
        print("-" * 60)

        pairs = await generate_gist_pairs(chunk, model=model, prompt=prompt)

        print(f"📦 JSON Pairs returned for Chunk {i + 1}:")
        print(json.dumps(pairs, indent=2))  # Pretty-print the JSON response

        for pair in pairs:
            gist = pair.get("gist", "")
            metadata = pair.get("metadata", {})
            if gist:
                gist_chunk_data.append({
                    "id": f"{name}_{current_id}",
                    "name": name,
                    "gist": gist,
                    "chunk": chunk,
                    "chunk_index": str(current_id),
                    "metadata": metadata
                })
                current_id += 1

    # Save output to local JSON
    json_filename = f"{name}_qa.json"
    json_path = f"./docs/{json_filename}"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(gist_chunk_data, jf, indent=2)

    # Create and upsert embeddings
    vectors = []
    for item in gist_chunk_data:
        embedding = await get_embedding(item["gist"])
        if embedding:
            metadata = {
                **item.get("metadata", {}),
                "chunk": item["chunk"],
                "chunk_index": item["chunk_index"],
                "name": item["name"]
            }
            vectors.append({
                "id": item["id"],
                "values": embedding,
                "metadata": metadata
            })

    await upsert_vector_batch(vectors)

    try:
        os.remove(json_path)
        print(f"🧹 Deleted temporary JSON file: {json_path}")
    except Exception as e:
        print(f"❌ Error deleting JSON file: {e}")

    return {
        "message": f"{'PDF' if is_pdf else 'Text'} file processed for '{name}'. Gist–chunk pairs saved and embeddings upserted.",
        "json_file": json_filename,
        "upserted_ids": [v["id"] for v in vectors],
        "max_id": len(gist_chunk_data)
    }


# API endpoints
@pdf_text_router.post("/text")
async def upload_text_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    model: str = Form(...),
    prompt: str = Form(...)
):
    return await process_file(file, name, model=model, prompt=prompt, is_pdf=False)


@pdf_text_router.post("/pdf")
async def upload_pdf_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    model: str = Form(...),
    prompt: str = Form(...)
):
    return await process_file(file, name, model=model, prompt=prompt, is_pdf=True)











@pdf_text_router.post("/json/qna/delete")
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
