import json
import os
import re
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, Form
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
import pdfplumber  # for PDF text extraction

from app.core.config import settings

openai = OpenAI(api_key=settings.OPENAI_API_KEY)
pinecone = Pinecone(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)
index = pinecone.Index(settings.PINECONE_INDEX_NAME)

upload_router = APIRouter(prefix="/upload", tags=["upload"])

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

def chunk_text(text: str, max_tokens: int = 2000):
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

async def generate_gist_pairs(chunk: str):
    prompt = f"""
Extract relevant information from the text below for use in a Retrieval-Augmented Generation (RAG) system with Pinecone.

Instructions:
1. Analyze the text carefully and split it into logical chunks if it contains multiple distinct sections.
2. For each chunk, provide the following:
   - "chunk": The exact original text chunk.
   - "gist": A concise and precise summary that captures the main context and key points of the chunk, suitable for generating vector embeddings.
   - "metadata": A JSON object with structured information derived only from the chunk. Each metadata field must be accurate and specifically reference the relevant content in the chunk. Use the following fields if applicable; omit any that do not apply:
     - scholarship_name
     - eligibility
     - benefits
     - deadline
     - application_process
     - number_of_scholarships
     - provider
     - category (e.g., merit-based, need-based)
     - location
     - any other key fact relevant to the chunk

Return ONLY a valid JSON array with objects in the exact format below:

[
  {{
    "chunk": "...",
    "gist": "...",
    "metadata": {{
      "scholarship_name": "...",
      "eligibility": "...",
      "benefits": "...",
      "deadline": "...",
      "application_process": "...",
      "number_of_scholarships": "...",
      "provider": "...",
      "category": "...",
      "location": "...",
      "other_key_fact": "..."
    }}
  }},
  ...
]

If a field is not present in the chunk, do not include that field in "metadata".

Text:
\"\"\"{chunk}\"\"\"
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
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
    """
    Recursively sanitize metadata so that all values are
    either primitives or JSON strings for complex objects.
    Pinecone metadata must be str, number, bool, or list of strings.
    """
    if isinstance(metadata, dict):
        clean_meta = {}
        for k, v in metadata.items():
            clean_meta[k] = sanitize_metadata(v)
        return clean_meta
    elif isinstance(metadata, list):
        # If list of strings, keep as is
        if all(isinstance(i, str) for i in metadata):
            return metadata
        else:
            # Serialize any other list to JSON string
            return json.dumps(metadata)
    elif isinstance(metadata, (str, int, float, bool)) or metadata is None:
        return metadata
    else:
        # For any other type (e.g., dict), convert to JSON string
        return json.dumps(metadata)

async def upsert_vector_batch(vectors):
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        # Sanitize metadata before upsert
        for vec in batch:
            if "metadata" in vec:
                vec["metadata"] = sanitize_metadata(vec["metadata"])
        index.upsert(vectors=batch)

@upload_router.post("/text")
async def upload_text_file(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    text = read_text_file(file)
    chunks = chunk_text(text)
    gist_chunk_data = []

    for idx, chunk in enumerate(chunks):
        pairs = await generate_gist_pairs(chunk)
        for pair_idx, pair in enumerate(pairs):
            gist = pair.get("gist", "")
            chunk_str = pair.get("chunk", "")
            metadata = pair.get("metadata", {})
            if gist and chunk_str:
                gist_chunk_data.append({
                    "id": f"{name}_{idx}_{pair_idx}",
                    "name": name,
                    "gist": gist,
                    "chunk": chunk_str,
                    "chunk_index": f"{idx}_{pair_idx}",
                    "metadata": metadata
                })

    json_filename = f"{name}_qa.json"
    json_path = f"./docs/{json_filename}"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(gist_chunk_data, jf, indent=2)

    vectors = []
    for item in gist_chunk_data:
        id_ = item["id"]
        gist = item["gist"]
        embedding = await get_embedding(gist)
        if embedding:
            metadata = {
                **item.get("metadata", {}),
                "chunk": item["chunk"],
                "chunk_index": item["chunk_index"],
                "name": item["name"]
            }
            vectors.append({
                "id": id_,
                "values": embedding,
                "metadata": metadata
            })

    await upsert_vector_batch(vectors)

    try:
        os.remove(json_path)
        print(f"Deleted temporary JSON file: {json_path}")
    except Exception as e:
        print(f"Error deleting JSON file: {e}")

    return {
        "message": f"Text file processed for '{name}'. Gist–chunk pairs saved and embeddings upserted.",
        "json_file": json_filename,
        "upserted_ids": [v["id"] for v in vectors],
        "total_chunks": len(gist_chunk_data)
    }

@upload_router.post("/pdf")
async def upload_pdf_file(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    text = read_pdf_file(file)  # Extract text from PDF using pdfplumber
    chunks = chunk_text(text)
    gist_chunk_data = []

    for idx, chunk in enumerate(chunks):
        pairs = await generate_gist_pairs(chunk)
        for pair_idx, pair in enumerate(pairs):
            gist = pair.get("gist", "")
            chunk_str = pair.get("chunk", "")
            metadata = pair.get("metadata", {})
            if gist and chunk_str:
                gist_chunk_data.append({
                    "id": f"{name}_{idx}_{pair_idx}",
                    "name": name,
                    "gist": gist,
                    "chunk": chunk_str,
                    "chunk_index": f"{idx}_{pair_idx}",
                    "metadata": metadata
                })

    json_filename = f"{name}_qa.json"
    json_path = f"./docs/{json_filename}"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(gist_chunk_data, jf, indent=2)

    vectors = []
    for item in gist_chunk_data:
        id_ = item["id"]
        gist = item["gist"]
        embedding = await get_embedding(gist)
        if embedding:
            metadata = {
                **item.get("metadata", {}),
                "chunk": item["chunk"],
                "chunk_index": item["chunk_index"],
                "name": item["name"]
            }
            vectors.append({
                "id": id_,
                "values": embedding,
                "metadata": metadata
            })

    await upsert_vector_batch(vectors)

    try:
        os.remove(json_path)
        print(f"Deleted temporary JSON file: {json_path}")
    except Exception as e:
        print(f"Error deleting JSON file: {e}")

    return {
        "message": f"PDF file processed for '{name}'. Gist–chunk pairs saved and embeddings upserted.",
        "json_file": json_filename,
        "upserted_ids": [v["id"] for v in vectors],
        "total_chunks": len(gist_chunk_data)
    }
