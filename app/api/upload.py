from fastapi import APIRouter, UploadFile, File, Form
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
import json
import os
import re
from app.core.config import settings

# Initialize OpenAI and Pinecone clients
openai = OpenAI(api_key=settings.OPENAI_API_KEY)
pinecone = Pinecone(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)
index = pinecone.Index(settings.PINECONE_INDEX_NAME)

upload_router = APIRouter(prefix="/upload", tags=["upload"])


def clean_gpt_response(text: str) -> str:
    text = re.sub(r"^```json?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def extract_text_from_pdf(file: UploadFile) -> str:
    text = ""
    try:
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        file.file.seek(0)  # Reset pointer to the start before reading bytes again
        images = convert_from_bytes(file.file.read())
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
    return text



def chunk_text(text: str, max_tokens: int = 500):
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
    prompt = f"""Generate text chunks of relevant information. I will use it in Pinecone for implementing RAG.

{chunk}

Return ONLY a JSON array with "chunk" and a "gist" that summarizes it.
Example:
[
  {{ "gist": "...", "chunk": "..." }},
  ...
]
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        print(response)
        content = response.choices[0].message.content
        print(content)
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


async def upsert_vector_batch(vectors):
    # Pinecone supports batch upsert up to 100 vectors at a time
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)


@upload_router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    # Extract text from PDF
    text = extract_text_from_pdf(file)
    print(text)
   
    # Split text into chunks of max_tokens tokens
    chunks = chunk_text(text)

    # Step 1: Generate gist-chunk pairs for each chunk
    gist_chunk_data = []
    for idx, chunk in enumerate(chunks):
        pairs = await generate_gist_pairs(chunk)
        for index, pair in enumerate(pairs):
            gist = pair.get("gist", "")
            chunk_str = pair.get("chunk", "")
            if gist and chunk_str:
                gist_chunk_data.append({
                    "id": f"{name}_{index}",
                    "name": name,
                    "gist": gist,
                    "chunk": chunk_str,
                    "chunk_index": f"{idx}_{index}"
                })


    # Save gist-chunk data to JSON file
    json_filename = f"{name}_qa.json"
    json_path = f"./docs/{json_filename}"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(gist_chunk_data, jf, indent=2)

    # Step 2: Load JSON and prepare vectors for batch upsert
    vectors = []
    for item in gist_chunk_data:
        id_ = item["id"]
        gist = item["gist"]
        chunk = item["chunk"]
        embedding = await get_embedding(gist)
        if embedding:
            vectors.append({
                "id": id_,
                "values": embedding,
                "metadata": {
                    "chunk": chunk
                }
            })

    # Batch upsert vectors to Pinecone index
    await upsert_vector_batch(vectors)

    # Delete JSON file to free space
    try:
        os.remove(json_path)
        print(f"Deleted temporary JSON file: {json_path}")
    except Exception as e:
        print(f"Error deleting JSON file: {e}")


    return {
        "message": f"PDF processed for '{name}'. Gist–chunk pairs saved and embeddings upserted.",
        "json_file": json_filename,
        "upserted_ids": [v["id"] for v in vectors]
    }
