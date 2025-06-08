from fastapi import UploadFile, File, Form, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import os
import shutil
import json
import re
import uuid
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import google.generativeai as genai
from starlette.concurrency import run_in_threadpool
import tiktoken
import textwrap
from pinecone import Pinecone

from app.core.config import settings  # Load your .env settings here

gemini_upload_router = APIRouter(prefix="/upload", tags=["Upload & Gemini"])

# Load API Keys and configure clients
gemini_api_key = settings.GEMINI_API_KEY
if not gemini_api_key:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=gemini_api_key)

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(host=settings.PINECONE_INDEX_URI)


# ===== Utility Functions =====

def extract_text_with_pdfplumber(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"⚠️ PDFPlumber failed: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    text = extract_text_with_pdfplumber(pdf_path)
    if text and len(text.strip()) >= 50:
        print("✅ Extracted text via PDFPlumber")
        return text

    print("⚠️ Falling back to OCR...")
    images = convert_from_path(pdf_path)
    ocr_text = "\n\n".join(pytesseract.image_to_string(img) for img in images)
    return ocr_text.strip()

def count_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))

def parse_and_repair_json(text: str):
    if "```json" in text:
        text = text.split("```json", 1)[-1].rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r',\s*([\}\]])', r'\1', text)
        try:
            return json.loads(repaired)
        except:
            return None

# ===== Gemini AI JSON creation from markdown =====

def create_json_from_markdown(md_text: str) -> List[Dict]:
    system_prompt = textwrap.dedent("""
        You are a specialist information architect for IIT (ISM) Dhanbad. Your mission is to transform raw markdown into an enriched JSON knowledge base. Your tone must be professional, factual, and concise.

        **CORE DIRECTIVES:**

        1.  **STRUCTURAL DECOMPOSITION:**
            *   For documents about a single department, you **MUST** break the information into logical sections (e.g., "Department Overview", "Faculty", "Academic Programs", "Facilities", "Contact Information"). Each section must be a separate JSON object.
            *   For documents that are simple lists (like "Heads of Department" or "Associate Deans"), you **MUST** create only **ONE** JSON object that contains all the information.

        2.  **CONTENT & TONE:**
            *   Enrich the content by explaining its institutional context, but remain factual and concise. Avoid exaggeration.
            *   For lists of people, the 'category' description **MUST** be a series of complete sentences. Each sentence should detail one person and their contact information, ending with a period.

        3.  **JSON OUTPUT:**
            *   Provide your response ONLY as a valid JSON array inside a single ```json block. Do not include any other text or explanations.
            *   Each object should have two keys: "text" (a concise heading for the section) and "category" (the detailed, enriched description).
    """)

    examples = textwrap.dedent("""
        ---
        **EXAMPLE 1: ANY DEPARTMENT PAGE OR ANY OTHER LIST OR MARKDOWN TEXT CONTAINING MULTIPLE HEADINGS, SUB-HEADINGS, BODIES OR A MARKDOWN TEXT IN TABULAR FORMAT WITH DATES, TIME, AND ACTIVITIES (MUST BE MULTIPLE OBJECTS)**
        INPUT: "## Department of Applied Geophysics..."
        OUTPUT:
        ```json
        [
          {
            "text": "Department of Applied Geophysics Overview",
            "category": "The Department of Applied Geophysics at IIT (ISM) Dhanbad, established in 1957, is a leading institution in India for geophysical education and research. It offers a range of academic programs and contributes significantly to the nation's mineral and energy sectors. Its official website is https://www.iitism.ac.in/applied-geophysics-home and it is located at https://maps.app.goo.gl/UChPJr6TjedRnHsf6."
          },
          {
            "text": "Faculty",
            "category": "The department boasts a distinguished faculty with expertise in various geophysical disciplines. A complete list of faculty members and their research interests is available in the official faculty directory: https://www.iitism.ac.in/applied-geophysics-faculty."
          },
          {
            "text": "Facilities",
            "category": "The department is equipped with state-of-the-art laboratories to support cutting-edge research, including a Seismological Observatory, Geophysical Inversion Lab, and a Rock & Petrophysics Lab."
          },
          {
            "text": "Contact Information",
            "category": "The department office can be reached at agp@iitism.ac.in or +91-326-223-5272. The current Head of Department is Prof. Sanjit Kumar Pal."
          }
        ]
        ```
        ---
        **EXAMPLE 2: A LIST OF PEOPLE/SHOPS/FACILITIES AVAILABLE (MUST BE ONE OBJECT)**
        INPUT: "## List of Associate Deans..."
        OUTPUT:
        ```json
        [
          {
            "text": "List of Associate Deans",
            "category": "The Associate Dean (Academic, Postgraduate) is Prof. Sushrut Das contactable at adpg@iitism.ac.in or +91-326-223-5218. The Associate Dean (Academic, Undergraduate) is Dr. Vipin Kumar contactable at adug@iitism.ac.in or +91-326-223-5230. The Associate Dean (IT & Networking and Infrastructure) is Prof. Badam Singh Kushvah contactable at adin@iitism.ac.in or +91-326-223-5286. The official website is https://www.iitism.ac.in/associate-deans."
          }
        ]
        ```
    """)

    user_prompt = f"{examples}\n\n--- \n\n**TASK:** Now, process the following markdown based on the principles and examples above.\n\n{md_text}"

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(
            [system_prompt.strip(), user_prompt.strip()],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192
            )
        )
        raw_response = response.text.strip()
        ai_data = parse_and_repair_json(raw_response)
        if not ai_data or not isinstance(ai_data, list):
            return [{"error": "Invalid AI response", "raw_output": raw_response}]
    except Exception as e:
        return [{"error": "Gemini API call failed", "details": str(e)}]

    def get_main_initials(data):
        heading = re.sub(r'[^A-Za-z\s]', '', data[0].get('text', 'Doc'))
        return ''.join(w[0].upper() for w in heading.split() if w)

    def create_text(heading, desc):
        words = re.findall(r'\b[A-Za-z-]{3,}\b', desc)
        seen = set()
        unique = [w for w in words if not (w.lower() in seen or seen.add(w.lower()))]
        return f"{heading.strip()} {' '.join(unique[:25])}".strip()

    main_initials = get_main_initials(ai_data)

    return [
        {
            "_id": f"{main_initials}{uuid.uuid4().hex[:6]}",
            "text": create_text(item["text"], item["category"]),
            "category": item["category"].strip()
        }
        for item in ai_data
        if "text" in item and "category" in item
    ]


# ====== Routes ======

@gemini_upload_router.post("/gemini/text")
async def upload_text_to_gemini(
    md_text: str = Form(...),
    doc_id: str = Form(...)
):
    """
    Accepts raw markdown text and an identifier (doc_id), converts to JSON entries,
    upserts them into Pinecone, and returns a success message.
    """
    if not md_text.strip():
        raise HTTPException(status_code=400, detail="Markdown text is empty.")
    if not doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id is required.")

    # Process markdown with Gemini AI
    data = await run_in_threadpool(create_json_from_markdown, md_text)

    records = []
    base_id = doc_id.replace(" ", "_")

    for idx, item in enumerate(data, start=1):
        text = item.get("text", "")
        category = item.get("category", "")

        if not text:
            continue

        _id = f"{base_id}_{idx}"

        records.append({
            "id": _id,
            "text": text,
            "category": category
        })

    if records:
        index.upsert_records("example-namespace", records)

    return JSONResponse(
        content={
            "status": "success",
            "message": f"✅ {len(records)} entries embedded & uploaded to Pinecone.",
            "max_id": len(data),
        }
    )


@gemini_upload_router.post("/gemini/pdf")
async def upload_pdf_to_gemini(
    filename: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Accepts a PDF file, extracts content using text or OCR, returns structured JSON,
    and upserts parsed JSON entries to Pinecone with IDs derived from filename.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)

    safe_filename = filename.replace(" ", "_")
    temp_path = os.path.join(upload_dir, safe_filename)

    # Save file temporarily
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Extract markdown text from PDF in threadpool
        markdown_text = await run_in_threadpool(extract_text_from_pdf, temp_path)

        # Process extracted markdown text with Gemini AI in threadpool
        data = await run_in_threadpool(create_json_from_markdown, markdown_text)

        records = []
        base_id = os.path.splitext(safe_filename)[0]

        for idx, item in enumerate(data, start=1):
            text = item.get("text", "")
            category = item.get("category", "")

            if not text:
                continue

            _id = f"{base_id}_{idx}"

            records.append({
                "id": _id,
                "text": text,
                "category": category
            })

        if records:
            index.upsert_records("example-namespace", records)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"✅ {len(records)} entries embedded & uploaded to Pinecone.",
                "max_id": len(data),
            }
        )
    finally:
        # Cleanup temporary PDF file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Failed to delete temp file: {e}")
