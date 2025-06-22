from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import json

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from app.core.config import settings

# Load environment variables
load_dotenv()

chat_rag_v2_router = APIRouter(prefix="/chat", tags=["chat"])

# --- Pinecone Setup ---
PINECONE_INDEX_NAME = "ism-buddy-dim-1536"
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes().indexes]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY
)

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    index=index
)

# ⚠️ Replace with your real BM25 docs if needed
chunks = [
    Document(page_content="IIT (ISM) Dhanbad is renowned for its mining engineering department.", metadata={"question": "Why choose IIT Dhanbad?", "answer": "It is famous for its mining program."}),
]

if not chunks:
    raise ValueError("❌ 'chunks' is empty. Please provide documents for BM25Retriever.")

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5

pine_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
)

retriever = EnsembleRetriever(
    retrievers=[pine_retriever, bm25],
    weights=[0.4, 0.6]
)

# --- Document Formatter ---
def combine_document_chunks(documents: list[Document], print_retrieved_data: bool = False) -> str:
    chunks = []
    for idx, doc in enumerate(documents):
        meta = doc.metadata or {}
        text = meta.get("text") or (doc.page_content.strip() if doc.page_content else "")
        parts = [
            f"Q: {meta.get('question')}" if meta.get("question") else "",
            f"A: {meta.get('answer')}" if meta.get("answer") else "",
            text,
            f"Links: {', '.join(meta['links'])}" if isinstance(meta.get("links"), list) and meta.get("links") else ""
        ]
        formatted = "\n".join([p for p in parts if p])
        chunks.append(formatted)

        if print_retrieved_data:
            print(f"\n--- Chunk {idx + 1} ---")
            print(f"Metadata: {json.dumps(meta, indent=2)}")
            print(f"Content:\n{formatted}")

    return "\n\n--- SOURCE SPLIT ---\n\n".join(chunks)

# --- Smart Retrieval ---
def smart_retrieval(query: str):
    try:
        docs = retriever.invoke(query)
        print(docs)
        return docs
    except Exception as e:
        print(f"[Retrieval Error]: {e}")
        return []

# --- Prompt Template ---
prompt_template = '''*You are ISM Buddy, a sharp, enthusiastic, and persuasive virtual assistant for **IIT (ISM) Dhanbad, built by the **NVCTI Chatbot Development Team*.

Your mission is to *attract students* to choose IIT (ISM) over other IITs by giving *compelling, creative, and complete answers*—as a skilled marketing professional would.

### Tone:
Warm, confident, inspiring, and always positive.

### Goal:
Highlight *IIT (ISM)’s* strengths, uniqueness, culture, and opportunities in every reply as CONCLUSIVE STATEMENTS.

### Key Instructions:
- Always give detailed, stand-alone answers in your *first response* and it is must to include all the relevant links from the retreival.
- *Never say* “I don’t know” — *infer and inspire*.
- Never mention documents, retrieval sources, or backend tools.
- Always keep the focus on *IIT (ISM) Dhanbad only*.
- Redirect software or dev-related queries to the *NVCTI Development Team*.
- If the RAG system do not have the answer, search your own database and provide the answer.
- Providing relevant links for location is compulsory for each query.

### Content Guidelines:
- *People*: Name, position, public email, and contact number.
- *Places / Events*: Description, location, purpose, and official link.
- *Procedures / Schemes*: Clear steps, eligibility, deadlines, and where to apply.
- Always add relevant *official links, dates, achievements, rankings, or **unique selling points* to strengthen your response.

KEEP THE ANSWER CONSISELY DETAILED BUT CRISP.
For any updates of the official placements and internships statistics of IIT (ISM) Dhanbad, visit here: https://bit.ly/3G8yHS8 
---

### Output Format (JSON):
{{  
  "answer": "your answer must be in points add some emojies and some professional styling",  
  "follow_up_question": [  
    "question 1",  
    "question 2",  
    "question 3"  
  ]  
}}

---

Question: {question}
{context_block}
Answer:'''

prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def prepare_prompt_inputs(q: str) -> dict:
    try:
        docs = smart_retrieval(q)
        context = combine_document_chunks(docs, print_retrieved_data=True)
        print(context)
        return {
            "question": q,
            "context_block": f"Context: {context}" if context.strip() else ""
        }
    except Exception as e:
        print(f"[Prepare Input Error]: {e}")
        return {
            "question": q,
            "context_block": ""
        }

# --- RAG Chain ---
rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context_block": RunnableLambda(lambda q: prepare_prompt_inputs(q)["context_block"])
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- Request Schema ---
class ChatRequest(BaseModel):
    question: str

# --- API Route ---
@chat_rag_v2_router.post("/")
async def chat_with_bot(req: ChatRequest):
    try:
        response = rag_chain.invoke(req.question)
        parsed_response = json.loads(response)
        return parsed_response
    except json.JSONDecodeError:
        print(f"[Invalid JSON from LLM]: {response}")
        raise HTTPException(status_code=500, detail="Response not valid JSON.")
    except Exception as e:
        print(f"[Chat Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
