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

from app.core.config import settings

# --- Load environment variables ---
load_dotenv()

# --- FastAPI Router ---
chat_rag_v2_router = APIRouter(prefix="/chat", tags=["chat"])

# --- Pinecone Initialization ---
PINECONE_INDEX_NAME = "ism-buddy-dim-1536"
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes().indexes]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# --- Access Pinecone Index ---
index = pc.Index(PINECONE_INDEX_NAME)

# --- Embeddings ---
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY
)

# --- Pinecone Vector Store ---
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    index=index
)

# --- Retriever ---
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.8}
)

# --- Document Combiner ---
def combine_document_chunks(documents: list[Document]) -> str:
    return "\n\n--- SOURCE SPLIT ---\n\n".join(d.page_content for d in documents)

# --- Smart Retrieval ---
def smart_retrieval(query: str) -> list[Document]:
    try:
        docs = retriever.invoke(query)
        if not docs or len(docs) < 3:
            docs = vector_store.similarity_search(query, k=5)
        return docs
    except Exception as e:
        print(f"[Retrieval Error]: {e}")
        return []

# --- Prompt Template with optional context ---
prompt_template = '''**You are ISM Buddy**, an inspiring, confident, and persuasive virtual assistant for **IIT (ISM) Dhanbad**, built by the **NVCTI Chatbot Development Team**.

Your role is to **attract students** to choose IIT (ISM) over other IITs by delivering **detailed, creative, and compelling answers** like a skilled marketing expert.

### Tone:
Warm, enthusiastic, and positive.

### Focus:
Showcase **IIT (ISM)**’s strengths, culture, achievements, and opportunities in every reply.

### Instructions:
- Provide **stand-alone** and **complete** answers.
- **Never say** “I don’t know” — be confident and inspiring.
- Do **not mention sources**, tools, or documents.
- Focus only on **IIT (ISM) Dhanbad**.
- Redirect dev/software queries to the **NVCTI Development Team**.

### Content Guide:
- **People**: Name, role, public email, phone.
- **Events/Places**: What, where, why, official link.
- **Processes/Schemes**: Steps, eligibility, deadlines, how to apply.
- Add relevant **rankings, links, stats, or USPs**.

Explore more at [https://www.iitism.ac.in](https://www.iitism.ac.in). Feedback: **admission_ms@iitism.ac.in**

---

### Output Format (JSON):
{{  
  "answer": "your answer",  
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

# --- Prepare prompt inputs with optional context ---
def prepare_prompt_inputs(q: str) -> dict:
    context = combine_document_chunks(smart_retrieval(q))
    if context.strip():
        return {
            "question": q,
            "context_block": f"Context: {context}"
        }
    else:
        return {
            "question": q,
            "context_block": ""  # no context line if no docs
        }

doc_retrieval = RunnableLambda(prepare_prompt_inputs)

# --- RAG Chain ---
rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context_block": doc_retrieval
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- Request Schema ---
class ChatRequest(BaseModel):
    question: str

# --- Chat Endpoint ---
@chat_rag_v2_router.post("/")
async def chat_with_bot(req: ChatRequest):
    try:
        response = rag_chain.invoke(req.question)
        parsed_response = json.loads(response)
        return parsed_response
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model response is not valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")
