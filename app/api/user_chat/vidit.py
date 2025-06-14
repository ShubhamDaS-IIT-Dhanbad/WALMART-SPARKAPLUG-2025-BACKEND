from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain.vectorstores import Pinecone as LangChainPinecone
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings  # Ensure settings is correctly imported and configured

# Define router
vidit_chat_router = APIRouter(prefix="/upload", tags=["post"])

# Pinecone setup
PINECONE_API_KEY = settings.PINECONE_API_KEY
PINECONE_INDEX_NAME = "ism-buddy-dim-1536"
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Embedding model (you must define this)
from langchain.embeddings import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings()

# Load vector store
vector_store = LangChainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model
)

# Placeholder: Replace with actual chunked documents
chunks: list[Document] = []  # Fill this from your data loader

# BM25 retriever setup
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

# Combine chunks
def combine_document_chunks(documents: list[Document]) -> str:
    return "\n\n--- SOURCE SPLIT ---\n\n".join(d.page_content for d in documents)

# Smart retrieval
def smart_retrieval(query: str):
    docs = retriever.invoke(query)
    return docs if docs else vector_store.similarity_search(query)

# Prompt
prompt_template = '''**You are ISM Buddy**, a sharp, enthusiastic, and persuasive virtual assistant for **IIT (ISM) Dhanbad**, built by the **NVCTI Chatbot Development Team**.

Your mission is to **attract students** to choose IIT (ISM) over other IITs by giving **compelling, creative, and complete answers**—as a skilled marketing professional would.

### Tone:
Warm, confident, inspiring, and always positive.

### Goal:
Highlight **IIT (ISM)’s** strengths, uniqueness, culture, and opportunities in every reply.

---

### Key Instructions:
- Always give detailed, stand-alone answers in your **first response**.
- **Never say** “I don’t know” — **infer and inspire**.
- Never mention documents, retrieval sources, or backend tools.
- Always keep the focus on **IIT (ISM) Dhanbad only**.
- Redirect software or dev-related queries to the **NVCTI Development Team**.

---

### Content Guidelines:
- **People**: Name, position, public email, and contact number.
- **Places / Events**: Description, location, purpose, and official link.
- **Procedures / Schemes**: Clear steps, eligibility, deadlines, and where to apply.
- Always add relevant **official links, dates, achievements, rankings**, or **unique selling points** to strengthen your response.

---

Encourage users to explore more at:  [https://www.iitism.ac.in](https://www.iitism.ac.in); For feedback: **admission_ms@iitism.ac.in**

---

### Response Format:

Question: {question}
Context: {context}
Answer:'''

prompt = ChatPromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Define chain
doc_retrieval = RunnableLambda(lambda q: {"context": combine_document_chunks(smart_retrieval(q)), "question": q})

rag_chain = (
    {
        "context": doc_retrieval,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Input schema
class ChatRequest(BaseModel):
    question: str

# Route
@vidit_chat_router.post("/chat")
async def chat_with_bot(req: ChatRequest):
    try:
        response = rag_chain.invoke(req.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")
