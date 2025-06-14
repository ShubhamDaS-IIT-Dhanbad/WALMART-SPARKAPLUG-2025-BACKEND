from pinecone import Pinecone
from app.core.config import settings
import Levenshtein
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()  # This loads .env into os.environ

from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index1 = pc.Index("iit-ism-chat-bot-1536")
index2 = pc.Index(host="https://iit-ism-llama-text-embed-v2-index-j9x5uds.svc.aped-4627-b74a.pinecone.io")


def levenshtein_similarity(a: str, b: str) -> float:
    return Levenshtein.ratio(a, b)


def query_index1_by_embedding(embedding: list[float], top_k: int = 5, final_k: int = 3) -> list[dict]:
    """
    Query index1 using an embedding vector and rerank using Levenshtein similarity.
    Returns metadata of top `final_k` results.
    """
    result = index1.query(vector=embedding, top_k=top_k, include_metadata=True)

    if not result or not result.get("matches"):
        print("No matches found in index1.")
        return []

    matches = result["matches"]
    for match in matches:
        metadata = match.get("metadata", {})
        text = metadata.get("text", "") or metadata.get("category", "")
        match["score"] = levenshtein_similarity(text, text)  # fallback if no query available (you can adjust)

    matches.sort(key=lambda x: x["score"], reverse=True)
    return [match["metadata"] for match in matches[:final_k]]


def query_index2_by_text(text: str, top_k: int = 5, final_k: int = 3, namespace: str = "example-namespace") -> list[Document]:
    """
    Search index2 using text and rerank using Levenshtein similarity.
    Returns top `final_k` documents as LangChain Document.
    """
    results = index2.search(
        namespace=namespace,
        query={"inputs": {"text": text}, "top_k": top_k},
        fields=["category", "text"]
    )

    hits = results.get("result", {}).get("hits", [])
    if not hits:
        print("No matches found in index2.")
        return []

    documents = []
    for hit in hits:
        fields = hit.get("fields", {})
        content = fields.get("text") or fields.get("category", "")
        similarity = levenshtein_similarity(text, content)
        documents.append((Document(page_content=content, metadata=fields), similarity))

    # Rerank by similarity score
    documents.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in documents[:final_k]]
