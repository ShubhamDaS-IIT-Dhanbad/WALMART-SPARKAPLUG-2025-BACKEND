from pinecone import Pinecone
from app.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("iit-ism-chat-bot-1536")

def query_pinecone(embedding: list[float], top_k: int = 1) -> str:
    result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    if result and result["matches"]:
        retrieved_text = result["matches"][0]["metadata"]
        return retrieved_text
    print("No matches found.")
    return ""
