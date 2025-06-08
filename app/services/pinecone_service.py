from pinecone import Pinecone
from app.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index1 = pc.Index("iit-ism-chat-bot-1536")
index2 = pc.Index(host="https://iit-ism-llama-text-embed-v2-index-j9x5uds.svc.aped-4627-b74a.pinecone.io")

def query_index1_by_embedding(embedding: list[float], top_k: int = 1) -> dict:
    """
    Query index1 using an embedding vector.
    Returns the metadata of the top match or empty dict if no matches.
    """
    result = index1.query(vector=embedding, top_k=top_k, include_metadata=True)
    if result and result.get("matches"):
        return result["matches"][0]["metadata"]
    print("No matches found in index1.")
    return {}

def query_index2_by_text(text: str, top_k: int = 2, namespace: str = "example-namespace") -> dict:
    """
    Search index2 using a text query.
    Returns the search results or empty dict if no results.
    """
    results = index2.search(
        namespace=namespace,
        query={
            "inputs": {"text": text},
            "top_k": top_k
        },
        fields=["category", "text"]
    )
    if results:
        return results
    print("No matches found in index2.")
    return {}
