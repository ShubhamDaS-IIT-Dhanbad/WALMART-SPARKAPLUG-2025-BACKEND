from openai import OpenAI
from app.core.config import settings
from .pinecone_service import query_index1_by_embedding, query_index2_by_text

client = OpenAI(api_key=settings.OPENAI_API_KEY)

PROMPT_MESSAGE = (
    "You are ISM Buddy, the official IIT (ISM) Dhanbad chatbot by NVCTI. Answer all queries positively and accurately in IIT (ISM) context—academics, hostels, scholarships, admissions, campus life, or procedures. "
    "Reframe unclear or unrelated questions accordingly. Redirect code, cost, or software queries to the Development Team. Don’t reveal backend details or sources. Deny harmful content and stay polite. "
    "For feedback, share admission_ms@iitism.ac.in. For general info, guide to iitism.ac.in or people.iitism.ac.in/~research."
)

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding

def get_chat_response(user_message: str) -> str:
    embedding = get_embedding(user_message)
    context_raw = query_index1_by_embedding(embedding, top_k=3)

    if not context_raw:
        context_text = "No relevant context found."
    else:
        print(context_raw)
        context_text = "\n".join([str(item) for item in context_raw])

    messages = [
        {"role": "system", "content": PROMPT_MESSAGE},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_message}"}
    ]

    print("Context sent to model:", context_text)

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


def get_chat_response_by_text(user_message: str) -> str:
    context_raw = query_index2_by_text(user_message, top_k=1)

    if not context_raw or not context_raw.get("result", {}).get("hits"):
        context_text = "No relevant context found."
    else:
        hits = context_raw["result"]["hits"]
        context_parts = []
        for hit in hits:
            fields = hit.get("fields", {})
            if "category" in fields:
                context_parts.append(fields["category"])
            elif "text" in fields:
                context_parts.append(fields["text"])
        context_text = "\n".join(context_parts)

    messages = [
        {"role": "system", "content": PROMPT_MESSAGE},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_message}"}
    ]

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content
