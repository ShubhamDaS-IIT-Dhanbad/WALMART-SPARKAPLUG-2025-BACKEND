from openai import OpenAI
from app.core.config import settings
from .pinecone_service import query_pinecone

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

    context_raw = query_pinecone(embedding, top_k=3)
    messages = [
        {"role": "system", "content": PROMPT_MESSAGE},
        {"role": "user", "content": f"Context: {context_raw}\n\nQuestion: {user_message}"}
    ]
    print(context_raw)
    # print("\n=== Final Prompt Sent to OpenAI ===", context_text, messages)

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content
