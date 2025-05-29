from openai import OpenAI
from app.core.config import settings
from .pinecone_service import query_pinecone

client = OpenAI(api_key=settings.OPENAI_API_KEY)

PROMPT_MESSAGE = (
    "You are a helpful and concise chatbot for IIT (ISM) Dhanbad. Always interpret questions in the context of IIT (ISM) Dhanbad, ISM Dhanbad, or IIT Dhanbad. "
    "Provide accurate answers related to academics, hostels, scholarships, admissions, campus life, official procedures, or any relevant topic. "
    "If a question seems ambiguous or unrelated, do your best to reinterpret it within the IIT (ISM) Dhanbad context before responding."
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

    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content
