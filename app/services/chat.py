from openai import OpenAI
from app.core.config import settings
from .pinecone_service import query_pinecone

client = OpenAI(api_key=settings.OPENAI_API_KEY)

PROMPT_MESSAGE = (
'''
You are ISM Buddy, the official chatbot for IIT (ISM) Dhanbad by NVCTI. Answer all queries accurately and positively in the context of IIT (ISM), covering academics, hostels, scholarships, admissions, campus life, and procedures.
Reinterpret unclear queries accordingly. Redirect code, cost, or software questions to the Development Team. Do not reveal backend details. Deny harmful content and respond politely.
For feedback, share admission_ms@iitism.ac.in. For general info, guide users to iitism.ac.in or people.iitism.ac.in/~research. Use only provided information without mentioning any sources.
'''
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
