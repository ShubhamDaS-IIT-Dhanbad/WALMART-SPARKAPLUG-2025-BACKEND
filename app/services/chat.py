from openai import OpenAI
from app.core.config import settings
from .pinecone_service import query_pinecone

client = OpenAI(api_key=settings.OPENAI_API_KEY)

PROMPT_MESSAGE = (
    "You are a chatbot for IIT (ISM) Dhanbad. Only respond to questions specifically related to IIT Dhanbad, ISM Dhanbad, IIT (ISM) Dhanbad, or ISM. "
    "This includes topics like scholarships, hostels, academics, campus life, admissions, and official procedures. "
    "If the question is unrelated, reply: 'I can only answer questions about IIT (ISM) Dhanbad.' Be brief and accurate."
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
