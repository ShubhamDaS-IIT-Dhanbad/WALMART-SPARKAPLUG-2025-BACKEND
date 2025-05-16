from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from app.core.config import settings

chat_direct_router = APIRouter(prefix="/testing", tags=["testing"])

class ChatRequest(BaseModel):
    message: str

# Create OpenAI and Pinecone clients
client = OpenAI(api_key=settings.OPENAI_API_KEY)
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("iit-ism-chat-bot-1536")

prompt_message = (
    "In brief, you are a helpful and concise chatbot designed to assist students of IIT ISM Dhanbad. "
    "Answer user questions clearly and efficiently."
)

@chat_direct_router.post("/")
async def direct_chat(request: ChatRequest):
    try:
        print("Received request message:", request.message)

        # Step 1: Embed the user message
        print("Generating embedding...")
        embed_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=request.message
        )
        embedding = embed_response.data[0].embedding
        print("Embedding generated. Length:", len(embedding))

        # Step 2: Query Pinecone
        print("Querying Pinecone index...")
        pinecone_results = index.query(vector=embedding, top_k=1, include_metadata=True)
        print("Pinecone query result:", pinecone_results)

        retrieved_info = pinecone_results['matches'][0]['metadata'].get('text', '') if pinecone_results['matches'] else ''
        print("Retrieved context from Pinecone:", retrieved_info)

        # Step 3: Construct prompt
        messages = [
            {"role": "system", "content": prompt_message},
            {"role": "user", "content": f"Context: {retrieved_info}\n\nQuestion: {request.message}"}
        ]
        print("Messages constructed for chat:", messages)

        # Step 4: Generate ChatGPT response
        print("Sending messages to OpenAI ChatGPT...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        print("OpenAI response received.")

        return {"response": response.choices[0].message.content}

    except Exception as e:
        print("Error occurred:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
