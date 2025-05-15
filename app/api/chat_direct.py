from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
from openai import OpenAI
from app.core.config import settings

chat_direct_router = APIRouter(prefix="/testing", tags=["testing"])

class ChatRequest(BaseModel):
    message: str

prompt_message = (
    "In brief, you are a helpful and concise chatbot designed to assist student of IIT ISM Dhanbad. Answer user questions clearly and efficiently."
)

@chat_direct_router.post("/")
async def direct_chat(request: ChatRequest):
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_message},
                {"role": "user", "content": "about iit dhanbad"}
            ],
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
