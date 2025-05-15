from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from app.core.config import settings

chat_direct_router = APIRouter(prefix="/testing", tags=["testing"])

class ChatRequest(BaseModel):
    message: str

prompt_message = (
    "In brief, you are a helpful and concise chatbot designed to assist students of IIT ISM Dhanbad. "
    "Answer user questions clearly and efficiently."
)

@chat_direct_router.post("/")
async def direct_chat(request: ChatRequest):
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4o (fast, cheap, latest)
            messages=[
                {"role": "system", "content": prompt_message},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
