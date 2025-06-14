from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.chat import get_chat_response,get_chat_response_by_text

user_chat_router = APIRouter(prefix="/testing", tags=["testing"])

class ChatRequest(BaseModel):
    message: str

@user_chat_router.post("/")
async def direct_chat(request: ChatRequest):
    try:
        return {"response": get_chat_response_by_text(request.message)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
