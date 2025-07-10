from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings


from app.api.user_chat.chat_rag_v2 import single_product_qa_router
from app.api.single_product.analytics import single_product_router_analytics
from app.api.scrap.scrap_product import single_product_scraop_router


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#USER CHAT ROUTE
app.include_router(single_product_qa_router)
app.include_router(single_product_router_analytics)
app.include_router(single_product_scraop_router)


@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def read_root():
    return {}