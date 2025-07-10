from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI

from app.core.config import settings

single_product_qa_router = APIRouter(
    prefix="/productqa",
    tags=["Product QA"]
)

class ProductQARequest(BaseModel):
    product_id: str
    query: str

@single_product_qa_router.post("/")
async def product_qa(request: ProductQARequest):
    product_id = request.product_id
    query = request.query

    data_dir = "app/product_data"
    os.makedirs(data_dir, exist_ok=True)

    md_path = os.path.join(data_dir, f"{product_id}.md")
    
    if not os.path.exists(md_path):
        raise HTTPException(status_code=404, detail="Product markdown file not found.")

    with open(md_path, "r", encoding="utf-8") as f:
        product_doc = f.read()

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful product assistant.
        Answer the following user query using the markdown document provided.

        ### Product Info:
        {product_info}

        ### Query:
        {user_query}
        """
    )

    # Initialize model
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o-mini")

    # Chain
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "product_info": product_doc,
        "user_query": query
    })

    return {"response": result}
