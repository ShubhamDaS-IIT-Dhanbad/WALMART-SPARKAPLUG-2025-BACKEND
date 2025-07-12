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
    product_id: str  # Now a comma-separated string
    query: str

@single_product_qa_router.post("/")
async def product_qa(request: ProductQARequest):
    product_ids = [pid.strip() for pid in request.product_id.split(",") if pid.strip()]
    query = request.query
    print(product_ids)

    data_dir = "app/product_data"
    os.makedirs(data_dir, exist_ok=True)

    product_docs = []

    for idx, pid in enumerate(product_ids, 1):
        md_path = os.path.join(data_dir, f"{pid}.md")
        if not os.path.exists(md_path):
            raise HTTPException(status_code=404, detail=f"Product markdown file for ID {pid} not found.")
        
        with open(md_path, "r", encoding="utf-8") as f:
            doc = f.read()
            product_docs.append(f"### Product {idx} (ID: {pid}):\n{doc}")

    combined_product_info = "\n\n".join(product_docs)

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful product assistant.
        Answer the following user query using the markdown documents provided for multiple products.

        {product_info}

        ### Query:
        {user_query}
        """
    )

    # LLM + chain
    llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "product_info": combined_product_info,
        "user_query": query
    })

    return {"response": result}
