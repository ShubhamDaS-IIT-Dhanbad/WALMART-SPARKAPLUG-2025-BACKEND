



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import requests
import json
from app.service.scrap import scrape_product
from app.core.config import settings

api_key = settings.SCRAPY_DOG_API_KEY

single_product_scraop_router = APIRouter(
    prefix="/scrap",
    tags=["scrap"]
)


class ProductURL(BaseModel):
    url: str

@single_product_scraop_router.post("")
def scrape(data: ProductURL):
    return scrape_product(data)