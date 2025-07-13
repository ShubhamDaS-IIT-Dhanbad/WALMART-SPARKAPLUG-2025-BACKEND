from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import requests
import json
from app.core.config import settings

api_key = settings.SCRAPY_DOG_API_KEY

single_product_router_analytics = APIRouter(
    prefix="/singleproduct/analytics",
    tags=["singleproductanalytics"]
)















class ProductRequest(BaseModel):
    product_id: str
    product_url: str

def write_markdown_summary(file_path: str, product_data: dict):
    try:
        product = product_data.get("product_results", {})
        reviews_data = product_data.get("reviews_results", {})
        print(reviews_data)
        title = product.get("title", "N/A")
        short_desc = product.get("short_description", "N/A")
        detailed_html = product.get("detailed_description_html", "")
        
        
        rating = reviews_data.get("reviews", {}).get("rating", "N/A")
        total_reviews = reviews_data.get("reviews", {}).get("count", "N/A")
        price_map = product.get("price_map", [])
        price = price_map[0] if isinstance(price_map, list) and price_map else "N/A"

        # Features
        specs = product.get("specifications", [])
        features = []
        for spec in specs:
            if spec.get("name", "").lower() == "features":
                features += spec.get("value", "").split(",")
            elif spec.get("name", "").lower() in ["battery life", "wireless technology"]:
                features.append(f"{spec['name']}: {spec['value']}")

        # Ratings breakdown
        rating_lines = []
        for r in sorted(reviews_data.get("ratings", []), key=lambda x: x["starts"], reverse=True):
            stars = r.get("starts", 0)
            count = r.get("count", 0)
            rating_lines.append(f"- {stars}★: {count} ratings")


        # Top positive/negative reviews
        top_pos = reviews_data.get("reviews", {}).get("top_positive", {})
        top_neg = reviews_data.get("reviews", {}).get("top_negative", {})

        # Top mentions
        mentions = reviews_data.get("reviews", {}).get("top_mentions", [])
        mention_lines = [f"- {m['name']}: {m['count']} mentions" for m in mentions[:10]]

        # Customer reviews (up to 5)
        customer_reviews = reviews_data.get("customer_reviews", [])[:5]
        customer_lines = []
        for review in customer_reviews:
            review_title = review.get("title") or "No Title"
            text = review.get("text", "")
            review_rating = review.get("rating", "")
            user = review.get("user_nickname", "anonymous")
            date = review.get("review_submission_time", "")
            customer_lines.append(
                f"**{review_title}** (⭐ {review_rating} by {user} on {date})\n{text}\n"
            )

        # Markdown Content
        markdown = f"""# {title}

**Short Description:**  
{short_desc}

**Detailed Description (HTML):**  
{detailed_html}

## 💰 Price
- ${price}

## ⭐ Overall Rating
- {rating} stars from {total_reviews} reviews

## 📊 Ratings Breakdown
{chr(10).join(rating_lines)}

## 🧩 Key Features
{chr(10).join(f"- {f.strip()}" for f in features if f.strip())}

## ✅ Top Positive Review
**{top_pos.get('title', 'No Title')}** (⭐ {top_pos.get('rating')} by {top_pos.get('user_nickname')})  
{top_pos.get('text', '')}

## ❌ Top Negative Review
**{top_neg.get('title', 'No Title')}** (⭐ {top_neg.get('rating')} by {top_neg.get('user_nickname')})  
{top_neg.get('text', '')}

## 🔍 Most Mentioned Keywords
{chr(10).join(mention_lines)}

## 🗣️ Top 5 Customer Reviews
{chr(10).join(customer_lines)}
"""

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    except Exception as e:
        print(f"⚠️ Error writing markdown summary: {e}")


def data_to_json_reponse(product_data: dict):
    product = product_data.get("product_results", {})
    reviews_data = product_data.get("reviews_results", {})

    title = product.get("title", "N/A")
    tp = product.get("tp", "N/A")
    tn= product.get("tn", "N/A")
    ss= product.get("ss", "N/A")
    short_desc = product.get("short_description", "N/A")
    specs = product.get("specifications", [])
    product_page_url = product.get("product_page_url", "N/A")
    price_map = product.get("price_map", {})
    min_quantity = product.get("min_quantity", "N/A")
    max_quantity = product.get("max_quantity", "N/A")
    images = product.get("images", [])

    total_reviews = product.get("reviews", "N/A")
    avg_rating = product.get("rating", "N/A")

    ratings = reviews_data.get("ratings", [])
    top_positive = reviews_data.get("reviews").get("top_positive", {})
    top_negative = reviews_data.get("reviews").get("top_negative", {})
    top_mentions = reviews_data.get("reviews").get("top_mentions", [])
    customer_reviews = reviews_data.get("reviews").get("customer_reviews", [])

    return {
            "title": title,
            "short_description": short_desc,
            "product_page_url": product_page_url,
            "images": images,
            "specifications": specs,
            "price_map": price_map,
            "min_quantity": min_quantity,
            "max_quantity": max_quantity,
            "tp":tp,
            "tn":tn,
            "ss":ss,
       
            "reviews_count": total_reviews,
            "average_rating": avg_rating,
            "ratings_distribution": ratings,
            "top_positive_review": top_positive,
            "top_negative_review": top_negative,
            "top_mentions": top_mentions,
            "customer_reviews": customer_reviews
        }
    


@single_product_router_analytics.post("/")
def get_single_product_analytics(data: ProductRequest):
    product_id = data.product_id
    product_url = data.product_url

    data_dir = "app/product_data"
    os.makedirs(data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, f"{product_id}.json")
    md_path = os.path.join(data_dir, f"{product_id}.md")

    # If already cached
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                product_info = json.load(f)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Corrupted product file.")

        if not os.path.exists(md_path):
            write_markdown_summary(md_path, product_info)

        product_info=data_to_json_reponse(product_info)
        return {"cached": True, "data": product_info}

    # Else fetch from ScrapingDog
    url = "https://api.scrapingdog.com/walmart/product"
    params = {
        "api_key": api_key,
        "url": product_url
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch product from API.")

    product_info = response.json()

    # Clean up unused fields
    for field in ["variant_swatches", "available_selections"]:
        product_info.pop(field, None)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(product_info, f, indent=2)

    # Save Markdown
    write_markdown_summary(md_path, product_info)


    product_info=data_to_json_reponse(product_info)
    return {"cached": False, "data": product_info}
