from bs4 import BeautifulSoup
import requests
import json
from fastapi import HTTPException

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive"
}


def scrape_product(data):
    try:
        print("ji")
        res = requests.get(data.url, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')
        print(soup)
        title = soup.find("h1", {"class": "prod-ProductTitle"}).text.strip()
        price_tag = soup.find("span", {"class": "price-characteristic"})
        price = price_tag['content'] if price_tag else "N/A"
        rating_tag = soup.find("span", {"itemprop": "ratingValue"})
        rating = rating_tag.text.strip() if rating_tag else "N/A"
        desc_tag = soup.find("div", {"class": "about-desc about-product-description xs-margin-top"})
        description = desc_tag.text.strip() if desc_tag else "No description available"

        product_data = {
            "title": title,
            "price": price,
            "rating": rating,
            "description": description
        }

        with open("product_data.json", "w") as f:
            json.dump(product_data, f, indent=2)

        md = f"# {title}\n\n**Price:** ${price}\n\n**Rating:** {rating}/5\n\n## Description\n{description}"
        with open("product_data.md", "w", encoding="utf-8") as f:
            f.write(md)

        return product_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
