import requests
import json
import os

api_key = "686e33f1d1c5e17f0fe89d0f"
product_id = "1816067961"
product_json_file = "products.json"




if os.path.exists(product_json_file):
    with open(product_json_file, "r") as f:
        try:
            products_data = json.load(f)
        except json.JSONDecodeError:
            products_data = {}
else:
    products_data = {}



if product_id in products_data:
    print(f"✅ Product ID {product_id} already exists in local file.")
    product_info = products_data[product_id]
else:
    print(f"📡 Fetching product {product_id} from ScrapingDog API...")
    url = "https://api.scrapingdog.com/walmart/product"
    params = {
        "api_key": api_key,
        "url": "https://www.walmart.com/ip/VEAT00L-P91-Wireless-Earbuds-Bluetooth-Headphones-V5-4-Stereo-Ear-buds-Noise-Cancelling-Mics-60H-Playback-Mini-Case-Dual-LED-Display-IP7-Waterproof-i/1816067961?classType=VARIANT&athbdg=L1800&from=/search "

    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        product_info = response.json()

        # 🔻 Exclude unwanted fields
        for field in ["variant_swatches", "available_selections"]:
            product_info.pop(field, None)

        products_data[product_id] = product_info

        # Save to JSON file
        with open(product_json_file, "w") as f:
            json.dump(products_data, f, indent=2)

        print(f"✅ Product {product_id} saved to {product_json_file}")
    else:
        print(f"❌ Failed to fetch product. Status code: {response.status_code}")
        product_info = None

# Use the product_info for further processing
if product_info:
    print(json.dumps(product_info, indent=2))