# arasaac_utils.py

import requests
from nltk.tokenize import word_tokenize

def get_arasaac_image_url(keyword):
    #request_url = "https://api.arasaac.org/api/pictograms/all/en/search/{keyword}"
    request_url = f"https://api.arasaac.org/api/pictograms/en/search/{keyword}"

    try:
        response = requests.get(request_url)
        if response.status_code == 200:
            pictograms = response.json()
            for pic in pictograms:
                for k in pic.get("keywords", []):
                    if k.get("keyword") == keyword:
                        pic_id = pic.get("_id")
                        return f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_2500.png"
    except Exception as e:
        print(f"Error fetching image: {e}")
    return None

def tokenize_text(text):
    return word_tokenize(text.lower())
