# arasaac_utils.py

import requests
from nltk.tokenize import word_tokenize

def get_arasaac_image_url(word):
    url = f"https://api.arasaac.org/api/pictograms/en/search/{word}"
    try:
        response = requests.get(url)
        data = response.json()
        if data and isinstance(data, list):
            pic_id = data[0]["_id"]
            return f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
    except:
        pass
    return ""

def get_category_image(category):
        url = f"https://api.arasaac.org/api/pictograms/en/search/{category}"
        response = requests.get(url)

        if response.status_code == 200:
            pictograms = response.json()
            if pictograms:
                first_pic_id = pictograms[0]["_id"]
                image_url = f"https://static.arasaac.org/pictograms/{first_pic_id}/{first_pic_id}_500.png"
                return image_url
            else:
                print("No pictograms found.")
                return None
        else:
            print("API Error")
            return None

def fetch_pictograms(category):
    url = f"https://api.arasaac.org/api/pictograms/en/search/{category}"
    try:
        response = requests.get(url)
        data = response.json()
        pictos = []
        for item in data[:16]:
            label = item.get("keywords", [{}])[0].get("keyword", category)
            pic_id = item["_id"]
            img_url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
            pictos.append((label, img_url))
        return pictos
    except:
        return []

def tokenize_text(text):
    return text.lower().split()
