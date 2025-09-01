import shelve
import requests
from nltk.corpus import wordnet

CACHE_DB = "cache_urls.db"

def get_wordnet_alternatives(word):
    """Return [singular, synonyms, hypernyms] for a word."""
    def singularize(w):
        return w[:-1] if w.endswith('s') else w

    synonyms = set()
    hypernyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                hypernyms.add(lemma.name().replace("_", " "))
    # Remove the original word from synonyms/hypernyms to avoid duplicates
    synonyms.discard(word)
    hypernyms.discard(word)
    # Order: singular, synonyms, hypernyms
    possible = []
    singular = singularize(word)
    if singular != word:
        possible.append(singular)
    possible += list(synonyms)
    possible += list(hypernyms)
    return possible

class Arasaac_helper:
    categories = [
        "food", "animals", "clothes", "emotions", "body", "sports", "school", "family",
        "nature", "transport", "weather", "home", "health", "jobs", "colors", "toys"
    ]

    @staticmethod
    def get_arasaac_image_url(word):
        """Fetch pictogram for a token (with caching). Try WordNet alternatives if not found."""
        key = (word or "").strip().lower()
        if not key:
            return ""
        with shelve.open(CACHE_DB) as cache:
            # Try original word first
            if key in cache:
                return cache[key]
            try:
                resp = requests.get(f"https://api.arasaac.org/api/pictograms/en/search/{key}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        pic_id = data[0]["_id"]
                        img_url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
                        cache[key] = img_url
                        return img_url
                # If not found (404 or empty), try WordNet alternatives
                alternatives = get_wordnet_alternatives(key)
                for alt in alternatives:
                    alt_key = alt.strip().lower()
                    if not alt_key:
                        continue
                    if alt_key in cache:
                        return cache[alt_key]
                    try:
                        resp_alt = requests.get(f"https://api.arasaac.org/api/pictograms/en/search/{alt_key}", timeout=5)
                        if resp_alt.status_code == 200:
                            data_alt = resp_alt.json()
                            if data_alt:
                                pic_id = data_alt[0]["_id"]
                                img_url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
                                cache[alt_key] = img_url
                                return img_url
                    except Exception as e:
                        print(f"API error for '{alt_key}': {e}")
            except Exception as e:
                print(f"API error: {e}")
        return ""

    @staticmethod
    def fetch_pictograms(category):
        """Get pictograms for a category (first 16)."""
        pictos = []
        try:
            resp = requests.get(f"https://api.arasaac.org/api/pictograms/en/search/{category}", timeout=5)
            data = resp.json()
            for item in data[:16]:
                label = item.get("keywords", [{}])[0].get("keyword", category)
                pic_id = item["_id"]
                img_url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
                pictos.append((label, img_url))
        except Exception as e:
            print(f"Fetch pictograms failed: {e}")
        return pictos