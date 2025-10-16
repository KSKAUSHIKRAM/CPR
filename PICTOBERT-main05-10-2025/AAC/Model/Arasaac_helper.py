import shelve
import requests
import json
from nltk.corpus import wordnet

CACHE_DB = "cache_urls.db"
CACHE_URL_DB = "cache_urls.db"

METADATA_FILE = "E:\PICTOBERT-main05-10-2025\AAC\Model\metadata_drive.json"
# --- NEW IMPORTS ---
import sqlite3
from io import BytesIO
from kivy.core.image import Image as CoreImage
import os
# --- NEW GLOBALS ---
ICON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "View", "icons"))
os.makedirs(ICON_DIR, exist_ok=True)   # ✅ auto-create folder
print(f"[INIT] ✅ Icon cache folder: {ICON_DIR}")

CACHE_DB = os.path.join(os.path.dirname(__file__), "picto_cache.db")
MAX_CACHE_MB = 100


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
    synonyms.discard(word)
    hypernyms.discard(word)

    possible = []
    singular = singularize(word)
    if singular != word:
        possible.append(singular)
    possible += list(synonyms)
    possible += list(hypernyms)
    return possible
def ensure_cache_table():
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS PictoCache (
            label TEXT PRIMARY KEY,
            image BLOB,
            last_used REAL DEFAULT (strftime('%s','now'))
        )
    """)
    conn.commit(); conn.close()


def prune_cache(max_size_mb=MAX_CACHE_MB):
    """Auto-delete old icons if /icons folder exceeds MAX_CACHE_MB."""
    try:
        files = [
            (os.path.join(ICON_DIR, f),
             os.path.getmtime(os.path.join(ICON_DIR, f)),
             os.path.getsize(os.path.join(ICON_DIR, f)))
            for f in os.listdir(ICON_DIR) if f.endswith(".png")
        ]
        total = sum(size for _, _, size in files) / (1024 * 1024)
        if total > max_size_mb:
            files.sort(key=lambda x: x[1])  # oldest first
            while total > max_size_mb and files:
                f, _, size = files.pop(0)
                os.remove(f)
                total -= size / (1024 * 1024)
                print(f"[CACHE] Removed old file: {os.path.basename(f)}")
    except Exception as e:
        print("[CACHE ERROR]", e)


def fetch_from_cache(label):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("SELECT image FROM PictoCache WHERE label=?", (label,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def save_to_cache(label, img_bytes):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO PictoCache(label, image, last_used)
        VALUES (?, ?, strftime('%s','now'))
    """, (label, sqlite3.Binary(img_bytes)))
    conn.commit(); conn.close()


class Arasaac_helper:
    categories = [
        "food", "animals", "clothes", "emotions", "body", "sports", "school", "family",
        "nature", "transport", "weather", "home", "health", "jobs", "colors", "toys"
    ]

    @staticmethod
    def load_metadata():
        """Load metadata_drive.json and normalize keys to lowercase."""
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return {k.lower(): v for k, v in metadata.items()}
        except Exception as e:
            print(f"Metadata load failed: {e}")
            return {}

    @staticmethod
    def get_arasaac_image_url(word):
        """Fetch pictogram for a token (metadata → cache → ARASAAC)."""
        key = (word or "").strip().lower()
        if not key:
            return ""

        metadata = Arasaac_helper.load_metadata()

        # 1️⃣ Metadata check
        if key in metadata:
            entry = metadata[key]
            if isinstance(entry, dict) and "url" in entry:
                return entry["url"]
            elif isinstance(entry, str):
                return entry

        # 2️⃣ Safe cache open
        try:
            with shelve.open(CACHE_DB) as cache:
                if key in cache:
                    return cache[key]
        except Exception as e:
            print(f"[SHELVE ERROR] {e} → Recreating cache.")
            # Auto-remove corrupted cache
            import os
            try:
                os.remove(CACHE_DB)
            except:
                pass

        # 3️⃣ Fetch new from ARASAAC
        try:
            resp = requests.get(f"https://api.arasaac.org/api/pictograms/en/search/{key}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    pic_id = data[0]["_id"]
                    img_url = f"https://static.arasaac.org/pictograms/{pic_id}/{pic_id}_500.png"
                    try:
                        with shelve.open(CACHE_DB) as cache:
                            cache[key] = img_url
                    except Exception as e:
                        print(f"[CACHE WRITE ERROR] {e}")
                    return img_url
        except Exception as e:
            print(f"[ARASAAC API ERROR] {e}")

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
   
    def get_cached_image(self, label, placeholder_path):
        """
        Retrieve pictogram URL from metadata / WordNet / ARASAAC.
        Caches the URL in cache_urls.db for reuse.
        Returns (None, url) so that Kivy AsyncImage can load directly.
        No image files are downloaded or saved.
        """
        import shelve

        key = (label or "").strip().lower()
        if not key:
            return None, placeholder_path

        try:
            # --- 1️⃣ Try cache first ---
            with shelve.open("cache_urls.db") as cache:
                if key in cache:
                    img_url = cache[key]
                    print(f"[CACHE-URL] Hit: {key} → {img_url}")
                    return None, img_url

            # --- 2️⃣ Otherwise, call existing URL fetcher ---
            img_url = self.get_arasaac_image_url(key)

            if img_url:
                # --- 3️⃣ Save to cache ---
                with shelve.open("cache_urls.db") as cache:
                    cache[key] = img_url
                print(f"[CACHE-URL] Added: {key} → {img_url}")
                return None, img_url
            else:
                print(f"[CACHE-URL] No URL found for {key}, using placeholder.")
                return None, placeholder_path

        except Exception as e:
            print(f"[CACHE-URL ERROR] {key}: {e}")
            return None, placeholder_path
