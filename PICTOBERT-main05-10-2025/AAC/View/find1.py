from __future__ import annotations
import difflib
import pandas as pd

# -------------------------------
# Load domain-specific words
# -------------------------------
domain_file = r"E:\pic_test\PICTOBERT-main 11-09-2025\AAC\domain_terms_aac.tsv"
df = pd.read_csv(domain_file, sep="\t", header=None, names=["word", "freq"])
domain_words = df["word"].dropna().astype(str).str.lower().tolist()

# -------------------------------
# Domain categories
# -------------------------------
FOOD = {
    # Existing
    "idly","idli","dosa","parotta","parota","pulao","biryani","sambar","rasam",
    "chutney","rice","curd","curdrice","chapati","poori","upma","pongal","noodles",
    "egg","chicken","fish","mutton","paneer","vegetable","veg","sabzi","sadam",

    # Breakfast / Indian dishes
    "bread","cornflakes","aaloopoori","eggs","poha","khichdi","paratha","omlette",
    "meduwada","porridge","sandwich","sambhar","utappam","roti","dal","dalkhichdi",
    "raita","pork","crabmeat","turkey","pizza","salad","soup","pasta","italian",
    "pavbhaji","bhakri",

    # Sweets
    "cake","icecream","gajarhalwa","gulabjamun","laddoo","barfi","jalebi",
    "fruitsalad","rasagulla","sheera",

    # Fruits
    "apple","banana","grapes","guava","mango","orangefruit","pineapple",
    "strawberry","blueberry","pomegranate","watermelon","pear","papaya",
    "muskmelon","chikoo","jackfruit","cherry"
}

DRINK = {
    "water", "milk", "bournvita", 
    "mango_juice", "apple_juice", "orange_juice", "lemon_juice", "pineapple_juice",
    "pepsi", "cocacola", "mirinda", "fanta", "maaza", "sprite", "mountain_dew",
    "milkshake", "chocolate_milkshake", "strawberry_milkshake", "banana_milkshake", 
    "mango_milkshake", "chikoo_milkshake",
    "tea", "coffee", "cold_coffee",
    "energy_drink", "lassi", "buttermilk", "chaas", 
    "coconut_water", "tender_coconut_water",
    "soda", "nimbu_pani", "badam_milk", "rose_milk"
}
PLACE = {
    "park","school","toilet","bathroom","restroom","hospital","clinic","temple","church","mosque",
    "market","shop","office","home","house","kitchen","station","bank","atm","pharmacy","playground",
    "bus stop","stop","outside","garden","beach","restaurant",
    "myhouse","mall","museum","restaurant","theatre","playground","park",
    "shop_india","friendshouse","relativeshouse","library","worship","zoo",
    "livingroom","bedroom","studyroom","playroom","balcony"
}
GREETINGS = {
    "hi", "hello", "bye", "good morning", "good afternoon", "good evening", "good night",
    "hi five", "nice to meetyou", "how are you", "how was your day", "how do you do",
    "grea tjob", "awesome", "congratulations", "well done",
    "see you later", "take care", "have a niceday", "pleasure to meet you","welcome"
}
POSITIVE_FEELINGS = {'happy', 'amazed'}
NEGATIVE_FEELINGS = {
    'sad', 'angry', 'afraid', 'irritated', 'confused', 'ashamed',
    'disappointed', 'bored', 'worried', 'stressed', 'tired', 
    'hot', 'cold', 'sick', 'hurt'
}
FAMILY = {'mother','father','brother','sister','grandfather','grandmother','uncle','aunt','cousin'}
PROFESSIONALS = {'teacher', 'doctor', 'nurse', 'therapist', 'caregiver', 'stranger'}
ABOUTME = {'aboutme'}

REQUESTS = {
    'please', 'thankyou', 'youarewelcome', 'pleasegiveme', 'pleasetellmeagain', 'pleaseshowme',
    'ineedabreak', 'iamalldone', 'excuseme', 'iamsorry', 'idontunderstand', 'pleaseshare',
    'pleaseslowdown', 'ineedhelp', 'pleasecomehere', 'pleasetakeme', 'ineedmoretime',
    'bealone', 'quietplease'
}
FUNCTION_WORDS= {
    "i","to","the","a","please","me","can","have","is","am","are","on","in","at","for",
    "of","and","or","it","you","he","she","they","we","this","that","these","those",
    "with","my","your","his","her","their","our","want","need","go","give","get",
    "drink","eat","like","let","let's"
}

# Help sentence templates
HELP_SENTENCES = [
    "Please help me with {obj}.",
    "I need help with {obj}.",
    "Can you help me with {obj}?",
    "I want help with {obj}.",
    "Help me, please with {obj}.",
    "Can I get some assistance with {obj}?"
]

# Combined lexicon
GENERAL_VOCAB = {"good", "morning", "night", "thank", "you", "please", "help", "how", "are"}
LEXICON = set(domain_words) | FOOD | DRINK | PLACE | GENERAL_VOCAB | FAMILY | POSITIVE_FEELINGS | NEGATIVE_FEELINGS | REQUESTS | GREETINGS | PROFESSIONALS | FUNCTION_WORDS

# -------------------------------
# Input Normalizer Class
# -------------------------------
class InputNormalizer:
    def __init__(self):
        self.lexicon = LEXICON

    def _infer_tag(self, word):
        if word in FOOD: return "food"
        if word in DRINK: return "drink"
        if word in PLACE: return "place"
        if word in POSITIVE_FEELINGS: return "positive_feeling"
        if word in NEGATIVE_FEELINGS: return "negative_feeling"
        if word in REQUESTS: return "request"
        if word in FAMILY: return "family"
        if word in PROFESSIONALS: return "professional"
        if word in ABOUTME: return "aboutme"
        if word in GREETINGS: return "greeting"
        return None

    def correct_word(self, word: str) -> str:
        word = word.lower()
        if word in FUNCTION_WORDS: return word
        if len(word) == 1: return word
        if word in self.lexicon: return word
        matches = difflib.get_close_matches(word, self.lexicon, n=3, cutoff=0.5)
        return matches[0] if matches else word

    def generate_hint_sentences(self, word: str):
        corrected_word = self.correct_word(word)
        return [s.format(obj=corrected_word) for s in HELP_SENTENCES]

    def _category_expansions(self, tokens):
        results = []
        for token in tokens:
            if token in FUNCTION_WORDS: continue
            token = self.correct_word(token)
            tag = self._infer_tag(token)
            if tag == "food":
                results.extend([f"I want to eat {token}", f"Can I have {token}", f"Please give me {token}"])
            elif tag == "drink":
                results.extend([f"I want to drink {token}", f"Can I have {token}", f"Please give me {token}"])
            elif tag == "place":
                results.extend([f"I want to go to the {token}", f"Please take me to the {token}", f"Can I go to the {token}"])
            elif tag == "greeting":
                results.append(token)
            elif tag=="positive_feeling":
                results.extend([f"I feel {token}", f"I feel very {token}", f"I want to be {token}"])
            elif tag=="negative_feeling":
                results.extend([f"I feel {token}", f"I feel very {token}", f"I am feeling {token}",f"I wish I didn't feel {token}"])
            elif tag=="request":
                results.append(token)
            elif tag=="family":
                results.extend([f"I want to see my {token}", f"I miss my {token}", f"I love my {token}"])
            elif tag=="professional":
                results.extend([f"I want to see the {token}", f"I need help from the {token}", f"I want to talk to the {token}"])
            else:
                # --- ENHANCED FALLBACK ---
                results.extend([
                    f"I want {token}",
                    f"Can I have {token}",
                    f"Please give me {token}",
                    f"I need {token}",
                    f"Help me with {token}"
                ])
        return results

    def normalize(self, text: str):
        tokens = text.lower().split()
        corrected_tokens = [self.correct_word(t) for t in tokens]
        expansions = self._category_expansions(corrected_tokens)
        if not expansions:
            return self.generate_hint_sentences(" ".join(corrected_tokens))
        return list(set(expansions))

    def process_input(self, text: str):
        tokens = text.lower().split()
        corrected_tokens = [self.correct_word(t) for t in tokens]
        content_tokens = [t for t in corrected_tokens if t not in FUNCTION_WORDS]
        results = self._category_expansions(content_tokens)
        if not results and corrected_tokens != tokens:
            return [" ".join(corrected_tokens)]
        if not results:
            return self.generate_hint_sentences(" ".join(content_tokens))
        return list(set(results))

# -------------------------------
# Example Usage
# -------------------------------
""""
if __name__ == "__main__":
    normalizer = InputNormalizer()

    # Examples
    #print(normalizer.process_input("therapi"))          # → corrected to 'therapist'
    print(normalizer.process_input("goomrng"))  # → 'doctor'
    #print(normalizer.process_input("blender"))          # → unknown object → fallback sentences
    #print(normalizer.process_input("hap"))              # → positive feeling 'happy'
    #print(normalizer.process_input("I want xyz"))       # → xyz is unknown → fallback sentences
    #print(normalizer.process_input("I nee noodl to ea"))       # → xyz is unknown → fallback sentences
"""""
