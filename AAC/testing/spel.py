# file: text_suggestor.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple
from symspellpy import SymSpell, Verbosity
from rapidfuzz import fuzz, process as rf_process
from wordfreq import zipf_frequency

# ---------- 1) Normalization ----------
COLLOQUIAL_MAP = {
    "wanna": "want to",
    "wanna.": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
    "gimme": "give me",
    "whatcha": "what are you",
    "innit": "isn't it",
    "donno": "don't know",
    "wate": "water",  # useful special-case
    "wann": "want",
}

def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    tokens = []
    for tok in t.split():
        tokens.extend(COLLOQUIAL_MAP.get(tok, tok).split())
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()

# ---------- 2) SymSpell setup ----------
class Speller:
    def __init__(self, dict_dir: str, max_edit: int = 2):
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=7)
        dict_dir = Path(dict_dir)
        # Unigram
        self.sym.load_dictionary(str(dict_dir/"frequency_dictionary_en_82_765.txt"),
                                 term_index=0, count_index=1)
        # Bigram (for lookup_compound context)
        self.sym.load_bigram_dictionary(str(dict_dir/"frequency_bigramdictionary_en_243_342.txt"),
                                        term_index=0, count_index=2)

        # --- Domain lexicon (unigrams) ---
        domain_path = dict_dir / "domain_terms_aac.tsv"
        if domain_path.exists():
            loaded = self.sym.load_dictionary(str(domain_path), term_index=0, count_index=1)
            if not loaded:
                with domain_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            term, freq = parts[0], parts[1]
                            try:
                                self.sym.create_dictionary_entry(term, int(freq))
                            except ValueError:
                                pass
        else:
            print(f"[Speller] domain_terms_aac.tsv not found in {dict_dir}")

    def correct_sentence(self, text: str, topk: int = 10) -> List[Tuple[str, float]]:
        """
        Returns list of (sentence, score). Higher score = better.
        SymSpell’s score is frequency-based; we lightly re-weight with wordfreq.
        """
        base = self.sym.lookup_compound(text, max_edit_distance=2)
        cand = [(s.term, s.distance, s.count) for s in base][:topk]

        reweighted = []
        for sent, dist, cnt in cand:
            toks = sent.split()
            if not toks:
                continue
            avg_zipf = sum(zipf_frequency(w, "en") for w in toks) / len(toks)
            score = avg_zipf + (cnt / 1_000_000.0) - 0.1*dist
            reweighted.append((sent, score))
        reweighted.sort(key=lambda x: x[1], reverse=True)
        return reweighted

# ---------- 3) Optional KenLM ranking ----------
try:
    import kenlm
    HAVE_KENLM = True
except Exception:
    HAVE_KENLM = False

class Ranker:
    def __init__(self, kenlm_path: str | None = None):
        self.model = None
        if kenlm_path and Path(kenlm_path).exists() and HAVE_KENLM:
            self.model = kenlm.Model(kenlm_path)

    def score(self, s: str) -> float:
        if self.model:
            return self.model.score(s, bos=True, eos=True)
        toks = s.split()
        return sum(zipf_frequency(w, "en") for w in toks) / max(1, len(toks))

    def rerank(self, cands: List[str], topk: int = 10) -> List[str]:
        scored = [(c, self.score(c)) for c in cands]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored][:topk]

# ---------- 4) Intent library & paraphrases ----------
INTENT_PATTERNS = {
    "request_drink": [
        "i want {item}",
        "i need {item}",
        "can i have {item}",
        "could i have {item}",
        "can i get {item}",
        "i would like {item}",
        "please give me {item}",
        "may i have {item}",
        "may i get {item}",
        "please may i have {item}",
    ],
    "request_food": [
        "i want {item}",
        "i need {item}",
        "can i have {item}",
        "could i have {item}",
        "can i get {item}",
        "i would like {item}",
        "please give me {item}",
        "i want some {item}",
        "i need some {item}",
        "may i have {item}",
    ],
    "request_help": [
        "i need help",
        "can you help me",
        "please help me",
        "i want assistance",
        "i need assistance",
        "could you help me",
    ],
    "request_bathroom": [
        "i want to go to the bathroom",
        "i need the restroom",
        "can i go to the toilet",
        "may i go to the toilet",
        "i need to use the bathroom",
    ],
}

DRINK_WORDS = {"water", "juice", "milk", "tea", "coffee"}
FOOD_WORDS  = {"rice", "bread", "idli", "dosa", "apple", "banana", "date"}

def detect_item(tokens: List[str]) -> str | None:
    for t in tokens:
        if t in DRINK_WORDS | FOOD_WORDS:
            return t
    return None

def guess_intent(tokens: List[str]) -> str:
    s = " ".join(tokens)
    choices = {
        "request_drink": "drink water juice milk tea coffee",
        "request_food": "food eat hungry rice bread idli dosa apple banana date",
        "request_help": "help assistance support",
        "request_bathroom": "bathroom toilet restroom washroom"
    }
    best = rf_process.extractOne(s, list(choices.values()), scorer=fuzz.token_set_ratio)
    if not best:
        return "request_help"
    idx = list(choices.values()).index(best[0])
    return list(choices.keys())[idx]

def paraphrase_variants(intent: str, item: str | None, limit: int) -> List[str]:
    pats = INTENT_PATTERNS.get(intent, [])
    outs = []
    for p in pats:
        if "{item}" in p:
            if item:
                outs.append(p.format(item=item))
        else:
            outs.append(p)
    # cap, capitalize, punctuate
    outs = outs[:max(0, limit)]
    return [cap(s) for s in outs]

def cap(s: str) -> str:
    s = s.strip()
    if not s: 
        return s
    s = s[0].upper() + s[1:]
    if not re.search(r"[.!?]$", s):
        s += "."
    return s

# ---------- 5) Public API ----------
class SentenceSuggester:
    def __init__(self, dict_dir: str, kenlm_path: str | None = None):
        self.speller = Speller(dict_dir)
        self.ranker = Ranker(kenlm_path)

    def suggest(self, raw_text: str, n: int = 10) -> List[str]:
        if not raw_text.strip():
            return []

        norm = normalize(raw_text)

        # --- SymSpell with ALL candidates ---
        # --- SymSpell compound correction (no verbosity arg allowed here) ---
        base = self.speller.sym.lookup_compound(norm, max_edit_distance=2)
        corrected_texts = [s.term for s in base]

        # fallback: also try word-level lookup to expand pool
        if len(corrected_texts) < n:
            words = norm.split()
            for w in words:
                alts = self.speller.sym.lookup(w, Verbosity.ALL, max_edit_distance=2)
                corrected_texts.extend([a.term for a in alts])

        # dedupe, preserve order
        corrected_texts = list(dict.fromkeys(corrected_texts))

        # --- rerank ---
        best = self.ranker.rerank(corrected_texts, topk=n)

        # --- build output list ---
        out: List[str] = []
        seen = set()

        def add(s: str):
            nice = cap(s)
            if nice.lower() not in seen:
                seen.add(nice.lower())
                out.append(nice)

        for s in best:
            add(s)

        # --- generate paraphrases no matter what ---
        if best:
            toks = best[0].split()
            item = detect_item(toks)
            intent = guess_intent(toks)
            paras = paraphrase_variants(intent, item, limit=n * 2)  # lots
            for p in paras:
                add(p)
                if len(out) >= n:
                    break

        # --- still short? generic variants ---
        if best:
            base_sent = best[0]
            toks = base_sent.split()
            if len(toks) >= 2:
                subj, rest = toks[0], " ".join(toks[1:])
                variants = [
                    f"I want {rest}",
                    f"I need {rest}",
                    f"Can I have {rest}",
                    f"Please give me {rest}",
                    f"Could I get {rest}",
                ]
                for v in variants:
                    add(v)
                    if len(out) >= n:
                        break

        return out[:n]


# --- quick demo (remove in production) ---
if __name__ == "__main__":
    sugg = SentenceSuggester(dict_dir="/home/i7/Downloads/PICTOBERT-main/AAC", kenlm_path=None)
    print(sugg.suggest("i wann idl", n=10))
