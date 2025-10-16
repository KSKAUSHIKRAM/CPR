# View/input_norm.py
from __future__ import annotations
import os, re, csv, math
from difflib import get_close_matches, SequenceMatcher
from typing import List, Tuple, Iterable, Set, Dict, Optional
from pathlib import Path

# ===================== Utilities =====================
WORD_RE = re.compile(r"[A-Za-z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def cap_i_and_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    s = re.sub(r"\bi\b", "I", s)
    return s

def find_data_file(name: str) -> Optional[str]:
    """Search for a data file near this module or in CWD."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / name,
        here.parent / name,
        Path.cwd() / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

# ===================== Heuristics & Templates =====================
FUNCTION_WORDS: Set[str] = {
    "i","to","the","a","please","me","can","have","is","am","are","on","in","at","for",
    "of","and","or","it","you","he","she","they","we","this","that","these","those",
    "with","my","your","his","her","their","our","want","need","go","give","get","drink","eat","like","let","let's"
}

FOOD_HINTS = {
    "idly","idli","dosa","parotta","parota","pulao","biryani","sambar","rasam",
    "chutney","rice","curd","curd rice","chapati","poori","upma","pongal","noodles",
    "egg","chicken","fish","mutton","paneer","vegetable","veg","sabzi","sadam"
}
DRINK_HINTS = {"tea","coffee","milk","juice","water","lassi","buttermilk"}
PLACE_HINTS = {
    "park","school","toilet","bathroom","restroom","hospital","clinic","temple","church","mosque",
    "market","shop","office","home","house","kitchen","station","bank","atm","pharmacy","playground",
    "bus stop","stop","outside","garden","beach","restaurant"
}

TAG_TEMPLATES: Dict[str, List[str]] = {
    "food": [
        "I want {w}",
        "I want to eat {w}",
        "Can I have {w}",
        "Please give me {w}",
        "I need {w}",
    ],
    "drink": [
        "I want {w}",
        "I want to drink {w}",
        "Can I have {w}",
        "Please give me {w}",
    ],
    "place": [
        "I want to go to {w}",
        "I need to go to {w}",
        "Take me to {w}",
        "Please take me to {w}",
    ],
    "object": [
        "I want {w}",
        "Please give me {w}",
        "I need {w}",
    ],
    "_default": [
        "I want {w}",
        "I need {w}",
        "Please give me {w}",
    ],
}
TAG_GENERIC_SENTENCES: Dict[str, List[str]] = {
    "food": ["I am hungry", "I want food"],
    "drink": ["I am thirsty", "I want a drink"],
    "place": ["I want to go out", "Let's go outside"],
}

# ===================== Self-contained suggester =====================
class _ContextAACSuggester:
    """
    Word-lexicon driven AAC suggester with category-aware expansions and a tiny bigram LM.
    Files (auto-located): domain_terms_aac.tsv, frequency_dictionary_en_82_765.txt, frequency_bigramdictionary_en_243_342.txt
    """

    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        unigram_path: Optional[str] = None,
        bigram_path: Optional[str] = None,
        top_k: int = 6,
        max_matches_per_token: int = 3,
        max_related_per_tag: int = 4,
        cutoff: float = 0.68,
        lambda_text_vs_lm: float = 0.6
    ):
        self.top_k = top_k
        self.max_matches_per_token = max_matches_per_token
        self.max_related_per_tag = max_related_per_tag
        self.cutoff = cutoff
        self.lambda_text_vs_lm = max(0.0, min(1.0, lambda_text_vs_lm))

        # Resolve data paths
        lexicon_path = lexicon_path or find_data_file("domain_terms_aac.tsv")
        unigram_path = unigram_path or find_data_file("frequency_dictionary_en_82_765.txt")
        bigram_path  = bigram_path  or find_data_file("frequency_bigramdictionary_en_243_342.txt")

        # --- load resources ---
        self.lexicon: Dict[str, float] = {}     # word -> prior score
        self.lex_tags: Dict[str, str] = {}      # word -> tag
        self._load_lexicon(lexicon_path)

        self.unigrams: Dict[str, int] = {}
        self.max_uni: int = 1
        self._load_unigrams(unigram_path)

        self.bigrams: Dict[Tuple[str, str], int] = {}
        self.max_bi: int = 1
        self._load_bigrams(bigram_path)

        # vocab for fuzzy correction
        self.vocab = self._build_vocab()

        # prior rescaling for scores (if very uneven)
        self._prior_min, self._prior_max = self._prior_range()

    # ---------- Loading ----------
    def _load_lexicon(self, path: Optional[str]):
        if not path or not os.path.exists(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tsv", ".csv"):
            self._load_lexicon_delimited(path, delimiter="\t" if ext == ".tsv" else ",")
        else:
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    w = line.strip()
                    if w and not w.lstrip().startswith("#"):
                        self.lexicon[w.lower()] = 1.0

    def _load_lexicon_delimited(self, path: str, delimiter: str):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            used_dict = False
            if reader.fieldnames:
                fields = {h.lower(): h for h in reader.fieldnames}
                col_text  = fields.get("text") or fields.get("word") or fields.get("term")
                col_score = fields.get("score")
                col_tag   = fields.get("tag")
                if col_text:
                    for row in reader:
                        w = (row.get(col_text) or "").strip()
                        if not w:
                            continue
                        score = 1.0
                        if col_score:
                            try:
                                score = float(row.get(col_score) or "1.0")
                            except ValueError:
                                score = 1.0
                        self.lexicon[w.lower()] = score
                        if col_tag:
                            tag = (row.get(col_tag) or "").strip().lower()
                            if tag:
                                self.lex_tags[w.lower()] = tag
                    used_dict = True
            if not used_dict:
                f.seek(0)
                rdr = csv.reader(f, delimiter=delimiter)
                for row in rdr:
                    if not row:
                        continue
                    if isinstance(row[0], str) and row[0].lstrip().startswith("#"):
                        continue
                    w = (row[0] or "").strip()
                    if not w:
                        continue
                    score = 1.0
                    if len(row) > 1:
                        try:
                            score = float((row[1] or "1.0"))
                        except ValueError:
                            score = 1.0
                    self.lexicon[w.lower()] = score
                    if len(row) > 2 and isinstance(row[2], str):
                        tag = (row[2] or "").strip().lower()
                        if tag:
                            self.lex_tags[w.lower()] = tag

    def _load_unigrams(self, path: Optional[str]):
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tok, cnt = line.rsplit(" ", 1)
                    cnt = int(cnt)
                except ValueError:
                    continue
                self.unigrams[tok.lower()] = cnt
        if self.unigrams:
            self.max_uni = max(self.unigrams.values())

    def _load_bigrams(self, path: Optional[str]):
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    phrase, cnt = line.rsplit(" ", 1)
                    cnt = int(cnt)
                except ValueError:
                    continue
                parts = phrase.split()
                if len(parts) != 2:
                    continue
                w1, w2 = parts[0].lower(), parts[1].lower()
                self.bigrams[(w1, w2)] = cnt
        if self.bigrams:
            self.max_bi = max(self.bigrams.values())

    # ---------- Vocab & normalization ----------
    def _build_vocab(self) -> Set[str]:
        vocab: Set[str] = set(FUNCTION_WORDS)
        for w in self.lexicon.keys():
            for t in tokenize(w):
                vocab.add(t)
        return vocab

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return text
        # micro-normalizations (extend as needed)
        text = re.sub(r"\bwann\b|\bwan\b", "want", text, flags=re.I)
        text = re.sub(r"\bidl\b|\bidlyy?\b", "idly", text, flags=re.I)
        text = re.sub(r"\bwate\b", "water", text, flags=re.I)
        return text

    # ---------- Prior handling ----------
    def _prior_range(self) -> Tuple[float, float]:
        vals = list(self.lexicon.values()) or [1.0]
        return min(vals), max(vals)

    def _prior_boost(self, raw: float | None) -> float:
        """Rescale lexicon prior to ~[0.9, 1.3] so it nudges but never dominates."""
        if raw is None:
            return 1.0
        lo, hi = self._prior_min, self._prior_max
        if hi <= lo:
            return 1.0
        x = max(0.0, min(1.0, (float(raw) - lo) / (hi - lo)))
        return 0.9 + 0.4 * x

    # ---------- Tag inference ----------
    def _infer_tag(self, term: str) -> Optional[str]:
        t = term.lower()
        if t in self.lex_tags:
            return self.lex_tags[t]
        if t in FOOD_HINTS:
            return "food"
        if t in DRINK_HINTS:
            return "drink"
        if t in PLACE_HINTS:
            return "place"
        return None

    # ---------- Nearest lexicon terms ----------
    def _nearest_terms(self, token: str) -> List[str]:
        if not self.lexicon:
            return []
        matches = get_close_matches(
            token.lower(),
            list(self.lexicon.keys()),
            n=self.max_matches_per_token,
            cutoff=self.cutoff,
        )
        return [m for m in matches if m not in FUNCTION_WORDS]

    # ---------- Candidate construction ----------
    def _direct_corrected_sentence(self, tokens: List[str]) -> Optional[str]:
        out: List[str] = []
        changed = False
        for w in tokens:
            if w in FUNCTION_WORDS:
                out.append(w)
                continue
            matches = self._nearest_terms(w)
            if matches:
                out.append(matches[0])
                changed = True
            else:
                out.append(w)
        if not out:
            return None
        if not changed and len(tokens) < 2:
            return None
        return cap_i_and_sentence(" ".join(out))

    def _template_expansions_for_terms(self, terms: List[str]) -> List[str]:
        expansions: Set[str] = set()
        for term in terms:
            tag = self._infer_tag(term) or "_default"
            templates = TAG_TEMPLATES.get(tag, TAG_TEMPLATES["_default"])
            for tmpl in templates:
                expansions.add(tmpl.format(w=term))
        return [cap_i_and_sentence(re.sub(r"\s+", " ", s).strip()) for s in expansions]

    def _collect_matched_terms(self, tokens: List[str]) -> Dict[str, Set[str]]:
        tag2words: Dict[str, Set[str]] = {}
        for w in tokens:
            if w in FUNCTION_WORDS:
                continue
            for term in self._nearest_terms(w):
                tag = self._infer_tag(term) or "_default"
                tag2words.setdefault(tag, set()).add(term)
        return tag2words

    def _related_terms_by_tag(self, tag: str, anchors: Set[str]) -> List[str]:
        words = []
        for w in self.lexicon.keys():
            inferred = self._infer_tag(w) or "_default"
            if inferred == tag and w not in FUNCTION_WORDS:
                words.append(w)
        if not words:
            return []
        def sim_to_anchors(w: str) -> float:
            return max((fuzzy_ratio(w, a) for a in anchors), default=0.0)
        lo, hi = self._prior_min, self._prior_max
        def norm_prior(w: str) -> float:
            p = self.lexicon.get(w, 1.0)
            if hi <= lo: return 1.0
            return (p - lo) / (hi - lo)
        scored = []
        for w in words:
            s = 0.65 * sim_to_anchors(w) + 0.35 * norm_prior(w)
            scored.append((s, w))
        scored.sort(reverse=True)
        out = []
        for a in sorted(anchors):
            if a in words:
                out.append(a)
        for _, w in scored:
            if w not in out:
                out.append(w)
            if len(out) >= max(len(anchors) + 4, 4):
                break
        return out

    def _category_expansions(self, tokens: List[str]) -> List[str]:
        tag2anchors = self._collect_matched_terms(tokens)
        expansions: Set[str] = set()
        for tag, anchors in tag2anchors.items():
            related = self._related_terms_by_tag(tag, anchors)
            terms_for_templates = related[: 4]
            for s in self._template_expansions_for_terms(terms_for_templates):
                expansions.add(s)
            for g in TAG_GENERIC_SENTENCES.get(tag, []):
                expansions.add(cap_i_and_sentence(g))
        return sorted(expansions)

    # ---------- Language model scoring ----------
    def _bigram_score(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        if len(tokens) == 1:
            c = self.unigrams.get(tokens[0], 0)
            return math.log1p(c) / math.log1p(self.max_uni) if self.max_uni > 0 else 0.0
        acc = 0.0
        n = 0
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            c = self.bigrams.get((w1, w2))
            if c is None:
                c2 = self.unigrams.get(w2, 0)
                pseudo = 0.4 * c2
                acc += math.log1p(pseudo) / math.log1p(self.max_bi)
            else:
                acc += math.log1p(c) / math.log1p(self.max_bi)
            n += 1
        return acc / max(1, n)

    def _tsv_prior_boost(self, candidate_tokens: List[str]) -> float:
        vals = [self.lexicon.get(t, 1.0) for t in candidate_tokens]
        prior_raw = max(vals) if vals else 1.0
        return self._prior_boost(prior_raw)

    def _score(self, query: str, candidate: str) -> float:
        q_toks = tokenize(query)
        c_toks = tokenize(candidate)
        f = fuzzy_ratio(query.lower(), candidate.lower())
        j = jaccard(q_toks, c_toks)
        text_score = 0.7 * f + 0.3 * j
        lm_score = self._bigram_score(c_toks)
        base = self.lambda_text_vs_lm * text_score + (1.0 - self.lambda_text_vs_lm) * lm_score
        return base * self._tsv_prior_boost(c_toks)

    # ---------- Public ----------
    def suggest(self, raw_text: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        k = top_k or self.top_k
        text = self._normalize_text(raw_text)
        if not text:
            return []
        tokens = tokenize(text)
        candidates: List[str] = []
        direct = self._direct_corrected_sentence(tokens)
        if direct:
            candidates.append(direct)
        matched_terms: List[str] = []
        for w in tokens:
            if w in FUNCTION_WORDS:
                continue
            matched_terms.extend(self._nearest_terms(w))
        if matched_terms:
            uniq_terms = list(dict.fromkeys(matched_terms))
            candidates.extend(self._template_expansions_for_terms(uniq_terms))
        candidates.extend(self._category_expansions(tokens))
        if not candidates:
            candidates = ["I need help", "Please help me"]
        scored = [(cand, self._score(text, cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        out, seen = [], set()
        for s, sc in scored:
            if s not in seen:
                out.append((s, float(sc)))
                seen.add(s)
            if len(out) >= k:
                break
        return out

# ===================== Public API =====================
# Create a single, reusable suggester instance at import-time
_SUGGESTER = _ContextAACSuggester(top_k=8, cutoff=0.66, lambda_text_vs_lm=0.6)

def get_suggestions(text: str, top_k: int = 6) -> list[str]:
    """
    Return top-k suggestion sentences for the given input.
    Called from View/AACScreen.py: on_go_clicked -> inputs_norm.get_suggestions(...)
    """
    if not text or not text.strip():
        return []
    pairs = _SUGGESTER.suggest(text, top_k=top_k)
    return [s for (s, _) in pairs]
