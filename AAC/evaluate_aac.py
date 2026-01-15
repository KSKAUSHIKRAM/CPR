# ------------------------------------------------------------
# evaluate_aac.py (Context-Aware Version)
# Adds location/time context to CPR-SAT evaluation
# ------------------------------------------------------------

import csv
import os
import re
import random
import argparse
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
LABEL_FUZZY_THRESHOLD = 0.60
SENTENCE_MATCH_THRESHOLD = 0.85
AUTOCORRECT_CUTOFF = 0.75

# Context weight values
CONTEXT_WEIGHT = 0.20     # How strongly context influences ranking
TIME_WEIGHT = 0.10        # How strongly time influences ranking


# -----------------------------
# CONTEXT DICTIONARIES
# -----------------------------

# Common AAC contexts (customizable)
LOCATION_CONTEXT = {
    "home": ["eat", "sleep", "bath", "milk", "rice"],
    "school": ["write", "read", "teacher", "friend", "class"],
    "market": ["buy", "water", "vegetables", "fruits"],
    "hospital": ["pain", "medicine", "help", "doctor"],
    "restaurant": ["food", "eat", "drink", "water"]
}

TIME_CONTEXT = {
    "morning": ["breakfast", "brush", "school", "milk"],
    "afternoon": ["lunch", "play", "water"],
    "evening": ["snack", "dinner", "study"],
    "night": ["sleep", "bath", "rest"]
}


# -----------------------------
# TIME BUCKET
# -----------------------------
def get_time_bucket(hour=None):
    if hour is None:
        hour = datetime.now().hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    return "night"


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset():
    paths = ["dataset.csv", os.path.join(os.getcwd(), "dataset.csv")]
    for path in paths:
        if os.path.exists(path):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    lbl = (r.get("label") or "").strip().lower()
                    sen = (r.get("yes_sentence") or "").strip().lower()
                    cat = (r.get("category") or "").strip().lower()
                    if lbl and sen:
                        rows.append({"label": lbl, "yes_sentence": sen, "category": cat})
            return rows
    raise FileNotFoundError("dataset.csv not found.")


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_input(text):
    mapping = {
        "wa": "want", "wan": "want", "giv": "give", "tak": "take",
        "plz": "please", "wat": "water", "mil": "milk", "hpy": "happy"
    }
    tokens = text.lower().split()
    return " ".join([mapping.get(w, w) for w in tokens])


# -----------------------------
# VOCAB
# -----------------------------
def build_vocab(dataset):
    vocab = set()
    for row in dataset:
        vocab.add(row["label"])
        for t in row["yes_sentence"].split():
            vocab.add(t)
    return vocab


# -----------------------------
# AUTOCORRECT
# -----------------------------
def autocorrect_with_dataset(text, vocab):
    corrected = []
    for w in text.split():
        if w in vocab:
            corrected.append(w)
        else:
            match = get_close_matches(w, vocab, n=1, cutoff=AUTOCORRECT_CUTOFF)
            corrected.append(match[0] if match else w)
    return " ".join(corrected)


# -----------------------------
# FUZZY LABEL & CATEGORY
# -----------------------------
def detect_label_and_category(sentence, dataset):
    best_label = None
    best_cat = None
    best_score = 0.0

    for row in dataset:
        lbl = row["label"]
        sent = row["yes_sentence"]

        # Direct match
        if lbl in sentence:
            return lbl, row["category"], "direct", 1.0

        score = SequenceMatcher(None, sentence, sent).ratio()
        if score > best_score:
            best_score = score
            best_label = lbl
            best_cat = row["category"]

    if best_score >= LABEL_FUZZY_THRESHOLD:
        return best_label, best_cat, "fuzzy", best_score

    return None, None, "none", 0.0


# -----------------------------
# TF-IDF BEST SENTENCE
# -----------------------------
def best_sentence_tfidf(text, dataset, vectorizer, matrix):
    sentences = [r["yes_sentence"] for r in dataset]
    vec = vectorizer.transform([text])
    scores = cosine_similarity(vec, matrix)[0]
    idx = scores.argmax()
    return dataset[idx], float(scores[idx])


# -----------------------------
# CONTEXT RE-RANKING FUNCTION
# -----------------------------
def context_score(label, location_bucket, time_bucket):
    score = 0.0

    if location_bucket in LOCATION_CONTEXT:
        if label in LOCATION_CONTEXT[location_bucket]:
            score += CONTEXT_WEIGHT

    if time_bucket in TIME_CONTEXT:
        if label in TIME_CONTEXT[time_bucket]:
            score += TIME_WEIGHT

    return score


# -----------------------------
# GENERATE NOISY VARIANTS
# -----------------------------
def make_noisy_variants(sentence, count):
    variants = set()
    words = sentence.split()

    for _ in range(count):
        w = words.copy()
        choice = random.randint(1, 4)

        if choice == 1:
            idx = random.randrange(len(w))
            w[idx] = re.sub(r"[aeiou]", "", w[idx])
        elif choice == 2:
            idx = random.randrange(len(w))
            tok = w[idx]
            if len(tok) > 2:
                i = random.randrange(len(tok) - 1)
                tok = list(tok)
                tok[i], tok[i + 1] = tok[i + 1], tok[i]
                w[idx] = "".join(tok)
        elif choice == 3:
            if len(w) > 1:
                idx = random.randrange(len(w) - 1)
                w[idx] = w[idx] + w[idx + 1]
                del w[idx + 1]
        else:
            idx = random.randrange(len(w))
            if len(w[idx]) > 1:
                w[idx] = w[idx][:-1]

        variants.add(" ".join(w))

    return list(variants)


# -----------------------------
# MAIN EVALUATION LOOP
# -----------------------------
def run_evaluation(dataset, noisy, use_autocorrect, use_tfidf, use_context, outfile):
    vocab = build_vocab(dataset)
    vectorizer = TfidfVectorizer().fit([r["yes_sentence"] for r in dataset])
    matrix = vectorizer.transform([r["yes_sentence"] for r in dataset])

    results = []
    total = 0
    correct_label = 0
    correct_cat = 0

    # context buckets
    time_bucket = get_time_bucket()
    possible_locations = list(LOCATION_CONTEXT.keys())
    location_bucket = random.choice(possible_locations)   # simulate

    for row in dataset:
        base = row["yes_sentence"]
        variants = [base] + make_noisy_variants(base, noisy)

        for inp in variants:
            total += 1

            normalized = normalize_input(inp)
            if use_autocorrect:
                normalized = autocorrect_with_dataset(normalized, vocab)

            lbl, cat, method, score = detect_label_and_category(normalized, dataset)

            # TF-IDF layer
            if use_tfidf:
                best_row, tfidf_score = best_sentence_tfidf(normalized, dataset, vectorizer, matrix)
                if tfidf_score >= SENTENCE_MATCH_THRESHOLD and tfidf_score > score:
                    lbl = best_row["label"]
                    cat = best_row["category"]
                    method = "tfidf"
                    score = tfidf_score

            # Apply CONTEXT
            if use_context and lbl is not None:
                score += context_score(lbl, location_bucket, time_bucket)
                method = method + "+context"

            if lbl == row["label"]:
                correct_label += 1
            if cat == row["category"]:
                correct_cat += 1

            results.append({
                "input": inp,
                "normalized": normalized,
                "label": lbl,
                "category": cat,
                "method": method,
                "score": score,
                "location_context": location_bucket,
                "time_context": time_bucket,
                "gt_label": row["label"],
                "gt_category": row["category"],
                "correct_label": int(lbl == row["label"]),
                "correct_category": int(cat == row["category"])
            })

    # SAVE RESULTS
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return {
        "total": total,
        "label_accuracy": correct_label / total,
        "category_accuracy": correct_cat / total,
        "outfile": outfile
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-per-case", type=int, default=2)
    parser.add_argument("--context", choices=["on", "off"], default="off")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset()

    use_context = (args.context == "on")

    if args.ablation:
        configs = [
            ("full_context.csv", True, True, use_context),
            ("no_ac.csv", False, True, use_context),
            ("no_tfidf.csv", True, False, use_context),
            ("no_ac_no_tfidf.csv", False, False, use_context),
        ]
        for fname, ac, tf, ctx in configs:
            r = run_evaluation(dataset, args.noisy_per_case, ac, tf, ctx, outfile=fname)
            print(fname, r)
    else:
        r = run_evaluation(dataset, args.noisy_per_case, True, True, use_context, outfile="eval_context.csv")
        print(r)


if __name__ == "__main__":
    main()
