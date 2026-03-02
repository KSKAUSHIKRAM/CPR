import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rapidfuzz import fuzz


# -----------------------------------------
# Dataset Loader
# -----------------------------------------
def load_dataset(path="CPR_CRD.csv"):
    df = pd.read_csv(path)
    df["normalized"] = df["normalized"].astype(str).str.lower().str.strip()
    df["correct_label"] = df["correct_label"].astype(str).str.strip()
    return df


# -----------------------------------------
# Hybrid Linguistic Normalization
# -----------------------------------------
def rule_based_correction(query):
    # Example deterministic corrections (extend as needed)
    rules = {
        "watr": "water",
        "mil": "milk",
        "pls": "please",
        "luv": "love"
    }
    tokens = query.split()
    corrected = [rules.get(tok, tok) for tok in tokens]
    return " ".join(corrected)


def grammar_valid(query):
    # Placeholder: assume valid if not empty
    return len(query.strip()) > 0


def lexical_confidence(query):
    # Basic confidence proxy using token length
    return min(1.0, len(query.split()) / 3.0)


def ai_normalize(query):
    # AI fallback stub (kept deterministic for reproducibility)
    # Replace with actual LLM call if needed
    return None


def hybrid_normalize(query, corpus_sentences, tau=0.75):

    q = query.lower().strip()

    # Stage I: Rule-based
    q_rule = rule_based_correction(q)

    if grammar_valid(q_rule) and lexical_confidence(q_rule) >= 0.8:
        return q_rule

    # Stage II: Similarity-based autocorrection
    similarity_scores = [
        fuzz.token_sort_ratio(q_rule, sent)/100.0
        for sent in corpus_sentences
    ]

    best_idx = np.argmax(similarity_scores)
    best_score = similarity_scores[best_idx]

    if best_score >= tau:
        return corpus_sentences[best_idx]

    # Stage III: AI fallback
    q_ai = ai_normalize(query)

    if q_ai is not None:
        return q_ai

    return q_rule


# -----------------------------------------
# Retriever Class
# -----------------------------------------
class CPRSatRetriever:
    def __init__(self, train_df, use_context=False):
        self.sentences = train_df["normalized"].tolist()
        self.labels = train_df["correct_label"].tolist()
        self.time = train_df["time_context"].tolist()
        self.location = train_df["location_context"].tolist()
        self.use_context = use_context

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.sentence_vectors = self.vectorizer.fit_transform(self.sentences)

    def rank(self, query, time_context=None, location_context=None):

        # 🔹 Hybrid Normalization (Manuscript aligned)
        query = hybrid_normalize(query, self.sentences)

        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.sentence_vectors).flatten()

        fuzzy_scores = np.array([
            fuzz.token_sort_ratio(query, sent)/100.0
            for sent in self.sentences
        ])

        scores = 0.7 * tfidf_scores + 0.3 * fuzzy_scores

        # Context-aware re-ranking
        if self.use_context and time_context is not None:
            context_boost = np.zeros(len(scores))
            for i in range(len(scores)):
                if self.time[i] == time_context:
                    context_boost[i] += 0.05
                if self.location[i] == location_context:
                    context_boost[i] += 0.05
            scores += context_boost

        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices


# -----------------------------------------
# Fold Evaluation
# -----------------------------------------
def evaluate_fold(train_df, test_df, use_context):

    retriever = CPRSatRetriever(train_df, use_context)
    predicted_ranks = []

    for _, row in test_df.iterrows():
        ranked_indices = retriever.rank(
            row["normalized"],
            row["time_context"],
            row["location_context"]
        )

        ranked_labels = [train_df["correct_label"].iloc[i] for i in ranked_indices]
        match = np.where(np.array(ranked_labels) == row["correct_label"])[0]

        if len(match) > 0:
            predicted_ranks.append(match[0] + 1)
        else:
            predicted_ranks.append(len(ranked_labels))

    predicted_ranks = np.array(predicted_ranks)

    mae = mean_absolute_error(np.ones(len(predicted_ranks)), predicted_ranks)
    rmse = np.sqrt(mean_squared_error(np.ones(len(predicted_ranks)), predicted_ranks))
    p1 = np.mean(predicted_ranks <= 1)
    p3 = np.mean(predicted_ranks <= 3)
    p5 = np.mean(predicted_ranks <= 5)

    return mae, rmse, p1, p3, p5


# -----------------------------------------
# Cross Validation
# -----------------------------------------
def main():

    df = load_dataset("CPR_CRD.csv")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("\n===== BASELINE (No Context) =====")
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["correct_label"])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        mae, rmse, p1, p3, p5 = evaluate_fold(train_df, test_df, use_context=False)

        print(f"Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}, P@1={p1:.3f}")

    print("\n===== CONTEXT-AWARE =====")
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["correct_label"])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        mae, rmse, p1, p3, p5 = evaluate_fold(train_df, test_df, use_context=True)

        print(f"Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}, P@1={p1:.3f}")


if __name__ == "__main__":
    main()