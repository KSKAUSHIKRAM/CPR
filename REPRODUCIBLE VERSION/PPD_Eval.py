import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from rapidfuzz import fuzz


# -----------------------------------------
# Load Dataset
# -----------------------------------------
def load_dataset(file_path="CPR_PPD.csv"):
    df = pd.read_csv(file_path, encoding="latin1")
    df["gt_label"] = df["gt_label"].astype(str).str.strip().str.replace(",", "", regex=False)
    df["normalized"] = df["normalized"].astype(str).str.lower().str.strip()
    return df


# -----------------------------------------
# CPR-SAT Hybrid Retriever
# -----------------------------------------
class CPRSatRetriever:
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.sentence_vectors = self.vectorizer.fit_transform(self.sentences)

    def rank(self, query):
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.sentence_vectors).flatten()

        fuzzy_scores = np.array([
                    fuzz.token_sort_ratio(query, sent) / 100.0
                    for sent in self.sentences
                ])

        final_scores = 0.7 * tfidf_scores + 0.3 * fuzzy_scores
        ranked_indices = np.argsort(final_scores)[::-1]
        return ranked_indices


# -----------------------------------------
# Evaluate One Fold
# -----------------------------------------
def evaluate_fold(train_df, test_df):
    top_n_values = [1, 9, 18, 25, 36]
    results = {n: 0 for n in top_n_values}

    train_sentences = train_df["normalized"].tolist()
    train_labels = train_df["gt_label"].tolist()

    retriever = CPRSatRetriever(train_sentences, train_labels)

    for _, row in test_df.iterrows():
        query = row["normalized"]
        ranked_indices = retriever.rank(query)
        ranked_labels = [train_labels[i] for i in ranked_indices]

        for n in top_n_values:
            if row["gt_label"] in ranked_labels[:n]:
                results[n] += 1

    total = len(test_df)
    for n in top_n_values:
        results[n] /= total

    return results


# -----------------------------------------
# 5-Fold Cross Validation
# -----------------------------------------
def main():
    print("\n=================================")
    print("CPR-SAT 5-Fold Cross Validation")
    print("=================================\n")

    df = load_dataset()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        fold_result = evaluate_fold(train_df, test_df)
        all_results.append(fold_result)

        print(f"Fold {fold+1} Results:")
        for k, v in fold_result.items():
            print(f"Top-{k}: {v:.4f}")
        print()

    # Compute mean & std
    summary = {}
    for n in [1, 9, 18, 25, 36]:
        values = [res[n] for res in all_results]
        summary[f"Top-{n}_mean"] = np.mean(values)
        summary[f"Top-{n}_std"] = np.std(values)

    print("=================================")
    print("Cross-Validation Summary")
    print("=================================\n")

    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    pd.DataFrame([summary]).to_csv("cprsat_5fold_results.csv", index=False)
    print("\nResults saved to cprsat_5fold_results.csv")


if __name__ == "__main__":
    main()