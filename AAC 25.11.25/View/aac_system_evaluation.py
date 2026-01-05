"""
====================================================================
 AAC / CPR SYSTEM EVALUATION FRAMEWORK
 Author: [Your Name]
 Description:
     Comprehensive evaluation of:
       - Input normalization
       - Category detection
       - Label matching
       - Pictogram retrieval
       - Sentence recommendation
       - System responsiveness
       - User interaction effectiveness
====================================================================
"""

import time
import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from statistics import mean
from Levenshtein import distance as levenshtein_distance

# --------------------------------------------------------------
# GLOBAL MODELS
# --------------------------------------------------------------
rouge = Rouge()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def print_title(title):
    print("\n" + "=" * 70)
    print(f"📘 {title}")
    print("=" * 70)

# --------------------------------------------------------------
# 1️⃣ INPUT NORMALIZATION ACCURACY
# --------------------------------------------------------------
def input_normalization_metrics(pairs):
    print_title("1️⃣ INPUT NORMALIZATION ACCURACY")
    total, correct, total_chars, total_edit = 0, 0, 0, 0
    for raw, pred, gold in pairs:
        total += 1
        if pred.strip().lower() == gold.strip().lower():
            correct += 1
        total_chars += len(gold)
        total_edit += levenshtein_distance(pred, gold)
    word_acc = (correct / total) * 100
    cer = total_edit / total_chars
    print(f"🔹 Word Accuracy: {word_acc:.2f}%")
    print(f"🔹 Character Error Rate (CER): {cer:.3f}")
    return word_acc, cer

# --------------------------------------------------------------
# 2️⃣ CATEGORY DETECTION — PRECISION, RECALL, F1
# --------------------------------------------------------------
def category_detection_metrics(true_labels, predicted_labels):
    print_title("2️⃣ CATEGORY DETECTION — PRECISION, RECALL, F1")
    p, r, f, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro', zero_division=0)
    print(f"🔹 Precision: {p:.3f}")
    print(f"🔹 Recall: {r:.3f}")
    print(f"🔹 F1-Score: {f:.3f}")
    return p, r, f

# --------------------------------------------------------------
# 3️⃣ LABEL MATCHING — TOP-1, MRR, P@K, R@K, MAP
# --------------------------------------------------------------
def label_matching_metrics(predicted_rankings, true_label, k=5):
    print_title("3️⃣ LABEL MATCHING — TOP-1, MRR, PRECISION@K, RECALL@K, MAP")
    top1, rr, precisions, recalls, avg_precisions = [], [], [], [], []
    for i, ranking in enumerate(predicted_rankings):
        gold = true_label[i]
        if gold in ranking:
            rank = ranking.index(gold) + 1
            rr.append(1 / rank)
            if rank == 1: top1.append(1)
        else:
            rr.append(0)
            top1.append(0)

        retrieved_k = ranking[:k]
        relevant = int(gold in retrieved_k)
        precisions.append(relevant / k)
        recalls.append(relevant / 1)
        ap = 1 / (ranking.index(gold) + 1) if gold in ranking else 0
        avg_precisions.append(ap)

    print(f"🔹 Top-1 Accuracy: {np.mean(top1):.3f}")
    print(f"🔹 MRR: {np.mean(rr):.3f}")
    print(f"🔹 Precision@{k}: {np.mean(precisions):.3f}")
    print(f"🔹 Recall@{k}: {np.mean(recalls):.3f}")
    print(f"🔹 MAP: {np.mean(avg_precisions):.3f}")
    return {
        "Top-1": np.mean(top1),
        "MRR": np.mean(rr),
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        "MAP": np.mean(avg_precisions)
    }

# --------------------------------------------------------------
# 4️⃣ PICTOGRAM RETRIEVAL — PRECISION@K, RECALL@K, MAP
# --------------------------------------------------------------
def pictogram_retrieval_metrics(predicted_pictos, true_pictos, k=5):
    print_title("4️⃣ PICTOGRAM RETRIEVAL — PRECISION@K, RECALL@K, MAP")
    precisions, recalls, aps = [], [], []
    for pred, gold in zip(predicted_pictos, true_pictos):
        pred_k = pred[:k]
        correct = len(set(pred_k) & set(gold))
        precisions.append(correct / k)
        recalls.append(correct / len(gold) if gold else 0)
        avg_p = sum([1/(i+1) for i, p in enumerate(pred_k) if p in gold]) / len(gold) if gold else 0
        aps.append(avg_p)
    print(f"🔹 Precision@{k}: {mean(precisions):.3f}")
    print(f"🔹 Recall@{k}: {mean(recalls):.3f}")
    print(f"🔹 MAP: {mean(aps):.3f}")
    return {
        "Precision@k": mean(precisions),
        "Recall@k": mean(recalls),
        "MAP": mean(aps)
    }

# --------------------------------------------------------------
# 5️⃣ SENTENCE RECOMMENDATION — BLEU / METEOR / ROUGE-L / COSINE / LIKERT
# --------------------------------------------------------------
def sentence_recommendation_metrics(predictions, references, expert_scores=None):
    print_title("5️⃣ SENTENCE RECOMMENDATION — BLEU / METEOR / ROUGE-L / COSINE / LIKERT")
    bleu_scores, meteor_scores, rouge_scores, cosine_scores = [], [], [], []

    for pred, ref in zip(predictions, references):
        bleu_scores.append(sentence_bleu([ref.split()], pred.split()))
        meteor_scores.append(meteor_score([ref], pred))
        rouge_l = rouge.get_scores(pred, ref)[0]['rouge-l']['f']
        rouge_scores.append(rouge_l)
        v1, v2 = embed_model.encode([pred, ref])
        cosine_scores.append(cosine_similarity([v1], [v2])[0][0])

    print(f"🔹 BLEU: {mean(bleu_scores):.3f}")
    print(f"🔹 METEOR: {mean(meteor_scores):.3f}")
    print(f"🔹 ROUGE-L: {mean(rouge_scores):.3f}")
    print(f"🔹 Cosine Similarity: {mean(cosine_scores):.3f}")
    if expert_scores:
        print(f"🔹 Expert Relevance (Likert): {mean(expert_scores):.2f}/5.00")

    return {
        "BLEU": mean(bleu_scores),
        "METEOR": mean(meteor_scores),
        "ROUGE-L": mean(rouge_scores),
        "Cosine": mean(cosine_scores),
        "Likert": mean(expert_scores) if expert_scores else None
    }

# --------------------------------------------------------------
# 6️⃣ SYSTEM RESPONSIVENESS — LATENCY
# --------------------------------------------------------------
def measure_latency(function_to_test, *args, **kwargs):
    print_title("6️⃣ SYSTEM RESPONSIVENESS — LATENCY")
    start = time.time()
    _ = function_to_test(*args, **kwargs)
    end = time.time()
    latency = end - start
    print(f"⚡ Response Time: {latency:.3f} sec")
    return latency

# --------------------------------------------------------------
# 7️⃣ USER INTERACTION EFFECTIVENESS — TASK SUCCESS, SUS
# --------------------------------------------------------------
def user_interaction_metrics(task_success, total_tasks, usability_score):
    print_title("7️⃣ USER INTERACTION EFFECTIVENESS — TASK SUCCESS, SUS")
    success_rate = (task_success / total_tasks) * 100
    print(f"🔹 Task Success Rate: {success_rate:.2f}%")
    print(f"🔹 System Usability Scale (SUS): {usability_score}/100")
    return success_rate, usability_score


# ==============================================================
# EXAMPLE RUN (Demonstration)
# ==============================================================
if __name__ == "__main__":

    normalization_pairs = [("giv", "give", "give"), ("wat", "water", "water"), ("luv", "luv", "love")]
    input_normalization_metrics(normalization_pairs)

    true_cats = ["food", "drink", "emotion"]
    pred_cats = ["food", "food", "emotion"]
    category_detection_metrics(true_cats, pred_cats)

    rankings = [["apple", "banana", "orange"], ["juice", "milk", "water"], ["happy", "sad"]]
    truths = ["banana", "water", "happy"]
    label_matching_metrics(rankings, truths, k=3)

    predicted_pictos = [["apple", "orange"], ["milk", "juice"], ["happy", "sad"]]
    true_pictos = [["apple"], ["juice"], ["happy"]]
    pictogram_retrieval_metrics(predicted_pictos, true_pictos)

    preds = ["I want water", "Give me food"]
    refs = ["I want some water", "I need food"]
    expert = [4, 5]
    sentence_recommendation_metrics(preds, refs, expert)

    def dummy_func(x): time.sleep(0.4); return x
    measure_latency(dummy_func, "test")

    user_interaction_metrics(9, 10, 88)
