# debug_eval_test.py
import time
import os
import sys
sys.path.append(os.path.abspath("."))  # ensure project root is on path

from View.evaluation_monitor import EvaluationMonitor
from View import aac_system_evaluation as evalmod

def run_direct_function_tests():
    # Test 1: normalization
    pairs = [
        ("i wnt watr", "I want water", "I want water"),
        ("giv me food", "give me food", "give me food"),
        ("turn on ligth", "turn on light", "turn on light"),
    ]
    print("\n== Normalization metrics test ==")
    wa, cer = evalmod.input_normalization_metrics(pairs)
    print("-> word_acc, cer:", wa, cer)

    # Test 2: category detection
    true = ["Drink", "Food", "Object", "Emotion"]
    pred = ["Drink", "Food", "Object", "Emotion"]
    print("\n== Category detection test ==")
    p, r, f = evalmod.category_detection_metrics(true, pred)
    print("-> p,r,f:", p, r, f)

    # Test 3: label matching (ranking)
    rankings = [
        ["drink water", "drink juice", "eat food"],
        ["eat food", "drink water", "sleep"],
        ["turn on light", "turn off light"]
    ]
    true_labels = ["drink water", "eat food", "turn on light"]
    print("\n== Label matching test ==")
    # label_matching_metrics may return a dict (check)
    try:
        out = evalmod.label_matching_metrics(rankings, true_labels, k=3)
        print("-> label matching result:", out)
    except Exception as e:
        print("Label matching threw:", e)

    # Test 4: pictogram retrieval (toy)
    preds = [["water.png"], ["food.png"], ["light.png"]]
    truths = [["water.png"], ["food.png"], ["light.png"]]
    print("\n== Pictogram retrieval test ==")
    try:
        out = evalmod.pictogram_retrieval_metrics(preds, truths, k=1)
        print("-> pictogram retrieval:", out)
    except Exception as e:
        print("Pictogram retrieval threw:", e)

    # Test 5: sentence evaluation
    preds = ["I want water", "I want food"]
    refs = ["I need water", "I need food"]
    print("\n== Sentence recommendation test ==")
    try:
        out = evalmod.sentence_recommendation_metrics(preds, refs, expert_scores=[])
        print("-> sentence recommendation result:", out)
    except Exception as e:
        print("Sentence evaluation threw:", e)

if __name__ == "__main__":
    run_direct_function_tests()

    print("\n== Now test EvaluationMonitor integration ==")
    em = EvaluationMonitor(interval=5)
    # record a couple of items
    em.record_normalization("i wnt watr", "I want water", "I want water")
    em.record_category("Drink", "Drink")
    em.record_label(["drink water", "drink juice"], "drink water")
    em.record_pictogram(["water.png"], ["water.png"])
    em.record_sentence("I want water", "I need water")

    # immediate final evaluation
    print("\nCalling evaluate_all() now:")
    em.evaluate_all()
    print("\nDone.")
