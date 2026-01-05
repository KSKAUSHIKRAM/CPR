# ------------------------------------------------------------
# Phonetic + Spelling Autocorrect Tester
# ------------------------------------------------------------
# Usage:
#   python phonetic_autocorrect_test.py
#
# Dependencies:
#   pip install textdistance
# ------------------------------------------------------------

from difflib import get_close_matches
from textdistance import soundex, levenshtein

# ------------------------------------------------------------
# Sample dataset vocabulary (replace with your AAC labels)
# ------------------------------------------------------------
dataset_vocab = {
    "want", "water", "food", "apple", "juice", "dosa", "drink",
    "bathroom", "toilet", "walk", "outside", "help", "teacher",
    "happy", "angry", "hungry", "sleep", "wake", "ball", "cat",
    "dog", "ant"
}

# ------------------------------------------------------------
# Autocorrect with phonetic fallback
# ------------------------------------------------------------
def autocorrect_with_dataset(text, vocab):
    """
    Performs lightweight spelling + phonetic correction.
    - Uses difflib for spelling similarity
    - Uses Soundex + Levenshtein for phonetic similarity
    """
    if not text:
        return text

    words = text.lower().split()
    corrected = []

    for w in words:
        # Exact match → keep as is
        if w in vocab:
            corrected.append(w)
            continue

        # Step 1: Edit-distance (difflib)
        match = get_close_matches(w, vocab, n=1, cutoff=0.75)

        # Step 2: Phonetic fallback (Soundex + Levenshtein)
        if not match:
            try:
                w_code = soundex(w)
                best_word, best_score = None, 0
                for cand in vocab:
                    score = levenshtein.normalized_similarity(soundex(cand), w_code)
                    if score > best_score:
                        best_word, best_score = cand, score
                if best_score > 0.6:  # threshold
                    match = [best_word]
            except Exception as e:
                print(f"[PHONETIC] error for '{w}': {e}")

        corrected.append(match[0] if match else w)

    corrected_text = " ".join(corrected)
    print(f"[INPUT] {text} → [CORRECTED] {corrected_text}")
    return corrected_text


# ------------------------------------------------------------
# Test phrases
# ------------------------------------------------------------
test_inputs = [
    "i wan wat",
    "ai wan wataa",
    "i wnt foood",
    "wan juis",
    "i want apple",
    "i wan dosa",
    "hepl me",
    "i ned slep"
]

print("🔹 Testing Phonetic + Spelling Autocorrection\n")
for t in test_inputs:
    autocorrect_with_dataset(t, dataset_vocab)
