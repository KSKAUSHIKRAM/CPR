import importlib.resources
from symspellpy import SymSpell, Verbosity

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Locate the dictionary file inside the symspellpy package
dict_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)

# Add a custom word
sym_spell.create_dictionary_entry("SymSpell", 1)

# Single-word correction
input_term = "Interet"
suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST, max_edit_distance=2)
for suggestion in suggestions:
    print(suggestion.term, suggestion.distance, suggestion.count)

# Sentence correction
input_term = "i wann wate"
suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=2)
for suggestion in suggestions:
    print(suggestion.term)
