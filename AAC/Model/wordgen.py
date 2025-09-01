from nltk.corpus import wordnet
class Wordgen:
    def __init__(self,word):
        self.word=word
    def singularize(self):    
        if self.word.endswith('s'):
            return self.word[:-1]
        return self.word
    def find_closest_word(self):
        synonyms = set()
        hypernyms = set()

        # Get synonyms from WordNet
        for syn in wordnet.synsets(self.word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))  # Convert underscores to spaces

            # Get hypernyms (more general words)
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    hypernyms.add(lemma.name().replace("_", " "))

        #print(f"🔎 Synonyms for '{word}': {synonyms}")
        #print(f"🔎 Hypernyms for '{word}': {hypernyms}")

        # Check in order: original word -> singular form -> synonyms -> hypernyms
        possible_words = [self.word, self.singularize(self.word)] + list(synonyms) + list(hypernyms)
        if (len(possible_words)==0):
            return self.word
        return possible_words