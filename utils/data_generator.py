import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

class SentenceGenerator:
    """
    This class is responsible for generating the new sentences as part of creating the new augmented dataset.
    The new sentences are created using the synonym replacement technique.
    For every given word of a sentence, a list of synonyms is fetched from Wordnet, and with each word in the list, a new sentence is generated.
    """
    def __init__(self):
        pass
    
    def get_eng_synonyms(self, word):
        """
        Given an English word, this function returns the list of all synonyms of it by checking the Wordnet.
        Args:
            word (str): English word whose synonyms are to looked.

        Returns:
            synonym (List): List of all synonyms of a requested word
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if len(nltk.word_tokenize(lemma.name())) == 1:
                    synonyms.add(lemma.name())
        return list(synonyms)
    
    def get_esp_synonyms(self, word):
        """
        Given a Spanish word, this function returns the list of all synonyms of it by checking OMW (Open Multilingual Wordnet).
        Args:
            word (str): Spanish word whose synonyms are to looked.

        Returns:
            synonym (List): List of all synonyms of a requested word
        """
        synonyms = set()
        for syn in wordnet.synsets(word, lang='spa'):
            for lemma in syn.lemma_names('spa'):
                if len(nltk.word_tokenize(lemma)) == 1:
                    synonyms.add(lemma)
        return list(synonyms)

    def get_synonyms(self, token, lang):
        """
        Wrapper function to choose the right synonym fetch function.

        Args:
            token (str): Token (word) whose synonyms are to be searched.
            lang (str): Language of the token (i.e., either English or Spanish).
        """
        if lang == 'lang2':                         
            return self.get_esp_synonyms(token)
        return self.get_eng_synonyms(token)
    
    def generate_new_sentence(self, tokens, language):
        """
        This function is responsible for returning the list of all possible sentences generated from synonym replacement 
        for a given sentece.
        Args:
            tokens (List): List of all tokens pertaining to a sentence
            language (str): Language of the sentence 

        Returns:
           new_sentences (List): List of all the possible sentences generated after the synonym replacement.
        """
        new_sentences = [''.join(tokens)]
        for i, token in enumerate(tokens):
            synonyms = self.get_synonyms(token, language)
            if synonyms:
                # Generate a new sentence for each synonym
                new_sentences_temp = []
                for synonym in synonyms:
                    if synonym.lower() != token.lower():
                        new_sentence = list(tokens)
                        new_sentence[i] = synonym
                        new_sentences_temp.append(' '.join(new_sentence))
                new_sentences.extend(new_sentences_temp)
        return new_sentences
    