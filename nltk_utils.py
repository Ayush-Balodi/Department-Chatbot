import numpy as np 
import nltk
# nltk.download('punkt)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer
def tokenize(sentence):
    """.git\split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = find the root form of the word
    exaples:
    words = ["orgeanize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [0, 1, 0, 1, 0, 0, 0] 
    """

    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # intialize bag with 0 or each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in wnumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag