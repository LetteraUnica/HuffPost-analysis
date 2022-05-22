from tqdm import tnrange
from copy import deepcopy
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import pandas as pd


def documents_to_words(documents):
    return [word for document in documents for word in document.split(" ")]


def remove_short_words(documents, min_length=2):
    filtered_documents = []
    for document in documents:
        document = document.split(" ")
        document = [word for word in document if len(word) > min_length]
        document = " ".join(document)
        filtered_documents.append(document)
    return filtered_documents


def find_collocations(documents, min_word_length=2, min_bigram_freq=10):
    documents = remove_short_words(documents, min_word_length)
    words = documents_to_words(documents)
    bgm = BigramAssocMeasures()
    score = bgm.mi_like
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(min_bigram_freq)
    collocations = [[bigram, pmi]
                    for bigram, pmi in finder.score_ngrams(score)]
    return pd.DataFrame(collocations, columns=['bigram', 'pmi'])


def apply_collocations_to_sentence(sentence, set_colloc):
    for b1, b2 in set_colloc:
        sentence = sentence.replace("%s %s" % (b1, b2), "%s_%s" % (b1, b2))
    return sentence


def apply_collocations(documents, set_collocations):
    documents = deepcopy(documents)

    for i in tnrange(len(documents), desc="Documents processed"):
        documents[i] = apply_collocations_to_sentence(
            documents[i], set_collocations)

    return documents
