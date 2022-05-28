from sklearn.metrics import f1_score, plot_confusion_matrix
import time
import re
from tqdm import tnrange
from copy import deepcopy
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import pandas as pd
import numpy as np


def documents_to_words(documents):
    all_words = []
    for i, document in enumerate(documents):
        try:
            words = document.split(" ")
            all_words.extend(words)
        except:
            pass
    return all_words


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


def show_topics(A, vocabulary, topn=5):
    """
    find the top N words for each of the latent dimensions (=rows) in a
    """
    topic_words = ([[vocabulary[i] for i in np.argsort(t)[:-topn-1:-1]]
                    for t in A])  # for each row
    return [', '.join(t) for t in topic_words]


def get_topic_descriptors(lda_model, num_words=3):
    descriptors = []
    for topic in lda_model.print_topics(num_words=num_words):
        topic_words = []
        for word in topic[1].split("+"):
            topic_words.append(
                re.sub(r"0\.[0-9]+\*", '', word).replace('"', '').strip())

        descriptors.append(", ".join(topic_words))

    return descriptors


def get_f1_scores(y_test, y_pred):
    print(f"F1 macro: {f1_score(y_test, y_pred, average='macro'):.2f}")
    print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted'):.2f}")


def get_model_report(model, X_test, y_test, confusion_matrix=False):
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    print(
        f"Model took {end-start:.2f} seconds to predict {len(y_pred)} documents")
    print(f"Time per document: {(end-start)/len(y_pred)*1e3:.3f} ms")
    get_f1_scores(y_test, y_pred)

    if confusion_matrix:
        plot_confusion_matrix(model, X_test, y_test, normalize="true",
                              xticks_rotation=90, include_values=False)
