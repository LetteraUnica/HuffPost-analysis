from sklearn.feature_extraction.text import CountVectorizer
from copy import deepcopy


class MyTokenizer:
    def __init__(self, lowercase=True, min_df=0.001, max_df=1.):
        self.count_vectorizer = CountVectorizer(lowercase=lowercase,
                                                min_df=min_df,
                                                max_df=max_df,
                                                ngram_range=(1, 1))

    def convert2ints(self, document):
        res = []
        for word in document.split(" "):
            word = word.strip()
            if word in self.word2int:
                res.append(self.word2int[word])
            else:
                res.append(self.word2int[self.UNK])
        return res

    def fit(self, X):
        self.count_vectorizer.fit(X)
        self.PAD = '-PAD-'
        self.UNK = '-UNK-'
        word2int = deepcopy(self.count_vectorizer.vocabulary_)
        # sum 2 to the values
        word2int = {k: v+2 for k, v in word2int.items()}
        word2int[self.PAD] = 0  # special token for padding
        word2int[self.UNK] = 1  # special token for unknown words

        self.word2int = word2int

    def transform(self, X):
        return [self.convert2ints(document) for document in X]

    def get_vocab_size(self):
        return len(self.word2int)


def get_next_multiple_of_eight(number):
    return ((number - 1) // 8 + 1) * 8


def pad_to_max_length(tokenized_documents, padding_token):
    max_length = max([len(document) for document in tokenized_documents])
    max_length = get_next_multiple_of_eight(max_length)
    return [document + [padding_token]*(max_length-len(document)) for document in tokenized_documents]
