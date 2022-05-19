from sklearn.feature_extraction.text import CountVectorizer
from copy import deepcopy
class MyTokenizer:
    def __init__(self, max_length, lowercase=True, min_df=0.001, max_df=1.):
        self.count_vectorizer = CountVectorizer(lowercase=lowercase,
                                                min_df=min_df, 
                                                max_df=max_df, 
                                                ngram_range=(1, 1))

        self.max_length = max_length

    def convert2ints(self, document, word2int):
        res = []
        for word in document.split(" "):
            word = word.strip()
            if word in word2int:
                res.append(word2int[word])
            else:
                res.append(word2int[self.UNK])
        return res

    def pad_sequence(self, document, max_length):
        res = []
        for i in range(max_length):
            if i < len(document):
                res.append(document[i])
            else:
                res.append(self.word2int[self.PAD])
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
        X = [self.convert2ints(document, self.word2int) for document in X]
        X = [self.pad_sequence(document, self.max_length) for document in X]
        return X

    def get_vocab_size(self):
        return len(self.word2int)