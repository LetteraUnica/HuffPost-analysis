import regex as re
from nltk import SnowballStemmer
import string
from copy import deepcopy
from typing import Sequence
from nltk.corpus import stopwords as stopword_list
from tqdm.notebook import tqdm


class DocumentCleaner:
    def __init__(self, lowercase=True, numbers=True, punctuation=True, hashtags=True, stemming=True, stopwords=True):
        self.lowercase = lowercase
        self.numbers = numbers
        self.punctuation = punctuation
        self.hashtags = hashtags
        self.stemming = stemming
        if stopwords:
            self.stopwords = stopword_list.words('english')

    def remove_case(self, documents):
        return [document.lower() for document in documents]

    def remove_numbers(self, documents):
        return [re.sub('[0-9]', '0', document) for document in documents]

    def remove_punctuation(self, documents):
        punctuation = string.punctuation.replace("$", "")
        punctuation = string.punctuation.replace("%", "")
        return [re.sub(f'[{punctuation}]', '', document) for document in documents]

    def remove_hashtags(self, documents):
        return [re.sub('#[a-z0-9_]+', '$hashtag', document) for document in documents]

    def stem(self, documents):
        stemmer = SnowballStemmer("english")
        stemmed_documents = []
        for _, document in tqdm(enumerate(documents), total=len(documents), desc="Stemming documents: "):
            document = document.split(" ")
            document = [stemmer.stem(word) for word in document]
            document = " ".join(document)
            stemmed_documents.append(document)
        return stemmed_documents

    def remove_stopwords(self, documents):
        filtered_documents = []
        for _, document in tqdm(enumerate(documents), total=len(documents), desc="Removing stopwords: "):
            document = document.split(" ")
            document = [
                word for word in document if word not in self.stopwords]
            document = " ".join(document)
            filtered_documents.append(document)
        return filtered_documents

    def clean(self, documents: Sequence):
        documents = deepcopy(documents)
        if self.lowercase:
            documents = self.remove_case(documents)
        if self.numbers:
            documents = self.remove_numbers(documents)
        if self.hashtags:
            documents = self.remove_hashtags(documents)
        if self.punctuation:
            documents = self.remove_punctuation(documents)
        if self.stemming:
            documents = self.stem(documents)
        if self.stopwords:
            documents = self.remove_stopwords(documents)

        return documents
