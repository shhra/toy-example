#!/usr/bin/env python
import pandas as pd
import spacy

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class PreprocessText:

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path,
                                names=["index", "emotion", "sentence"],
                                index_col="index")
        self.labels = self.data['emotion'].tolist()
        self.sentence = self.data['sentence']
        self.label_encoder = preprocessing.LabelEncoder()
        self.nlp = spacy.load("en_core_web_sm")

    def merged_labels(self):
        data = self.data.copy()
        data.emotion = self.data.emotion.apply(
                lambda x: "anger" if x == "disgust" else x)
        data.emotion = self.data.emotion.apply(
                lambda x: "guilt" if x == "shame" else x)
        return data.emotion

    def encode_labels(self, labels):
        classes = labels.unique()
        self.label_encoder.fit(classes)
        encoded_labels = self.label_encoder.transform(labels)
        return encoded_labels

    @staticmethod
    def filter_stop(doc):
        return [x.text for x in doc if not x.is_stop]

    def tokenize(self, sentence):
        tokenized = sentence.apply(self.nlp)
        filtered = tokenized.apply(self.filter_stop)
        return filtered

    def tokenized_sentence(self, tokens):
        data = self.data.copy()
        data['tokenized'] = tokens.apply(lambda x: ' '.join(x))
        return data['tokenized']

    def vectorize(self, sentence, flag='T'):
        if flag == 'T':
            vectorier = TfidfVectorizer(max_features=1500)
        elif flag == 'C':
            vectorier = CountVectorizer(max_features=1500)
        vectorized = vectorier.fit_transform(sentence)
        return vectorized


if __name__ == '__main__':
    processor = PreprocessText("ml/data/ISEAR.csv")
    merged_labels = processor.merged_labels()
    print(processor.data.emotion.value_counts())
    print(merged_labels.value_counts())
    print(processor.encode_labels())

