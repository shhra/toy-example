from nltk import everygrams
from nltk.classify import NaiveBayesClassifier, util


class NBClassifier:

    def __init__(self, features, labels, n=2):
        self.features = features.tolist()
        self.labels = labels.tolist()
        self.n = n
        self.length = features.shape[0]
        self.train_set, self.test_set = self.create_train_test(self.n)

    def create_ngram_features(self, words, n=2):
        ngram_vocab = everygrams(words, 1, n)
        my_dict = dict([(ng, True) for ng in ngram_vocab])
        return my_dict

    def create_train_test(self, n):
        pos_data = []
        for i in range(self.length):
            pos_data.append(
                    (self.create_ngram_features(self.features[i], n),
                        self.labels[i]))
        train_set = pos_data[:5000]
        test_set = pos_data[5000:]
        return train_set, test_set

    def train(self):
        classifier = NaiveBayesClassifier.train(self.train_set)
        return classifier

    def test(self, classifier):
        accuracy = util.accuracy(classifier, self.test_set)
        print(str(self.n)+'-gram accuracy:', accuracy)

