from process import PreprocessText
from models import NBClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_token_nb():
    processor = PreprocessText("data/ISEAR.csv")
    merged_labels = processor.merged_labels()
    sentence = processor.data.sentence
    print("Starting tokenization")
    tokens = processor.tokenize(sentence)
    print("Training classifier")
    classifier = NBClassifier(tokens, merged_labels)
    trained = classifier.train()
    classifier.test(trained)


def train_sentence_nb(test=False):
    processor = PreprocessText("data/ISEAR.csv")
    merged_labels = processor.merged_labels()
    sentence = processor.data.sentence
    print("Starting tokenization")
    tokens = processor.tokenize(sentence)
    tokenized_sentence = processor.tokenized_sentence(tokens)
    vectorized = processor.vectorize(tokenized_sentence, 'C')
    train_x, test_x, train_y, test_y = train_test_split(
            vectorized,
            merged_labels,
            test_size=0.35,
            random_state=44)

    print("Training classifier")
    classifier = MultinomialNB()
    classifier.fit(train_x, train_y)

    if test:
        y_pred = classifier.predict(test_x)
        print(accuracy_score(test_y, y_pred))


if __name__ == '__main__':
    train_sentence_nb(test=True)
