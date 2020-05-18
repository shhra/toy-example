# Utils
import os
import pickle
import spacy


def save_model(model, name):
    if not os.path.exists('./saved_models'):
        os.mkdir('./saved_models')
    pickle.dump(model, open('./saved_models/'+name, 'wb'))


def load_model(name):
    model_dir = './saved_models/'
    model = pickle.load(open(model_dir + name, 'rb'))
    return model


class Inference:
    def __init__(self):
        self.vectorizer = load_model('vectorizer.sav')
        self.model = load_model('sentence_classifier.sav')
        self.encoder = load_model('encoder.sav')

    def infer(self, txt):
        nlp = spacy.load("en_core_web_sm")
        tokens = nlp(txt)
        sentence = ' '.join([x.text for x in tokens])
        out = self.vectorizer.transform([sentence])
        output = self.model.predict(out)
        result = self.encoder.inverse_transform(output)
        return result[0]


if __name__ == '__main__':
    inf = Inference()
    emotion = inf.infer("WHAT AM I DOING WITH LIFE")
    print(emotion)

