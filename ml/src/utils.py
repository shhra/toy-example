# Utils
import os
import pickle
import spacy
from sklearn.feature_extraction.text import CountVectorizer


def save_model(model, name):
    if not os.path.exists('./saved_models'):
        os.mkdir('./saved_models')
    pickle.dump(model, open('./saved_models/'+name, 'wb'))


def load_model(name):
    model_dir = './saved_models/'
    model = pickle.load(open(model_dir + name, 'rb'))
    return model


def infer(txt):
    nlp = spacy.load("en_core_web_sm")
    tokens = nlp(txt)
    sentence = ' '.join([x.text for x in tokens])
    vectorizer = load_model('vectorizer.sav')
    out = vectorizer.transform([sentence])
    model = load_model('sentence_classifier.sav')
    output = model.predict(out)
    encoder = load_model('encoder.sav')
    result = encoder.inverse_transform(output)
    return result[0]


if __name__ == '__main__':
    emotion = infer("WHAT AM I DOING WITH LIFE")
    print(emotion)

