import logging
import torch

from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm


def _token_iterator(tokens, ngrams, labels=None):
    yield_cls = False
    if labels:
        yield_cls = True
    for i, each in enumerate(tokens):
        if yield_cls:
            yield labels[i], ngrams_iterator(each, ngrams)
        else:
            yield ngrams_iterator(each, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK,
                                        [vocab[token] for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
        return data, set(labels)


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, data, labels):
        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def _setup_dataset(tokens, labels, ngrams=1, vocab=None, include_unk=False):
    if vocab is None:
        vocab = build_vocab_from_iterator(_token_iterator(
            tokens, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary isn not of type Vocab")
    logging.info(f'Vocab has {len(vocab)} entires')
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
            vocab, _token_iterator(tokens[:5000], ngrams, labels), include_unk)
    test_data, test_labels = _create_data_from_iterator(
            vocab, _token_iterator(tokens[5000:], ngrams, labels), include_unk)
    return(TextClassificationDataset(vocab, train_data, train_labels),
           TextClassificationDataset(vocab, test_data, test_labels))


def emotion_dataset(tokens, labels, ngrams, vocab=None, include_unk=False):
    return _setup_dataset(tokens, labels, ngrams, vocab, include_unk)


if __name__ == '__main__':
    from process import PreprocessText
    processor = PreprocessText("data/ISEAR.csv")
    merged_labels = processor.merged_labels()
    labels = processor.encode_labels(merged_labels)
    sentence = processor.data.sentence
    print("Starting tokenization")
    tokens = processor.tokenize(sentence)
    tokenized_sentence = processor.tokenized_sentence(tokens)
    train, test = emotion_dataset(tokenized_sentence.tolist(),
                              labels.tolist(), ngrams=2)
    dataset_iter = iter(train)
    print(next(dataset_iter))







