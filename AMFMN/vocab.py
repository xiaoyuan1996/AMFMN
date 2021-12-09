# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

import nltk
from collections import Counter
import argparse
import os
import json

annotations = {
    'coco_splits': ['train_caps.txt', 'val_caps.txt', 'test_caps.txt'],
    'flickr30k_splits': ['train_caps.txt', 'val_caps.txt', 'test_caps.txt'],
    'rsicd_precomp': ['train_caps.txt', 'val_caps.txt', 'test_caps.txt'],
    'rsitmd_precomp': ['train_caps.txt', 'val_caps.txt'],
    'ucm_precomp': ['train_caps.txt', 'val_caps.txt'],
    'sydney_precomp': ['train_caps.txt', 'val_caps.txt'],

    }


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, caption_file, threshold):
    """Build a simple vocabulary wrapper."""

    stopword_list = list(set(nltk.corpus.stopwords.words('english')))
    counter = Counter()
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        captions = from_txt(full_path)

        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
            tokens = [k for k in tokens if k not in punctuations]
            tokens = [k for k in tokens if k not in stopword_list]
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    vocab.add_word('<unk>')

    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, caption_file=annotations, threshold=5)
    serialize_vocab(vocab, 'vocab/%s_vocab.json' % data_name)
    print("Saved vocabulary file to ", 'vocab/%s_vocab.json' %(data_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--data_name', default='sydney_precomp',
                        help='{coco,f30k}')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)
