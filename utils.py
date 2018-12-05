import os
from collections import Counter

import jieba
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

PAD = 0
UNK = 1



def _clean_str(string):
    string = string.replace(",", "，")
    return string.strip()


def make_data(train_file, result_dir="results"):
    """Build vocab dict.
    Write vocab and num to results/vocab.txt

    Arguments:
        train_file: train data file path.
        result_dir: vocab dict directory
    """
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    dic_filepath = os.path.join(result_dir, "vocab.txt")

    df = pd.read_csv(train_file)

    vocab2num = Counter()
    lengths = []
    for sentence in df["text"]:
        if sentence is None:
            continue
        sentence = _clean_str(sentence)
        vocabs = jieba.lcut(sentence.strip())
        lengths.append(len(vocabs))
        for vocab in vocabs:
            vocab = vocab.strip()
            if vocab and vocab != "":
                vocab2num[vocab] += 1
    with open(dic_filepath, "w", encoding="utf-8") as fw:
        fw.write("{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>"))
        for vocab, num in vocab2num.most_common():
            fw.write("{}\t{}\n".format(vocab, num))

    print("Vocab Size {}".format(len(vocab2num)))
    print("Train Data Size {}".format(len(lengths)))
    print("Average Sentence Length {}".format(sum(lengths) / len(lengths)))
    print("Max Sentence Length {}".format(max(lengths)))


def load_data(file, max_len=100, min_count=10, result_dir="results"):
    """Load texts and labels for train or test.

    Arguments:
        file: data file path.
        max_len: Sequences longer than this will be filtered out, and shorter than this will be padded with PAD.
        min_count: Vocab num less than this will be replaced with UNK.
        result_dir: vocab dict dir
    Returns:
        X: numpy array with shape (data_size, max_len)
        y: numpy array with shape (data_size, )
        vocab_size: a scalar
    """
    with open(os.path.join(result_dir, "vocab.txt"), "r", encoding="utf-8") as fr:
        vocabs = [line.split()[0] for line in fr.readlines() if int(line.split()[1]) >= min_count]
    vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}
    vocab_size = len(vocabs)
    df = pd.read_csv(file)
    x_list = []
    for sentence in df["text"].values:
        sentence = _clean_str(sentence)
        x = [vocab2idx.get(vocab, UNK) for vocab in jieba.cut(sentence)]
        x = x[:max_len]
        n_pad = max_len - len(x)
        x = x + n_pad * [PAD]  # pad with zero
        x_list.append(x)
    X = np.array(x_list, dtype=np.int64)
    print("{} Data size {}".format("Train" if "train" in file else "Test",  len(X)))

    y = df["label"].values.astype(np.int64) if "label" in df.columns else None
    return X, y, vocab_size


class CustomDataset(Dataset):
    """Custom Dataset for PyTorch."""
    def __init__(self, file, max_len=100, min_count=10, result_dir="results"):
        super(CustomDataset, self).__init__()
        self.X, self.y, self.vocab_size = load_data(file,
                                                    max_len=max_len,
                                                    min_count=min_count,
                                                    result_dir=result_dir)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]

    def get_vocab_size(self):
        return self.vocab_size
