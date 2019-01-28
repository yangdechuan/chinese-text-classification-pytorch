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


def make_vocab(train_file, result_dir="results", text_col_name=None):
    """Build vocab dict.
    Write vocab and num to results/vocab.txt

    Arguments:
        train_file: train data file path.
        result_dir: vocab dict directory.
        text_col_name: column name for text.
    """
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    dic_filepath = os.path.join(result_dir, "vocab.txt")

    if train_file[-4:] == ".csv":
        df = pd.read_csv(train_file)
    elif train_file[-5:] == ".xlsx":
        df = pd.read_excel(train_file)
    else:
        raise ValueError

    vocab2num = Counter()
    lengths = []
    for sentence in df[text_col_name]:
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

    print("All Vocab Size {}".format(len(vocab2num)))
    print("Train Data Size {}".format(len(lengths)))
    print("Average Sentence Length {}".format(sum(lengths) / len(lengths)))
    print("Max Sentence Length {}".format(max(lengths)))


def get_vocab(result_dir="results", min_count=1):
    with open(os.path.join(result_dir, "vocab.txt"), "r", encoding="utf-8") as fr:
        vocabs = [line.split()[0] for line in fr.readlines() if int(line.split()[1]) >= min_count]
    vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}
    print("used vocab size {}".format(len(vocab2idx)))
    return vocab2idx


def load_data(file, max_len=100,
              vocab2idx=None,
              text_col_name=None,
              label_col_name=None,
              class_names=None):
    """Load texts and labels for train or test.

    Arguments:
        file: data file path.
        max_len: Sequences longer than this will be filtered out, and shorter than this will be padded with PAD.
        vocab2idx: dict. e.g. {"你好": 1, "世界": 7, ...}
        text_col_name: column name for text.
        label_col_name: column name for label.
        class_names: list of label name.
    Returns:
        X: numpy array with shape (data_size, max_len)
        y: numpy array with shape (data_size, )
        vocab_size: a scalar
    """
    if file[-4:] == ".csv":
        df = pd.read_csv(file)
    elif file[-5:] == ".xlsx":
        df = pd.read_excel(file)
    else:
        raise ValueError

    x_list = []
    for sentence in df[text_col_name].values:
        sentence = _clean_str(sentence)
        x = [vocab2idx.get(vocab, UNK) for vocab in jieba.cut(sentence)]
        x = x[:max_len]
        n_pad = max_len - len(x)
        x = x + n_pad * [PAD]  # pad with zero
        x_list.append(x)
    X = np.array(x_list, dtype=np.int64)
    print("{} data size {}".format(file,  len(X)))

    if label_col_name:
        label2idx = {label: idx for idx, label in enumerate(class_names)}
        y = [label2idx[label] for label in df[label_col_name].values]
        y = np.array(y, dtype=np.int64)
    else:
        y = None

    return X, y
