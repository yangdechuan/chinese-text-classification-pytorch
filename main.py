import os
import argparse
import configparser

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import make_vocab, get_vocab, load_data
from cnn import CNNTextModel

cfg = configparser.ConfigParser()
cfg.read("settings.ini", encoding="utf-8")

TRAIN_FILE = cfg["file"]["train_file"].replace("/", os.path.sep)
TEST_FILE = cfg["file"]["test_file"].replace("/", os.path.sep)
PREDICT_FILE = cfg["file"]["predict_file"].replace("/", os.path.sep)
MODEL_DIR = cfg["file"]["model_dir"].replace("/", os.path.sep)
RESULT_DIR = cfg["file"]["result_dir"].replace("/", os.path.sep)
TEXT_COL_NAME = cfg["file"]["text_col_name"]
LABEL_COL_NAME = cfg["file"]["label_col_name"]
CLASS_NAMES = eval(cfg["file"]["class_names"])

USE_CUDA = cfg["train"]["use_cuda"].lower() == "true"
BATCH_SIZE = int(cfg["train"]["batch_size"])
EPOCHS = int(cfg["train"]["epochs"])

MAX_LEN = int(cfg["process"]["max_sentence_len"])
MIN_COUNT = int(cfg["process"]["min_word_count"])

EMBEDDING_DIM = int(cfg["model"]["embedding_dim"])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Load data.
    print("Load data...")
    vocab2idx = get_vocab(result_dir=RESULT_DIR, min_count=MIN_COUNT)
    X_train, y_train = load_data(file=TRAIN_FILE,
                                 max_len=MAX_LEN,
                                 vocab2idx=vocab2idx,
                                 text_col_name=TEXT_COL_NAME,
                                 label_col_name=LABEL_COL_NAME,
                                 class_names=CLASS_NAMES)
    X_test, y_test = load_data(file=TEST_FILE,
                               max_len=MAX_LEN,
                               vocab2idx=vocab2idx,
                               text_col_name=TEXT_COL_NAME,
                               label_col_name=LABEL_COL_NAME,
                               class_names=CLASS_NAMES)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE)
    print("Load data success.")

    vocab_size = len(vocab2idx)

    # Build model.
    model = CNNTextModel(vocab_size=vocab_size,
                         embedding_dim=EMBEDDING_DIM,
                         num_classes=len(CLASS_NAMES))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_size = len(train_dataset)
    batch_num = data_size // BATCH_SIZE + 1

    for epoch in range(1, EPOCHS + 1):
        # Train model.
        model.train()
        batch = 1
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)  # (N, L)
            batch_ys = batch_ys.to(device)  # (N, )
            batch_out = model(batch_xs)  # (N, num_classes)
            loss = F.cross_entropy(batch_out, batch_ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch % 30 == 0:
            #     print("epoch {}, batch {}/{}, train loss {}".format(epoch, batch, batch_num, loss.item()))
            batch += 1
        checkpoint_path = os.path.join(MODEL_DIR, "model_epoch_{}.ckpt".format(epoch))
        torch.save(model, checkpoint_path)

        # Test model.
        model.eval()
        total = 0
        correct = 0
        for batch_xs, batch_ys in test_loader:
            batch_xs = batch_xs.to(device)  # (N, L)
            batch_ys = batch_ys.to(device)  # (N, )
            batch_out = model(batch_xs)  # (N, num_classes)
            batch_pred = batch_out.argmax(dim=-1)  # (N, )
            correct += (batch_ys == batch_pred).sum().item()
            total += batch_ys.shape[0]
        print("epoch {}, test accuracy {}%".format(epoch, correct / total * 100))


def predict(epoch_idx):
    """Load model in `models`and predict."""
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

    model = torch.load(os.path.join(MODEL_DIR, "model_epoch_{}.ckpt".format(epoch_idx)))
    model = model.to(device)

    vocab2idx = get_vocab(result_dir=RESULT_DIR, min_count=MIN_COUNT)
    X, _ = load_data(PREDICT_FILE,
                     max_len=MAX_LEN,
                     vocab2idx=vocab2idx,
                     text_col_name=TEXT_COL_NAME)
    X = torch.from_numpy(X).to(device)  # (N, L)
    out = model(X)  # (N, num_classes)
    pred = out.argmax(dim=-1)  # (N, )
    pred = pred.cpu().numpy()
    with open(os.path.join(RESULT_DIR, "predict.txt"), "w", encoding="utf-8") as fw:
        for label in pred:
            fw.write(str(label) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-vocab", action="store_true",
                        help="Set this flag if you want to make vocab from train data.")
    parser.add_argument("--do-train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do-predict", action="store_true",
                        help="Whether to run prediction.")
    parser.add_argument("--epoch-idx", type=int, default=1,
                        help="Choose which model to predict.")

    args = parser.parse_args()

    if args.make_vocab:
        make_vocab(train_file=TRAIN_FILE, result_dir=RESULT_DIR, text_col_name=TEXT_COL_NAME)
    if args.do_train:
        train()
    if args.do_predict:
        predict(args.epoch_idx)
