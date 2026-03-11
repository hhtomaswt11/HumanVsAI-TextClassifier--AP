import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import TfidfDataset, SequenceDataset, clean_text, build_vocab
from models import BaselineClassifier, DNNClassifier, LSTMClassifier, GRUClassifier
from train import train, evaluate, predict, DEVICE

DATA_PATH = "../../data/dataset_limpo.csv"
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Hiperparametros
TFIDF_MAX_WORDS = 2000
MAX_LEN = 150          # sequence length for RNNs
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
PATIENCE = 10
VAL_SIZE = 0.1         
TEST_SIZE = 0.2
RANDOM_STATE = 42

CLASSES = ["Google", "Human", "Meta", "Mistral", "OpenAI"]
N_CLASSES = len(CLASSES)


# Data Loading 

def load_data(path: str):
    df = pd.read_csv(path, sep=";")
    df = df.dropna(subset=["Text", "Label"])

    label2idx = {cls: i for i, cls in enumerate(CLASSES)}
    labels = df["Label"].map(label2idx).tolist()
    texts = df["Text"].tolist()
    return texts, labels


def split_data(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# TF-IDF Pipeline 

def build_tfidf_loaders(X_train, X_val, X_test, y_train, y_val, y_test):
    train_ds = TfidfDataset(X_train, y_train, max_words=TFIDF_MAX_WORDS, train=True)
    val_ds   = TfidfDataset(X_val,   y_val,   max_words=TFIDF_MAX_WORDS,
                             vectorizer=train_ds.vectorizer, train=False)
    test_ds  = TfidfDataset(X_test,  y_test,  max_words=TFIDF_MAX_WORDS,
                             vectorizer=train_ds.vectorizer, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


# Sequence Pipeline 

def build_sequence_loaders(X_train, X_val, X_test, y_train, y_val, y_test):
    clean_train = [clean_text(t) for t in X_train]
    word_index = build_vocab(clean_train, max_words=10000)
    vocab_size = len(word_index) + 2  # +2 for PAD (0) and UNK (1)

    train_ds = SequenceDataset(X_train, y_train, word_index, max_len=MAX_LEN)
    val_ds   = SequenceDataset(X_val,   y_val,   word_index, max_len=MAX_LEN)
    test_ds  = SequenceDataset(X_test,  y_test,  word_index, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, vocab_size


# Reporting 

def report(model, test_loader, name="Model"):
    preds = predict(model, test_loader).numpy()
    true_labels = [y.item() for _, y in test_loader.dataset]

    print(f"\n{'='*50}")
    print(f" {name} — Test Results")
    print(f"{'='*50}")
    print(classification_report(true_labels, preds, target_names=CLASSES))

    _, acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {acc*100:.2f}%")
    return acc


def plot_history(history: dict, title: str = "Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{title.replace(' ', '_')}.png"))
    plt.show()


# Main 

def main():
    print(f"Using device: {DEVICE}\n")
    texts, labels = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    criterion = nn.CrossEntropyLoss()

    # 1. Baseline: Logistic Regression (TF-IDF)
    print("\n--- Baseline: Logistic Regression ---")
    train_loader, val_loader, test_loader = build_tfidf_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    baseline = BaselineClassifier(input_dim=TFIDF_MAX_WORDS, n_classes=N_CLASSES)
    history_base = train(baseline, train_loader, val_loader, criterion,
                         epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=True)
    report(baseline, test_loader, name="Baseline (LogReg + TF-IDF)")
    torch.save(baseline.state_dict(), os.path.join(MODELS_DIR, "baseline.pt"))
    plot_history(history_base, "Baseline Training History")

    # 2. DNN (TF-IDF)
    print("\n--- DNN (TF-IDF) ---")
    dnn = DNNClassifier(
        input_dim=TFIDF_MAX_WORDS, n_classes=N_CLASSES,
        topology=[256, 128, 64], dropout=0.3
    )
    history_dnn = train(dnn, train_loader, val_loader, criterion,
                        epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=True)
    report(dnn, test_loader, name="DNN (TF-IDF)")
    torch.save(dnn.state_dict(), os.path.join(MODELS_DIR, "dnn_tfidf.pt"))
    plot_history(history_dnn, "DNN TF-IDF Training History")

    # 3. LSTM (Embeddings)
    print("\n--- LSTM (Embeddings) ---")
    train_loader_seq, val_loader_seq, test_loader_seq, vocab_size = \
        build_sequence_loaders(X_train, X_val, X_test, y_train, y_val, y_test)

    lstm = LSTMClassifier(
        vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
        n_classes=N_CLASSES, num_layers=2, dropout=0.3, bidirectional=True
    )
    history_lstm = train(lstm, train_loader_seq, val_loader_seq, criterion,
                         epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=True)
    report(lstm, test_loader_seq, name="BiLSTM (Embeddings)")
    torch.save(lstm.state_dict(), os.path.join(MODELS_DIR, "bilstm.pt"))
    plot_history(history_lstm, "BiLSTM Training History")

    # 4. GRU (Embeddings)
    print("\n--- GRU (Embeddings) ---")
    gru = GRUClassifier(
        vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
        n_classes=N_CLASSES, num_layers=2, dropout=0.3, bidirectional=True
    )
    history_gru = train(gru, train_loader_seq, val_loader_seq, criterion,
                        epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=True)
    report(gru, test_loader_seq, name="BiGRU (Embeddings)")
    torch.save(gru.state_dict(), os.path.join(MODELS_DIR, "bigru.pt"))
    plot_history(history_gru, "BiGRU Training History")


if __name__ == "__main__":
    main()
