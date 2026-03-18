import re
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    # Lowercase e remove caracteres não alfabéticos
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def build_vocab(texts: list[str], max_words: int = 10000) -> dict:
    # Conta as palavras mais comuns
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_words)
    # index 0 = PAD, index 1 = UNK
    word_index = {word: i + 2 for i, (word, _) in enumerate(most_common)}
    return word_index


def encode(text: str, word_index: dict, max_len: int = 200) -> torch.Tensor:
    # Encode de palavras em inteiros
    tokens = clean_text(text).split()
    sequence = [word_index.get(word, 1) for word in tokens]
    sequence = sequence[:max_len]
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    return torch.tensor(sequence, dtype=torch.long)

class TfidfDataset(Dataset): # Dataset com vetores TF-IDF

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        max_words: int = 2000,
        vectorizer: TfidfVectorizer | None = None,
        train: bool = True,
    ):
        self.texts = texts
        self.labels = labels

        if train:
            self.vectorizer = TfidfVectorizer(
                max_features=max_words,
                ngram_range=(1, 2),   # unigrams + bigrams
                sublinear_tf=True,    # log normalization of term frequencies
                min_df=2,             # ignore terms that appear in only 1 document
            )
            X = self.vectorizer.fit_transform(texts).toarray()
        else:
            assert vectorizer is not None, "Provide a fitted vectorizer when train=False"
            self.vectorizer = vectorizer
            X = self.vectorizer.transform(texts).toarray()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset): #Dataset com sequencias de inteiros

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        word_index: dict,
        max_len: int = 200,
    ):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.word_index, self.max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
