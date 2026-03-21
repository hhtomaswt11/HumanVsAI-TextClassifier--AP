import torch
from torch.utils.data import Dataset

# Reutilizar funções de texto do pytorch_models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pytorch_models.dataset import clean_text, build_vocab, encode


class TransformerSequenceDataset(Dataset):
    """
    Dataset com sequências de inteiros para o TransformerClassifier manual.
    Reutiliza clean_text, build_vocab, encode do pytorch_models.
    """

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


class BERTDataset(Dataset):
    """
    Dataset para fine-tuning de BERT.
    Usa o BertTokenizer do HuggingFace para tokenização.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
