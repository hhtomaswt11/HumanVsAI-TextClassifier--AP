import torch
import torch.nn as nn


def _masked_mean_pool(sequence_output: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    mask = (tokens != 0).unsqueeze(-1).float()
    summed = (sequence_output * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return summed / lengths

class BaselineClassifier(nn.Module):
    # input_dim  : numero de features de input (TF-IDF)
    # n_classes  : numero de classes de output (5)


    def __init__(self, input_dim: int, n_classes: int = 5):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # raw logits; use CrossEntropyLoss


class DNNClassifier(nn.Module):
    # input_dim  : numero de features de input (TF-IDF)
    # n_classes  : numero de classes de output (5)
    # topology   : lista de tamanhos de camadas escondidas, por exemplo [128, 64].
    # dropout    : probabilidade de dropout aplicada após cada camada ReLU (0 = sem dropout).

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 5,
        topology: list[int] = [64],
        dropout: float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = []

        prev_dim = input_dim
        for hidden_dim in topology:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))  # output logits

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits; use CrossEntropyLoss


class EmbeddingClassifier(nn.Module):
    # vocab_size  : tamanho do vocabulario (incluindo PAD no indice 0).
    # embed_dim   : dimensao da embedding.
    # n_classes   : numero de classes de output (5).
    # dropout     : probabilidade de dropout aplicada após a embedding (0 = off).

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        embedded = self.embedding(x)         # (B, L, E)
        pooled = embedded.mean(dim=1)        # (B, E)
        pooled = self.dropout(pooled)
        return self.fc(pooled)               # (B, n_classes)


class RNNClassifier(nn.Module):
    # vocab_size  : tamanho do vocabulario (incluindo PAD no indice 0).
    # embed_dim   : dimensao da embedding.
    # hidden_dim  : dimensao do estado oculto do RNN.
    # n_classes   : numero de classes de output (5).
    # num_layers  : numero de camadas RNN empilhadas.
    # dropout     : dropout na saida do estado oculto final.

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_classes: int = 5,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)                     # (B, L, E)
        _, hidden = self.rnn(embedded)                   # hidden: (num_layers, B, H)
        last_hidden = self.dropout(hidden[-1])           # (B, H)
        return self.fc(last_hidden)                      # (B, n_classes)


class LSTMClassifier(nn.Module):
    # vocab_size  : tamanho do vocabulario (incluindo PAD no indice 0).
    # embed_dim   : dimensao da embedding.
    # hidden_dim  : dimensao do estado oculto do RNN.
    # n_classes   : numero de classes de output (5).
    # num_layers  : numero de camadas RNN empilhadas.
    # dropout     : dropout na saida do estado oculto final.
    # bidirectional  : se True, usa um LSTM bidireccional.

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_classes: int = 5,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, n_classes)
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)                           # (B, L, E)
        output, _ = self.lstm(embedded)                        # output: (B, L, H*dirs)
        pooled = _masked_mean_pool(output, x)                  # (B, H*dirs)
        pooled = self.dropout(pooled)
        return self.fc(pooled)                                 # (B, n_classes)


class GRUClassifier(nn.Module):
    # vocab_size  : tamanho do vocabulario (incluindo PAD no indice 0).
    # embed_dim   : dimensao da embedding.
    # hidden_dim  : dimensao do estado oculto do GRU.
    # n_classes   : numero de classes de output (5).
    # num_layers  : numero de camadas GRU empilhadas.
    # dropout     : dropout na saida do estado oculto final.
    # bidirectional  : se True, usa um GRU bidireccional.

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_classes: int = 5,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, n_classes)
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)              # (B, L, E)
        output, _ = self.gru(embedded)            # output: (B, L, H*dirs)
        pooled = _masked_mean_pool(output, x)     # (B, H*dirs)
        pooled = self.dropout(pooled)
        return self.fc(pooled)                    # (B, n_classes)
