import torch
import torch.nn as nn
import math


class TransformerClassifier(nn.Module):
    """
    Transformer manual para classificação de texto (5 classes).
    Usa embedding + positional embedding + TransformerEncoder + classificador linear.

    Parâmetros:
        vocab_size  : tamanho do vocabulário (incluindo PAD no índice 0).
        embed_dim   : dimensão da embedding.
        num_heads   : número de cabeças de atenção.
        ff_dim      : dimensão da camada feed-forward interna.
        num_layers  : número de camadas TransformerEncoder empilhadas.
        max_len     : comprimento máximo da sequência.
        n_classes   : número de classes de output (5).
        dropout     : probabilidade de dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        max_len: int = 200,
        n_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Token embedding + positional embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classificador
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) — sequência de índices inteiros
        retorna: (B, n_classes) — logits
        """
        batch_size, seq_len = x.shape

        # Criar máscara de padding (True onde é PAD=0)
        padding_mask = (x == 0)  # (B, L)

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Global average pooling (ignora posições de padding)
        mask = (~padding_mask).unsqueeze(-1).float()  # (B, L, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, E)

        # Classificação
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)  # (B, n_classes)


class BERTClassifier(nn.Module):
    """
    Fine-tuning de BERT pré-treinado para classificação de texto (5 classes).
    Usa o token [CLS] como representação da sequência.

    Parâmetros:
        model_name  : nome do modelo pré-treinado (default: "bert-base-uncased").
        n_classes   : número de classes de output (5).
        dropout     : probabilidade de dropout antes do classificador.
        freeze_bert : se True, congela os pesos do BERT (treina apenas o classificador).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        n_classes: int = 5,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768 para bert-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        input_ids: (B, L) — tokens BERT
        attention_mask: (B, L) — máscara de atenção (1=real, 0=pad)
        retorna: (B, n_classes) — logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)
