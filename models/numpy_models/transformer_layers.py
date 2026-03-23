#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer layers implemented purely with NumPy.

Designed to be imported by the numpy_transformers_models.ipynb notebook.
All forward passes, backward passes, and weight updates use only NumPy.
"""

import re
import copy
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Maps words to integer indices and pads/truncates sequences to a fixed
    length.  Two special tokens are reserved:
      0 → <PAD>   (used to fill shorter sequences)
      1 → <UNK>   (used for out-of-vocabulary words at inference time)
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_vocab: int = 5000, max_seq_len: int = 128):
        self.max_vocab   = max_vocab
        self.max_seq_len = max_seq_len
        self.word2idx: dict = {}
        self.vocab_size: int = 0

    # ------------------------------------------------------------------
    def _tokenize(self, text: str):
        return re.findall(r"[A-Za-z0-9']+", str(text).lower())

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        vocab = ["<PAD>", "<UNK>"] + [
            w for w, _ in counter.most_common(self.max_vocab - 2)
        ]
        self.word2idx  = {w: i for i, w in enumerate(vocab)}
        self.vocab_size = len(self.word2idx)
        return self

    def transform(self, texts):
        out = np.zeros((len(texts), self.max_seq_len), dtype=np.int32)
        for i, text in enumerate(texts):
            tokens  = self._tokenize(text)[: self.max_seq_len]
            indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
            out[i, : len(indices)] = indices
        return out

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


# ---------------------------------------------------------------------------
# Embedding look-up table
# ---------------------------------------------------------------------------

class EmbeddingLayer:
    """Trainable word-embedding table."""

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.weights    = None
        self._opt       = None

    def initialize(self, optimizer):
        scale        = 0.1
        self.weights = np.random.uniform(
            -scale, scale, (self.vocab_size, self.embed_dim)
        )
        self._opt = copy.deepcopy(optimizer)

    def forward(self, indices, training: bool = True):
        """indices: (B, L)  →  output: (B, L, E)"""
        self._indices = indices
        return self.weights[indices]

    def backward(self, grad):
        """grad: (B, L, E) — accumulates gradients into embedding weights."""
        d_w = np.zeros_like(self.weights)
        np.add.at(d_w, self._indices, grad)
        self.weights = self._opt.update(self.weights, d_w)
        # Integer indices have no gradient to propagate further
        return None


# ---------------------------------------------------------------------------
# Fixed sinusoidal positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding:
    """Adds fixed sinusoidal position signals to token embeddings."""

    def __init__(self, max_seq_len: int, embed_dim: int):
        positions = np.arange(max_seq_len)[:, np.newaxis]        # (L_max, 1)
        dims      = np.arange(embed_dim)[np.newaxis, :]           # (1, E)
        angles    = positions / np.power(10_000, (2 * (dims // 2)) / embed_dim)
        encoding  = np.where(dims % 2 == 0, np.sin(angles), np.cos(angles))
        self._enc = encoding                                       # (L_max, E)

    def forward(self, x, training: bool = True):
        """x: (B, L, E)  →  (B, L, E)"""
        seq_len = x.shape[1]
        return x + self._enc[:seq_len]

    def backward(self, grad):
        """Positional encoding is fixed — gradient passes through unchanged."""
        return grad


# ---------------------------------------------------------------------------
# Layer Normalisation
# ---------------------------------------------------------------------------

class LayerNorm:
    """Layer normalisation over the last (feature) dimension."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps     = eps
        self.gamma   = np.ones(d_model)
        self.beta    = np.zeros(d_model)
        self._g_opt  = None
        self._b_opt  = None

    def initialize(self, optimizer):
        self._g_opt = copy.deepcopy(optimizer)
        self._b_opt = copy.deepcopy(optimizer)

    def forward(self, x, training: bool = True):
        """x: (..., d_model)  →  same shape."""
        self._x      = x
        self._mean   = x.mean(axis=-1, keepdims=True)
        self._var    = x.var(axis=-1, keepdims=True)
        self._x_norm = (x - self._mean) / np.sqrt(self._var + self.eps)
        return self.gamma * self._x_norm + self.beta

    def backward(self, grad):
        """grad: (..., d_model)  →  dL/dx, same shape."""
        N            = self.d_model
        x_norm       = self._x_norm
        reduce_axes  = tuple(range(grad.ndim - 1))

        d_gamma = np.sum(grad * x_norm, axis=reduce_axes)
        d_beta  = np.sum(grad,          axis=reduce_axes)

        self.gamma = self._g_opt.update(self.gamma, d_gamma)
        self.beta  = self._b_opt.update(self.beta,  d_beta)

        # Gradient through the normalisation step
        d_xn    = grad * self.gamma
        std_inv = 1.0 / np.sqrt(self._var + self.eps)
        d_x = std_inv / N * (
            N * d_xn
            - np.sum(d_xn,          axis=-1, keepdims=True)
            - x_norm * np.sum(d_xn * x_norm, axis=-1, keepdims=True)
        )
        return d_x


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention:
    """Scaled multi-head dot-product self-attention."""

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

    # ------------------------------------------------------------------
    def initialize(self, optimizer):
        scale = np.sqrt(2.0 / (self.d_model + self.d_k))

        def _w(shape):
            return np.random.normal(0, scale, shape)

        self.W_Q, self.b_Q = _w((self.d_model, self.d_model)), np.zeros(self.d_model)
        self.W_K, self.b_K = _w((self.d_model, self.d_model)), np.zeros(self.d_model)
        self.W_V, self.b_V = _w((self.d_model, self.d_model)), np.zeros(self.d_model)
        self.W_O, self.b_O = _w((self.d_model, self.d_model)), np.zeros(self.d_model)

        for name in ("W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O", "b_O"):
            setattr(self, f"_opt_{name}", copy.deepcopy(optimizer))

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    def _split_heads(self, x):
        """(B, L, d_model)  →  (B, n_heads, L, d_k)"""
        B, L, _ = x.shape
        return x.reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        """(B, n_heads, L, d_k)  →  (B, L, d_model)"""
        B, H, L, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)

    # ------------------------------------------------------------------
    def forward(self, x, training: bool = True):
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        self._x = x

        Q = x @ self.W_Q + self.b_Q          # (B, L, d_model)
        K = x @ self.W_K + self.b_K
        V = x @ self.W_V + self.b_V

        Q_h = self._split_heads(Q)            # (B, H, L, d_k)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        # Scaled dot-product attention
        scores = Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)  # (B, H, L, L)
        attn   = self._softmax(scores)        # (B, H, L, L)

        context = attn @ V_h                  # (B, H, L, d_k)
        context = self._merge_heads(context)  # (B, L, d_model)

        out = context @ self.W_O + self.b_O  # (B, L, d_model)

        # Cache for backward
        self._Q_h, self._K_h, self._V_h = Q_h, K_h, V_h
        self._attn    = attn
        self._context = context
        return out

    # ------------------------------------------------------------------
    def backward(self, grad):
        """grad: (B, L, d_model)  →  dL/dx."""
        x  = self._x
        d  = self.d_model
        fl = lambda t: t.reshape(-1, d)

        # Output projection
        d_W_O = self._context.reshape(-1, d).T @ grad.reshape(-1, d)
        d_b_O = grad.sum(axis=(0, 1))
        d_ctx = grad @ self.W_O.T             # (B, L, d_model)

        d_ctx_h = self._split_heads(d_ctx)    # (B, H, L, d_k)

        # Gradient w.r.t. attention weights and V
        d_attn = d_ctx_h @ self._V_h.transpose(0, 1, 3, 2)     # (B, H, L, L)
        d_V_h  = self._attn.transpose(0, 1, 3, 2) @ d_ctx_h    # (B, H, L, d_k)

        # Backward through softmax: d_scores[i] = a[i] * (da[i] - sum(a[i]*da[i]))
        a         = self._attn
        d_scores  = a * (d_attn - (a * d_attn).sum(axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.d_k)

        # Gradient w.r.t. Q and K
        d_Q_h = d_scores @ self._K_h                             # (B, H, L, d_k)
        d_K_h = d_scores.transpose(0, 1, 3, 2) @ self._Q_h      # (B, H, L, d_k)

        d_Q = self._merge_heads(d_Q_h)  # (B, L, d_model)
        d_K = self._merge_heads(d_K_h)
        d_V = self._merge_heads(d_V_h)

        # Gradient w.r.t. projection matrices
        x_flat = x.reshape(-1, d)
        d_W_Q  = x_flat.T @ d_Q.reshape(-1, d)
        d_W_K  = x_flat.T @ d_K.reshape(-1, d)
        d_W_V  = x_flat.T @ d_V.reshape(-1, d)
        d_b_Q  = d_Q.sum(axis=(0, 1))
        d_b_K  = d_K.sum(axis=(0, 1))
        d_b_V  = d_V.sum(axis=(0, 1))

        # Gradient w.r.t. input x
        d_x = d_Q @ self.W_Q.T + d_K @ self.W_K.T + d_V @ self.W_V.T

        # Update parameters
        param_grads = [
            ("W_Q", d_W_Q), ("b_Q", d_b_Q),
            ("W_K", d_W_K), ("b_K", d_b_K),
            ("W_V", d_W_V), ("b_V", d_b_V),
            ("W_O", d_W_O), ("b_O", d_b_O),
        ]
        for p, gp in param_grads:
            new_val = getattr(self, f"_opt_{p}").update(getattr(self, p), gp)
            setattr(self, p, new_val)

        return d_x


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward block
# ---------------------------------------------------------------------------

class FeedForwardBlock:
    """Two-layer MLP with ReLU activation, applied independently per position."""

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff    = d_ff

    def initialize(self, optimizer):
        scale1 = np.sqrt(2.0 / self.d_model)   # He for ReLU input
        scale2 = np.sqrt(2.0 / self.d_ff)
        self.W1, self.b1 = (
            np.random.normal(0, scale1, (self.d_model, self.d_ff)),
            np.zeros(self.d_ff),
        )
        self.W2, self.b2 = (
            np.random.normal(0, scale2, (self.d_ff, self.d_model)),
            np.zeros(self.d_model),
        )
        for p in ("W1", "b1", "W2", "b2"):
            setattr(self, f"_opt_{p}", copy.deepcopy(optimizer))

    def forward(self, x, training: bool = True):
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        self._x  = x
        self._h  = np.maximum(0, x @ self.W1 + self.b1)   # (B, L, d_ff)  ReLU
        return self._h @ self.W2 + self.b2                 # (B, L, d_model)

    def backward(self, grad):
        """grad: (B, L, d_model)  →  dL/dx."""
        d_W2 = self._h.reshape(-1, self.d_ff).T   @ grad.reshape(-1, self.d_model)
        d_b2 = grad.sum(axis=(0, 1))
        d_h  = grad @ self.W2.T                             # (B, L, d_ff)

        d_h_relu = d_h * (self._h > 0)                     # ReLU derivative
        d_W1     = self._x.reshape(-1, self.d_model).T @ d_h_relu.reshape(-1, self.d_ff)
        d_b1     = d_h_relu.sum(axis=(0, 1))
        d_x      = d_h_relu @ self.W1.T

        for p, gp in [("W2", d_W2), ("b2", d_b2), ("W1", d_W1), ("b1", d_b1)]:
            setattr(self, p, getattr(self, f"_opt_{p}").update(getattr(self, p), gp))

        return d_x


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerBlock:
    """
    Post-LN encoder block (original Transformer formulation):

        x1 = LayerNorm( x  + Dropout(Attention(x))  )
        x2 = LayerNorm( x1 + Dropout(FFN(x1))       )
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.0):
        self.attention    = MultiHeadSelfAttention(d_model, n_heads)
        self.ff           = FeedForwardBlock(d_model, d_ff)
        self.norm1        = LayerNorm(d_model)
        self.norm2        = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
        self._mask1       = None
        self._mask2       = None

    def initialize(self, optimizer):
        for component in (self.attention, self.ff, self.norm1, self.norm2):
            component.initialize(optimizer)

    # ------------------------------------------------------------------
    def _dropout_fwd(self, x, slot: str, training: bool):
        if training and self.dropout_rate > 0:
            keep     = 1.0 - self.dropout_rate
            mask     = (np.random.rand(*x.shape) < keep).astype(float) / keep
            setattr(self, slot, mask)
            return x * mask
        setattr(self, slot, None)
        return x

    def _dropout_bwd(self, grad, slot: str):
        mask = getattr(self, slot)
        return grad * mask if mask is not None else grad

    # ------------------------------------------------------------------
    def forward(self, x, training: bool = True):
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        # Sub-layer 1: self-attention + residual + LN
        attn_out = self.attention.forward(x, training=training)
        attn_out = self._dropout_fwd(attn_out, "_mask1", training)
        x1       = self.norm1.forward(x + attn_out, training=training)
        self._x1 = x1

        # Sub-layer 2: feed-forward + residual + LN
        ff_out = self.ff.forward(x1, training=training)
        ff_out = self._dropout_fwd(ff_out, "_mask2", training)
        x2     = self.norm2.forward(x1 + ff_out, training=training)
        return x2

    def backward(self, grad):
        """grad: (B, L, d_model)  →  dL/dx."""
        # Sub-layer 2
        d_z2    = self.norm2.backward(grad)          # dL/d(x1 + ff_out)
        d_ff    = self._dropout_bwd(d_z2, "_mask2")
        d_x1_ff = self.ff.backward(d_ff)             # dL/dx1 via FFN
        d_x1    = d_z2 + d_x1_ff                    # residual merge

        # Sub-layer 1
        d_z1      = self.norm1.backward(d_x1)        # dL/d(x + attn_out)
        d_attn    = self._dropout_bwd(d_z1, "_mask1")
        d_x_attn  = self.attention.backward(d_attn)  # dL/dx via attention
        d_x       = d_z1 + d_x_attn                 # residual merge
        return d_x


# ---------------------------------------------------------------------------
# Full Transformer Classifier
# ---------------------------------------------------------------------------

class TransformerClassifier:
    """
    End-to-end numpy Transformer encoder for multi-class text classification.

    Architecture
    ------------
    Embedding  →  Positional Encoding  →  N × TransformerBlock
    →  Global Average Pooling  →  Dense  →  Softmax

    All computations — forward pass, backward pass (manual chain rule),
    and parameter updates — use only NumPy.

    Parameters
    ----------
    vocab_size   : Vocabulary size (including <PAD> and <UNK>).
    n_classes    : Number of output classes.
    embed_dim    : Token / model embedding dimension (d_model).
    n_heads      : Number of parallel attention heads.
    d_ff         : Hidden dimension of the position-wise FFN.
    n_blocks     : Number of stacked encoder blocks.
    max_seq_len  : Sequence length for the positional encoding table.
    dropout_rate : Dropout probability applied inside each block.
    epochs       : Maximum training epochs.
    batch_size   : Mini-batch size.
    optimizer    : Optimiser instance (deep-copied per weight tensor).
    patience     : Early-stopping patience (epochs without val-loss improvement).
    verbose      : Print a progress line every `verbose` epochs (0 = silent).
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes:  int,
        embed_dim:  int   = 64,
        n_heads:    int   = 4,
        d_ff:       int   = 128,
        n_blocks:   int   = 2,
        max_seq_len: int  = 128,
        dropout_rate: float = 0.0,
        epochs:     int   = 30,
        batch_size: int   = 32,
        optimizer         = None,
        patience:   int   = 10,
        verbose:    int   = 5,
    ):
        self.vocab_size   = vocab_size
        self.n_classes    = n_classes
        self.embed_dim    = embed_dim
        self.n_heads      = n_heads
        self.d_ff         = d_ff
        self.n_blocks     = n_blocks
        self.max_seq_len  = max_seq_len
        self.dropout_rate = dropout_rate
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.optimizer    = optimizer
        self.patience     = patience
        self.verbose      = verbose
        self.history: dict = {}

    # ------------------------------------------------------------------
    def _build(self):
        opt = self.optimizer

        self._embedding = EmbeddingLayer(self.vocab_size, self.embed_dim)
        self._embedding.initialize(opt)

        self._pos_enc = PositionalEncoding(self.max_seq_len, self.embed_dim)

        self._blocks = [
            TransformerBlock(self.embed_dim, self.n_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.n_blocks)
        ]
        for block in self._blocks:
            block.initialize(opt)

        # Classification head
        scale         = np.sqrt(2.0 / (self.embed_dim + self.n_classes))
        self._W_cls   = np.random.normal(0, scale, (self.embed_dim, self.n_classes))
        self._b_cls   = np.zeros(self.n_classes)
        self._opt_W   = copy.deepcopy(opt)
        self._opt_b   = copy.deepcopy(opt)

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    # ------------------------------------------------------------------
    def _forward(self, X_idx, training: bool = True):
        """X_idx: (B, L)  →  probs: (B, n_classes)"""
        x = self._embedding.forward(X_idx, training)  # (B, L, E)
        x = self._pos_enc.forward(x, training)
        for block in self._blocks:
            x = block.forward(x, training)
        self._seq_len = x.shape[1]
        self._pooled  = x.mean(axis=1)                # (B, E) — global avg pool
        logits        = self._pooled @ self._W_cls + self._b_cls
        return self._softmax(logits)

    # ------------------------------------------------------------------
    def _backward(self, grad_combined):
        """
        grad_combined: (B, n_classes)
            Combined gradient of softmax + cross-entropy: (probs - y) / N
        """
        # Classification head
        d_W = self._pooled.T @ grad_combined
        d_b = grad_combined.sum(axis=0)
        d_pooled = grad_combined @ self._W_cls.T           # (B, E)
        self._W_cls = self._opt_W.update(self._W_cls, d_W)
        self._b_cls = self._opt_b.update(self._b_cls, d_b)

        # Global average pool backward
        d_x = np.repeat(
            d_pooled[:, np.newaxis, :], self._seq_len, axis=1
        ) / self._seq_len                                   # (B, L, E)

        # Transformer blocks (in reverse order)
        for block in reversed(self._blocks):
            d_x = block.backward(d_x)

        # Positional encoding (pass-through)
        d_x = self._pos_enc.backward(d_x)

        # Embedding
        self._embedding.backward(d_x)

    # ------------------------------------------------------------------
    @staticmethod
    def _ce_loss_and_grad(y_oh, probs):
        """Categorical cross-entropy loss + combined softmax-CE gradient."""
        p    = np.clip(probs, 1e-15, 1.0)
        loss = -np.mean(np.sum(y_oh * np.log(p), axis=1))
        grad = (probs - y_oh) / y_oh.shape[0]   # combined softmax + CE gradient
        return loss, grad

    # ------------------------------------------------------------------
    def fit(self, X_idx, y_oh, X_val=None, y_val_oh=None):
        """
        Parameters
        ----------
        X_idx    : (N, seq_len)  integer token indices
        y_oh     : (N, n_classes) one-hot labels
        X_val    : (M, seq_len)  optional validation token indices
        y_val_oh : (M, n_classes) optional validation one-hot labels
        """
        self._build()
        N           = len(X_idx)
        best_val    = np.inf
        no_improve  = 0
        best_snap   = None

        for epoch in range(1, self.epochs + 1):
            perm  = np.random.permutation(N)
            X_idx = X_idx[perm]
            y_oh  = y_oh[perm]

            ep_probs, ep_y = [], []
            for start in range(0, N, self.batch_size):
                end   = min(start + self.batch_size, N)
                X_b   = X_idx[start:end]
                y_b   = y_oh[start:end]
                probs = self._forward(X_b, training=True)
                _, g  = self._ce_loss_and_grad(y_b, probs)
                self._backward(g)
                ep_probs.append(probs)
                ep_y.append(y_b)

            ep_probs = np.concatenate(ep_probs)
            ep_y     = np.concatenate(ep_y)
            tr_loss  = -np.mean(np.sum(ep_y * np.log(np.clip(ep_probs, 1e-15, 1)), axis=1))
            tr_acc   = (np.argmax(ep_probs, axis=1) == np.argmax(ep_y, axis=1)).mean()

            val_str = ""
            if X_val is not None and y_val_oh is not None:
                val_probs = self._forward(X_val, training=False)
                val_loss  = -np.mean(
                    np.sum(y_val_oh * np.log(np.clip(val_probs, 1e-15, 1)), axis=1)
                )
                val_str = f" | val_loss: {val_loss:.4f}"

                if val_loss < best_val - 1e-4:
                    best_val   = val_loss
                    no_improve = 0
                    best_snap  = self._snapshot()
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if self.verbose:
                            print(f"  Early stopping at epoch {epoch}")
                        if best_snap:
                            self._restore(best_snap)
                        break

            self.history[epoch] = {"loss": tr_loss, "acc": tr_acc}
            if self.verbose and epoch % self.verbose == 0:
                print(f"Epoch {epoch:4d} | loss: {tr_loss:.4f} | acc: {tr_acc:.4f}{val_str}")

    # ------------------------------------------------------------------
    def predict(self, X_idx):
        """X_idx: (N, seq_len)  →  probs: (N, n_classes)"""
        return self._forward(X_idx, training=False)

    # ------------------------------------------------------------------
    def _snapshot(self):
        snap = {
            "W_cls": self._W_cls.copy(),
            "b_cls": self._b_cls.copy(),
            "emb":   self._embedding.weights.copy(),
        }
        for i, blk in enumerate(self._blocks):
            attn = blk.attention
            for key in ("W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O", "b_O"):
                snap[f"b{i}_{key}"] = getattr(attn, key).copy()
            snap[f"b{i}_n1_g"] = blk.norm1.gamma.copy()
            snap[f"b{i}_n1_b"] = blk.norm1.beta.copy()
            snap[f"b{i}_n2_g"] = blk.norm2.gamma.copy()
            snap[f"b{i}_n2_b"] = blk.norm2.beta.copy()
            for key in ("W1", "b1", "W2", "b2"):
                snap[f"b{i}_ff_{key}"] = getattr(blk.ff, key).copy()
        return snap

    def _restore(self, snap):
        self._W_cls               = snap["W_cls"]
        self._b_cls               = snap["b_cls"]
        self._embedding.weights   = snap["emb"]
        for i, blk in enumerate(self._blocks):
            attn = blk.attention
            for key in ("W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O", "b_O"):
                setattr(attn, key, snap[f"b{i}_{key}"])
            blk.norm1.gamma = snap[f"b{i}_n1_g"]
            blk.norm1.beta  = snap[f"b{i}_n1_b"]
            blk.norm2.gamma = snap[f"b{i}_n2_g"]
            blk.norm2.beta  = snap[f"b{i}_n2_b"]
            for key in ("W1", "b1", "W2", "b2"):
                setattr(blk.ff, key, snap[f"b{i}_ff_{key}"])
