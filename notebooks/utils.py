
import re
from collections import Counter

import numpy as np
import pandas as pd

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def self_attention(input_sequence):
    output = np.zeros(shape=input_sequence.shape)
    for i, pivot_vector in enumerate(input_sequence):
        scores = np.zeros(shape=(len(input_sequence),))
        for j, vector in enumerate(input_sequence):
            scores[j] = np.dot(pivot_vector, vector.T)
        scores /= np.sqrt(input_sequence.shape[1])
        scores = softmax(scores)
        new_pivot_representation = np.zeros(shape=pivot_vector.shape)
        for j, vector in enumerate(input_sequence):
            new_pivot_representation += vector * scores[j]
        output[i] = new_pivot_representation
    return output

def one_hot_encode(labels):
    classes = sorted(pd.Series(labels).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=int)
    for i, lbl in enumerate(labels):
        y[i, class_to_idx[lbl]] = 1
    return y, classes, class_to_idx

def stratified_split_indices(labels, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    labels = np.array(labels)
    unique_classes = np.unique(labels)

    train_idx = []
    test_idx = []

    for c in unique_classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_test = max(1, min(len(idx) - 1, n_test)) if len(idx) > 1 else len(idx)

        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    train_idx, test_idx = stratified_split_indices(y, test_size, random_state)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

EN_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'for', 'with',
    'at', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'as', 'that', 'this', 'it', 'its', 'into', 'than', 'then', 'their', 'they'
}

class NumpyDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

class NumpyTfidfVectorizer:
    def __init__(self, max_features=5000, analyzer='word', ngram_range=(1, 1), stop_words=None):
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.vocab_ = {}
        self.idf_ = None

    def _word_tokens(self, text):
        tokens = re.findall(r"[A-Za-z0-9']+", str(text).lower())
        if self.stop_words == 'english':
            tokens = [t for t in tokens if t not in EN_STOPWORDS]
        return tokens

    def _char_tokens(self, text):
        return list(str(text).lower())

    def _make_ngrams(self, tokens):
        n_min, n_max = self.ngram_range
        grams = []
        if self.analyzer == 'char':
            text = ''.join(tokens)
            for n in range(n_min, n_max + 1):
                if len(text) >= n:
                    grams.extend([text[i:i+n] for i in range(len(text) - n + 1)])
        else:
            for n in range(n_min, n_max + 1):
                if len(tokens) >= n:
                    grams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        return grams

    def _extract_terms(self, text):
        if self.analyzer == 'char':
            base = self._char_tokens(text)
        else:
            base = self._word_tokens(text)
        return self._make_ngrams(base)

    def fit(self, docs):
        df_counts = Counter()
        tf_counts = Counter()
        n_docs = len(docs)

        for doc in docs:
            terms = self._extract_terms(doc)
            if not terms:
                continue
            tf_counts.update(terms)
            df_counts.update(set(terms))

        if self.max_features is not None:
            candidates = sorted(tf_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
            vocab_terms = [t for t, _ in candidates]
        else:
            vocab_terms = list(tf_counts.keys())

        self.vocab_ = {t: i for i, t in enumerate(vocab_terms)}

        idf = np.zeros(len(self.vocab_), dtype=float)
        for term, j in self.vocab_.items():
            df_t = df_counts.get(term, 0)
            idf[j] = np.log((1 + n_docs) / (1 + df_t)) + 1.0
        self.idf_ = idf
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab_)), dtype=float)

        for i, doc in enumerate(docs):
            terms = self._extract_terms(doc)
            if not terms:
                continue

            counts = Counter(t for t in terms if t in self.vocab_)
            total = sum(counts.values())
            if total == 0:
                continue

            for term, c in counts.items():
                j = self.vocab_[term]
                tf = c / total
                X[i, j] = tf * self.idf_[j]

            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] /= norm

        return X

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

def confusion_matrix_np(y_true, y_pred, labels):
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm

def accuracy_score_np(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())

def matthews_corrcoef_multiclass(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix_np(y_true, y_pred, labels)

    t_k = cm.sum(axis=1)
    p_k = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()

    num = c * s - np.dot(t_k, p_k)
    den = np.sqrt((s**2 - np.dot(p_k, p_k)) * (s**2 - np.dot(t_k, t_k)))
    return 0.0 if den == 0 else float(num / den)

def classification_report_np(y_true, y_pred, labels):
    cm = confusion_matrix_np(y_true, y_pred, labels)
    lines = []
    lines.append(f"{'class':<14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")

    precisions, recalls, f1s, supports = [], [], [], []

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        lines.append(f"{label:<14}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}{support:>10d}")

    total = int(np.sum(supports))
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f = float(np.mean(f1s)) if f1s else 0.0

    w_p = float(np.average(precisions, weights=supports)) if total > 0 else 0.0
    w_r = float(np.average(recalls, weights=supports)) if total > 0 else 0.0
    w_f = float(np.average(f1s, weights=supports)) if total > 0 else 0.0

    lines.append('')
    lines.append(f"{'macro avg':<14}{macro_p:>10.4f}{macro_r:>10.4f}{macro_f:>10.4f}{total:>10d}")
    lines.append(f"{'weighted avg':<14}{w_p:>10.4f}{w_r:>10.4f}{w_f:>10.4f}{total:>10d}")
    return '\n'.join(lines)

def prepare_label_arrays(y_train_arr, y_val_arr, y_test_arr):
    labels_local = sorted(np.unique(y_train_arr).tolist())
    class_to_idx_local = {label: i for i, label in enumerate(labels_local)}
    n_classes_local = len(labels_local)

    y_train_idx_local = np.array([class_to_idx_local[label] for label in y_train_arr])
    y_val_idx_local = np.array([class_to_idx_local[label] for label in y_val_arr])
    y_test_idx_local = np.array([class_to_idx_local[label] for label in y_test_arr])

    y_train_oh_local = np.eye(n_classes_local, dtype=float)[y_train_idx_local]
    y_val_oh_local = np.eye(n_classes_local, dtype=float)[y_val_idx_local]
    y_test_oh_local = np.eye(n_classes_local, dtype=float)[y_test_idx_local]

    return {
        'labels': labels_local,
        'class_to_idx': class_to_idx_local,
        'n_classes': n_classes_local,
        'human_idx': class_to_idx_local['Human'],
        'y_train_idx': y_train_idx_local,
        'y_val_idx': y_val_idx_local,
        'y_test_idx': y_test_idx_local,
        'y_train_oh': y_train_oh_local,
        'y_val_oh': y_val_oh_local,
        'y_test_oh': y_test_oh_local,
    }

def build_vectorized_datasets(X_train_arr, X_val_arr, X_test_arr, y_train_oh_arr, y_val_oh_arr, y_test_oh_arr, vectorizer_params):
    vectorizer_local = NumpyTfidfVectorizer(**vectorizer_params)
    X_train_vec_local = vectorizer_local.fit_transform(X_train_arr)
    X_val_vec_local = vectorizer_local.transform(X_val_arr)
    X_test_vec_local = vectorizer_local.transform(X_test_arr)

    train_ds_local = NumpyDataset(X_train_vec_local, y_train_oh_arr)
    val_ds_local = NumpyDataset(X_val_vec_local, y_val_oh_arr)
    test_ds_local = NumpyDataset(X_test_vec_local, y_test_oh_arr)

    return {
        'vectorizer': vectorizer_local,
        'X_train_vec': X_train_vec_local,
        'X_val_vec': X_val_vec_local,
        'X_test_vec': X_test_vec_local,
        'train_ds': train_ds_local,
        'val_ds': val_ds_local,
        'test_ds': test_ds_local,
    }

def balanced_accuracy_from_preds(y_true_str, y_pred_str, class_labels):
    cm_local = confusion_matrix_np(y_true_str, y_pred_str, class_labels)
    recalls = []
    for i in range(len(class_labels)):
        support = cm_local[i, :].sum()
        recalls.append((cm_local[i, i] / support) if support > 0 else 0.0)
    return float(np.mean(recalls))

def apply_human_confidence_rule(probs, class_labels, human_class_idx, threshold):
    pred_idx = np.argmax(probs, axis=1)
    second_idx = np.argsort(probs, axis=1)[:, -2]
    adjusted_idx = pred_idx.copy()

    human_mask = pred_idx == human_class_idx
    low_conf_mask = probs[np.arange(len(probs)), human_class_idx] < threshold
    switch_mask = human_mask & low_conf_mask

    adjusted_idx[switch_mask] = second_idx[switch_mask]
    return np.array([class_labels[i] for i in adjusted_idx])

def tune_human_threshold(probs, y_true_str, class_labels, human_class_idx, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.arange(0.40, 0.86, 0.05)

    best_threshold_local = 0.65
    best_bal_acc_local = -1.0
    best_acc_local = -1.0

    for th in threshold_grid:
        candidate_preds = apply_human_confidence_rule(probs, class_labels, human_class_idx, th)
        bal_acc = balanced_accuracy_from_preds(y_true_str, candidate_preds, class_labels)
        acc_val = accuracy_score_np(y_true_str, candidate_preds)

        if (bal_acc > best_bal_acc_local) or (bal_acc == best_bal_acc_local and acc_val > best_acc_local):
            best_bal_acc_local = bal_acc
            best_acc_local = acc_val
            best_threshold_local = float(th)

    return best_threshold_local, best_bal_acc_local, best_acc_local