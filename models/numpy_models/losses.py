#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        # Derivada dividida pelo batch_size
        return 2 * (y_pred - y_true) / y_true.shape[0]


class BinaryCrossEntropy(LossFunction):
    
    def __init__(self, class_weights=None):
        """
        class_weights: array-like of shape (2,) for [class 0, class 1]
        """
        self.class_weights = np.array(class_weights) if class_weights is not None else None

    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        if self.class_weights is not None:
            # y_true is (samples, 1) or (samples,)
            weights = np.where(y_true == 1, self.class_weights[1], self.class_weights[0])
            if weights.ndim == 1: weights = weights.reshape(-1, 1)
            loss *= weights
        return np.mean(loss)

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        grad = (((1 - y_true) / (1 - p)) - (y_true / p))
        if self.class_weights is not None:
            weights = np.where(y_true == 1, self.class_weights[1], self.class_weights[0])
            if weights.ndim == 1: weights = weights.reshape(-1, 1)
            grad *= weights
        return grad / y_true.shape[0]


class CategoricalCrossEntropy(LossFunction):
    """Essencial para Multi-Classe (5 classes)"""
    def __init__(self, class_weights=None):
        """
        class_weights: array-like of shape (n_classes,)
        """
        self.class_weights = np.array(class_weights) if class_weights is not None else None

    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(p), axis=1)
        if self.class_weights is not None:
            # y_true is one-hot (samples, classes)
            weights = np.dot(y_true, self.class_weights)
            loss *= weights
        return np.mean(loss)

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        grad = -(y_true / p) # (samples, classes)
        if self.class_weights is not None:
            weights = np.dot(y_true, self.class_weights).reshape(-1, 1)
            grad *= weights
        return grad / y_true.shape[0]