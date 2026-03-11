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
    
    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (((1 - y_true) / (1 - p)) - (y_true / p)) / y_true.shape[0]


class CategoricalCrossEntropy(LossFunction):
    """Essencial para Multi-Classe (5 classes)"""
    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(p), axis=1))

    def derivative(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / p) / y_true.shape[0]