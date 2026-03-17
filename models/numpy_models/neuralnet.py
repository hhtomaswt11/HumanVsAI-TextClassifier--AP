#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from optimizer import Optimizer
from losses import MeanSquaredError
from metrics import mse

class NeuralNetwork:
 
    def __init__(self, epochs=100, batch_size=128, optimizer=None,
                 learning_rate=0.01, momentum=0.90, verbose=False, 
                 loss=MeanSquaredError, metric:callable=mse, patience=10):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_proto = optimizer if optimizer is not None else Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss() if isinstance(loss, type) else loss
        self.metric = metric
        self.patience = patience


        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer_proto)
        self.layers.append(layer)
        return self

    def backward_propagation(self, error):
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_indices = indices[start:end]
            if y is not None:
                yield X[batch_indices], y[batch_indices]
            else:
                yield X[batch_indices]

    def forward_propagation(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training=training)
        return output

    def fit(self, dataset, val_dataset=None, patience=20):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)
                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)
            loss = self.loss.loss(y_all, output_x_all)
            metric = self.metric(y_all, output_x_all) if self.metric else 'NA'
            metric_s = f"{self.metric.__name__}: {metric:.4f}" if self.metric else "NA"

            # Validação
            val_loss_s = ""
            if val_dataset is not None:
                val_output = self.forward_propagation(val_dataset.X, training=False)
                val_loss = self.loss.loss(val_dataset.y, val_output)
                val_loss_s = f" - val_loss: {val_loss:.4f}"

                # Early stopping
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = [np.copy(l.weights) if hasattr(l, 'weights') else None
                                for l in self.layers]
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if self.verbose:
                            print(f"\n[Early Stopping] Parou na época {epoch}. Melhor Val Loss: {best_val_loss:.4f}")
                        # Restaurar melhores pesos
                        for l, w in zip(self.layers, best_weights):
                            if w is not None:
                                l.weights = w
                        break

            self.history[epoch] = {'loss': loss, 'metric': metric}
            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f}{val_loss_s} - {metric_s}")

        return self

    def predict(self, dataset):
        X = dataset.X if hasattr(dataset, 'X') else dataset
        return self.forward_propagation(X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")