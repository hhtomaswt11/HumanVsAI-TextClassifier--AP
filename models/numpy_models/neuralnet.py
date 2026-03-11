#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from optimizer import Optimizer
from losses import MeanSquaredError
from metrics import mse

class NeuralNetwork:
 
    def __init__(self, epochs=100, batch_size=128, optimizer=None,
                 learning_rate=0.01, momentum=0.90, verbose=False, 
                 loss=MeanSquaredError, metric:callable=mse):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

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

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            
            for X_batch, y_batch in self.get_mini_batches(X, y):
                # 1. Forward propagation
                output = self.forward_propagation(X_batch, training=True)
                
                # 2. Calcula o erro da Loss Function
                epoch_loss += self.loss.loss(y_batch, output)
                error = self.loss.derivative(y_batch, output)
                
                # 3. Backward propagation
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)
            
            # Avaliação por Epoch
            loss = epoch_loss / (X.shape[0] / self.batch_size)
            output_x_all = self.predict(dataset)
            
            if self.metric is not None:
                metric = self.metric(y, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = 'NA'

            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def predict(self, dataset):
        # Aqui podemos passar um dataset formatado ou uma matriz X
        X = dataset.X if hasattr(dataset, 'X') else dataset
        return self.forward_propagation(X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")