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
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.patience = patience


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

    def fit(self, dataset, val_dataset=None): # Adicionado val_dataset
        X = dataset.X
        y = dataset.y
        best_val_loss = np.inf
        wait = 0 

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            
            # Treino (Batch Processing)
            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)
                epoch_loss += self.loss.loss(y_batch, output)
                error = self.loss.derivative(y_batch, output)
                
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)
            
            # Perda média de treino
            train_loss = epoch_loss / (X.shape[0] / self.batch_size)
            
            # CÁLCULO DA VALIDAÇÃO (Opcional mas crucial)
            val_loss = None
            if val_dataset is not None:
                val_output = self.forward_propagation(val_dataset.X, training=False)
                val_loss = self.loss.loss(val_dataset.y, val_output)
            
            # LÓGICA DE EARLY STOPPING (Agora baseada na Validação se existir)
            monitor_loss = val_loss if val_loss is not None else train_loss
            
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"\n[Early Stopping] Parou na época {epoch}. Melhor Val Loss: {best_val_loss:.4f}")
                    break
            
            # Guardar no histórico para os teus gráficos
            all_preds = self.predict(X)
            current_metric = self.metric(y, all_preds)
            self.history[epoch] = {
                'loss': train_loss, 
                'val_loss': val_loss if val_loss is not None else 0,
                'metric': current_metric
            }

            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                val_str = f" - val_loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f}{val_str} - {self.metric.__name__}: {current_metric:.4f}")
        
        return self

    def predict(self, dataset):
        X = dataset.X if hasattr(dataset, 'X') else dataset
        return self.forward_propagation(X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")