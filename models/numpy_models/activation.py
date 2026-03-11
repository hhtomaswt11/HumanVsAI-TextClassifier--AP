#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from layers import Layer

class ActivationLayer(Layer):

    def forward_propagation(self, input, training=True):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0
    
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        sig = self.activation_function(input)
        return sig * (1 - sig)

class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return (input > 0).astype(float)

class SoftmaxActivation(Layer):
    """Essencial para o projeto de Multi-Classe (5 classes)"""
    def forward_propagation(self, input, training=True):
        self.input = input
        # Subtrair o máximo para estabilidade numérica (evitar overflow do np.exp)
        exp_vals = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output

    def backward_propagation(self, output_error):
        # A derivada do Softmax requer a matriz Jacobiana
        input_error = np.zeros_like(output_error)
        for i in range(len(self.output)):
            s = self.output[i].reshape(-1, 1)
            jacobiana = np.diagflat(s) - np.dot(s, s.T)
            input_error[i] = np.dot(jacobiana, output_error[i])
        return input_error

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0