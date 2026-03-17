#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    

class DenseLayer(Layer):
    
    def __init__(self, n_units, input_shape=None, init_type='he'):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.init_type = init_type

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        
    def initialize(self, optimizer):
        n_in, n_out = self.input_shape()[0], self.n_units
        
        if self.init_type == 'he':
            # He Initialization: std = sqrt(2/n_in)
            std = np.sqrt(2.0 / n_in)
            self.weights = np.random.normal(0, std, (n_in, n_out))
        elif self.init_type == 'xavier':
            # Xavier/Glorot Initialization: std = sqrt(2/(n_in + n_out))
            std = np.sqrt(2.0 / (n_in + n_out))
            self.weights = np.random.normal(0, std, (n_in, n_out))
        else:
            # Original basic initialization
            self.weights = np.random.rand(n_in, n_out) - 0.5
            
        # initialize biases to 0
        self.biases = np.zeros((1, n_out))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
 
    def backward_propagation(self, output_error):
         # Calcula o erro de entrada da camada (dE/dX) para passar para a camada anterior
         input_error = np.dot(output_error, self.weights.T)
    
         # Calcula o erro dos pesos (dE/dW = X.T * dE/dY)
         weights_error = np.dot(self.input.T, output_error)
         
         # Calcula o erro do bias (dE/dB = sum(dE/dY))
         bias_error = np.sum(output_error, axis=0, keepdims=True)
    
         # Atualiza os parâmetros usando o otimizador
         self.weights = self.w_opt.update(self.weights, weights_error)
         self.biases = self.b_opt.update(self.biases, bias_error)
         
         return input_error
    
    def output_shape(self):
         return (self.n_units,)

class DropoutLayer(Layer):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward_propagation(self, inputs, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
            return (inputs * self.mask) / (1 - self.dropout_rate)
        else:
            return inputs
    
    def backward_propagation(self, output_error):
        return (output_error * self.mask) / (1 - self.dropout_rate)
    
    def output_shape(self):
        return self._input_shape 
    
    def parameters(self):
        return 0