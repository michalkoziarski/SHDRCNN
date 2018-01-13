import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, inputs, n_layers=20, k=3, n_filters=64, n_channels=3, activation='relu'):
        self.inputs = inputs
        self.n_layers = n_layers
        self.k = k
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.activation = activation
        self.weights = []
        self.biases = []
        self.outputs = self.inputs

        for i in range(self.n_layers):
            if i == 0:
                in_shape = self.n_channels
            else:
                in_shape = self.n_filters

            if i == self.n_layers - 1:
                out_shape = self.n_channels
            else:
                out_shape = self.n_filters

            weight = tf.Variable(tf.random_normal([self.k, self.k, in_shape, out_shape],
                                                  stddev=np.sqrt(2 / (k ** 2 * in_shape))))
            bias = tf.Variable(tf.zeros([out_shape]))

            self.weights.append(weight)
            self.biases.append(bias)
            self.outputs = tf.nn.bias_add(tf.nn.conv2d(self.outputs, weight, strides=[1, 1, 1, 1], padding='SAME'),
                                          bias)

            if i < self.n_layers - 1:
                self.outputs = tf.nn.relu(self.outputs)

        self.residual = self.outputs
        self.outputs = tf.add(self.outputs, self.inputs)

        if self.activation == 'relu':
            self.outputs = tf.minimum(tf.maximum(self.outputs, 0.0), 1.0)
        elif self.activation == 'sigmoid':
            self.outputs = tf.nn.sigmoid(self.outputs)
        elif self.activation is not None:
            raise NotImplementedError
