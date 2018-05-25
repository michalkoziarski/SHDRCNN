import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, inputs, n_layers=20, k=3, n_filters=64, n_channels=3, inner_activation='tanh',
                 outer_activation='sigmoid'):
        self.inputs = inputs

        if type(n_layers) is int:
            self.n_layers = [n_layers]
        else:
            self.n_layers = n_layers

        self.k = k
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.inner_activation = inner_activation
        self.outer_activation = outer_activation
        self.weights = []
        self.biases = []
        self.outputs = self.inputs

        for stream_n_layers in self.n_layers:
            stream_output = self.inputs

            for i in range(stream_n_layers):
                if i == 0:
                    in_shape = self.n_channels
                else:
                    in_shape = self.n_filters

                if i == stream_n_layers - 1:
                    out_shape = self.n_channels
                else:
                    out_shape = self.n_filters

                weight = tf.Variable(tf.random_normal([self.k, self.k, in_shape, out_shape],
                                                      stddev=np.sqrt(2 / (k ** 2 * in_shape))))
                bias = tf.Variable(tf.zeros([out_shape]))

                self.weights.append(weight)
                self.biases.append(bias)

                stream_output = tf.nn.bias_add(tf.nn.conv2d(stream_output, weight, strides=[1, 1, 1, 1],
                                                            padding='SAME'), bias)

                if i < stream_n_layers - 1:
                    if self.inner_activation == 'relu':
                        stream_output = tf.nn.relu(stream_output)
                    elif self.inner_activation == 'tanh':
                        stream_output = tf.nn.tanh(stream_output)
                    else:
                        raise NotImplementedError

            self.outputs = tf.add(self.outputs, stream_output)

        self.residual = self.outputs

        if self.outer_activation == 'relu':
            self.outputs = tf.minimum(tf.maximum(self.outputs, 0.0), 1.0)
        elif self.outer_activation == 'sigmoid':
            self.outputs = tf.nn.sigmoid(self.outputs)
        elif self.outer_activation is not None:
            raise NotImplementedError
