from tensorflow import keras
import tensorflow as tf


class SimpleDense(keras.layers.Layer):
    def __init__(self, _units, _activation=None):
        super().__init__()
        self.b = None
        self.W = None
        self.units = _units
        self.activation = _activation

    def build(self, __input_shape):
        input_dim = __input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer="random_normal")
        self.b = self.add_weight(shape=self.units, initializer="zeros")

    def call(self, __inputs):
        y = tf.matmul(__inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y
