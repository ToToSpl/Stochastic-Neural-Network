import tensorflow as tf
from tensorflow import keras


class Stochastic(keras.layers.Layer):
    def __init__(self, units=32, input_dim=None):
        super(Stochastic, self).__init__()
        self.units = int(units)
        self.input_dim = input_dim

    def build(self, input_shape):
        in_shape = self.input_dim if self.input_dim != None else input_shape[-1]
        w_init = tf.random_uniform_initializer(
            minval=-0.1, maxval=0.1)
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(in_shape, self.units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.random_uniform_initializer(
            minval=-0.1, maxval=0.1)
        self.b = tf.Variable(
            initial_value=b_init(
                shape=(self.units,), dtype="float32"),
            trainable=True
        )

    def call(self, inputs):
        sig_out = tf.math.sigmoid(
            tf.matmul(inputs, self.w) + self.b)
        random_vals = tf.random.uniform(
            shape=(self.units,), minval=0.0, maxval=1.0)
        greater = tf.cast(
            tf.greater(sig_out, random_vals),
            dtype=tf.float32)
        return 2.0 * greater - 1.0
