import math

from keras.losses import Loss
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class PowWeightedMSE(Loss):
    def __init__(self, base: float = math.e, normalize=True, *args, **kwargs):
        self.__base = base
        self.__weights = None
        self.__normalize = normalize
        super().__init__(*args, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        n = y_true.shape[1]
        self.__weights = tf.reshape(tf.linspace(n, 0, n), shape=(1, n))
        self.__weights = tf.cast(self.__weights, tf.float32)
        self.__weights = tf.pow(self.__base, self.__weights)
        self.__weights = tf.divide(self.__weights, tf.reduce_sum(self.__weights, axis=1))

        diff_square = tf.square(y_true - y_pred)
        weighted_diff = tf.multiply(self.__weights, diff_square)
        return tf.reduce_sum(weighted_diff, axis=1) / n

    def get_config(self):
        config = dict()
        config.update({'base': self.__base})
        config.update({'normalize': self.__normalize})
        return config

