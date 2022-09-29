import math

from keras.losses import Loss
import tensorflow as tf


@tf.function
def exp_weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor):
    n = y_true.shape[1]
    weights = tf.range(0, n, dtype=y_true.dtype)
    diff_square = tf.square(y_true - y_pred)
    weighted_diff = tf.multiply(tf.exp(-weights), diff_square)
    return tf.reduce_sum(weighted_diff, axis=1) / n


class ExpWeightedMSE(Loss):
    def __init__(self, base: float = math.e, *args):
        self.__base = base
        super().__init__(*args)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        n = y_true.shape[1]
        weights = tf.range(0, n, dtype=y_true.dtype)
        diff_square = tf.square(y_true - y_pred)
        weighted_diff = tf.multiply(tf.pow(self.__base, -weights), diff_square)
        return tf.reduce_sum(weighted_diff, axis=1) / n
