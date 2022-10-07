import tensorflow as tf


@tf.function()
def iso_dx(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 3, 0] - y_pred[..., 3, 0])


@tf.function()
def iso_dy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 3, 1] - y_pred[..., 3, 1])


@tf.function()
def iso_dz(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 3, 2] - y_pred[..., 3, 2])
