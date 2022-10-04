import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def last_gamma_diff(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., -1] - y_pred[..., -1])


@tf.keras.utils.register_keras_serializable()
def first_gamma_diff(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 0] - y_pred[..., 0])


@tf.keras.utils.register_keras_serializable()
def gamma_xyz_norm(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.linalg.norm(y_true[..., :3], y_pred[..., :3], axis=1)


@tf.keras.utils.register_keras_serializable()
def gamma_andle_axis_norm(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.linalg.norm(y_true[..., 3:], y_pred[..., 3:], axis=1)


@tf.keras.utils.register_keras_serializable()
def max_diff_abs(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_max(tf.abs(y_true - y_pred), axis=1)
