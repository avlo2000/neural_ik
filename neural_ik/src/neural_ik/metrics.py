import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def last_gamma_diff(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., -1] - y_pred[..., -1])


@tf.keras.utils.register_keras_serializable()
def first_gamma_diff(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 0] - y_pred[..., 0])


@tf.keras.utils.register_keras_serializable()
def gamma_xyz_max(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_max(tf.abs(y_true[..., 3:] - y_pred[..., 3:]), axis=1)


@tf.keras.utils.register_keras_serializable()
def x(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 0] - y_pred[..., 0])


@tf.keras.utils.register_keras_serializable()
def y(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 0] - y_pred[..., 0])


@tf.keras.utils.register_keras_serializable()
def z(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.abs(y_true[..., 0] - y_pred[..., 0])


@tf.keras.utils.register_keras_serializable()
def angle_axis_l2(y_true: tf.Tensor, y_pred: tf.Tensor):
    aa_true = y_true[..., :3]
    aa_pred = y_pred[..., :3]
    return tf.sqrt(tf.reduce_sum(tf.square(aa_true - aa_pred)))


@tf.keras.utils.register_keras_serializable()
def gamma_andle_axis_max(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_max(tf.abs(y_true[..., :3] - y_pred[..., :3]), axis=1)
