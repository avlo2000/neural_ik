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
        self.__weights = tf.cast(self.__weights, tf.float64)
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


@tf.keras.utils.register_keras_serializable()
class CompactXYZL2CosAA(Loss):
    @tf.function
    def call(self, y_true, y_pred):
        xyz_diff = tf.sqrt(tf.reduce_sum(tf.square(y_true[..., :3] - y_pred[..., :3]), axis=0))
        aa_true = y_true[..., 3:]
        aa_pred = y_pred[..., 3:]
        sim = tf.keras.losses.cosine_similarity(aa_true, aa_pred, axis=0)
        diff = tf.multiply((sim + 1.0), xyz_diff)
        return diff


@tf.keras.utils.register_keras_serializable()
class CompactL2L2(Loss):
    def __init__(self, xyz_weight: float = 1.0, aa_weight: float = 1.0):
        super().__init__()
        self.xyz_weight = xyz_weight
        self.aa_weight = aa_weight

    @tf.function
    def call(self, y_true, y_pred):
        xyz_diff = tf.sqrt(tf.reduce_sum(tf.square(y_true[..., :3] - y_pred[..., :3]), axis=0))
        aa_diff = tf.sqrt(tf.reduce_sum(tf.square(y_true[..., 3:] - y_pred[..., 3:]), axis=0))

        return self.xyz_weight * xyz_diff + self.aa_weight * aa_diff

    def get_config(self):
        config = dict()
        config.update({'xyz_weight': self.xyz_weight})
        config.update({'aa_weight': self.aa_weight})
        return config


@tf.keras.utils.register_keras_serializable()
class CompactL4L4(Loss):
    def __init__(self, xyz_weight: float = 1.0, aa_weight: float = 1.0):
        super().__init__()
        self.xyz_weight = xyz_weight
        self.aa_weight = aa_weight

    @tf.function
    def call(self, y_true, y_pred):
        xyz_diff = tf.pow(tf.reduce_sum(tf.square(y_true[..., :3] - y_pred[..., :3]), axis=0), 4)
        aa_diff = tf.pow(tf.reduce_sum(tf.square(y_true[..., 3:] - y_pred[..., 3:]), axis=0), 4)

        return self.xyz_weight * xyz_diff + self.aa_weight * aa_diff

    def get_config(self):
        config = dict()
        config.update({'xyz_weight': self.xyz_weight})
        config.update({'aa_weight': self.aa_weight})
        return config


@tf.function
def qp_loss(x, p, q):
    return 0.5 * x @ p @ tf.transpose(x) + q @ tf.transpose(x)


@tf.keras.utils.register_keras_serializable()
class QP(Loss):
    def __init__(self, p: tf.Tensor, q: tf.Tensor):
        super().__init__()
        self.p = p
        self.q = tf.reshape(q, [1, -1])

    @tf.function
    def call(self, y_true, y_pred):
        diff = y_true - y_pred
        return qp_loss(diff, self.p, self.q)

    def get_config(self):
        config = dict()
        config.update({'p': self.p})
        config.update({'q': self.q})
        return config


@tf.keras.utils.register_keras_serializable()
@tf.function
def loss_l_inf(y, y_goal):
    return tf.math.pow(tf.reduce_sum((y - y_goal) ** 50, axis=1), 1 / 50) / y_goal.shape[1]


@tf.keras.utils.register_keras_serializable()
@tf.function
def loss_l4(y, y_goal):
    return tf.math.pow(tf.reduce_sum((y - y_goal) ** 4, axis=1), 1 / 4) / y_goal.shape[1]


@tf.keras.utils.register_keras_serializable()
@tf.function
def loss_l2(y, y_goal):
    return tf.reduce_sum((y - y_goal) ** 2, axis=0)


@tf.keras.utils.register_keras_serializable()
@tf.function
def loss_l1(y, y_goal):
    return tf.reduce_sum(tf.abs(y - y_goal), axis=1)


@tf.keras.utils.register_keras_serializable()
@tf.function
def loss_l1l1(y, y_goal):
    return tf.reduce_sum(tf.abs(y[..., :3] - y_goal[..., :3]), axis=1) + \
           tf.reduce_sum(tf.abs(y[..., 3:] - y_goal[..., 3:]), axis=1)


@tf.keras.utils.register_keras_serializable()
@tf.function
def loss_l2l2(y, y_goal):
    return loss_l2(y[..., :3], y_goal[..., :3]) + loss_l2(y[..., 3:], y_goal[..., 3:])


#@tf.function
def qp_loss(x, p, q):
    x = tf.expand_dims(x, axis=1)
    q = tf.expand_dims(q, axis=2)
    x_p = 0.5 * tf.linalg.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
    res = x_p + x @ q
    return res


@tf.function
def auxiliary_fn(x, x_low, x_high):
    mask = not (x_low < x < x_high)
    return mask * 10000


@tf.keras.utils.register_keras_serializable()
class ConstrainedQP(Loss):
    pass