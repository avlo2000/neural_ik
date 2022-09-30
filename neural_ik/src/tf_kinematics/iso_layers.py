import tensorflow as tf

from .tf_transformations import tf_compact
from keras.layers import Layer


class IsometryWeightedL2Norm(Layer):
    def __init__(self, translation_weight: float, rotation_weight: float, **kwargs):
        self.__translation_weight = translation_weight
        self.__rotation_weight = rotation_weight
        super(IsometryWeightedL2Norm, self).__init__(**kwargs)

    def call(self, iso, **kwargs):
        compact = tf_compact(iso)
        tr_norm = tf.linalg.norm(compact[..., :3], axis=1)
        rot_norm = tf.linalg.norm(compact[..., 3:], axis=1)
        return tf.expand_dims(tr_norm * self.__translation_weight + rot_norm * self.__rotation_weight, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


class IsometryCompact(Layer):
    def __init__(self, **kwargs):
        super(IsometryCompact, self).__init__(**kwargs)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        return tf_compact(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 6, 1


class IsometryInverse(Layer):
    def __init__(self, **kwargs):
        super(IsometryInverse, self).__init__(**kwargs)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        return tf.linalg.inv(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4, 4
