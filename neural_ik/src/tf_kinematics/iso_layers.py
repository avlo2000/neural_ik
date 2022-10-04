import math

import tensorflow as tf

from .tf_transformations import tf_compact
from keras.layers import Layer


@tf.keras.utils.register_keras_serializable()
class ArcCos(Layer):
    value_range = (0, math.pi)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        clipped = tf.clip_by_value(inputs, -1.0, 1.0)
        return tf.math.acos(clipped)

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable()
class IsometryCompact(Layer):
    def call(self, inputs, **kwargs) -> tf.Tensor:
        tf.debugging.check_numerics(inputs, f"{self.name}: {inputs}")
        compact = tf_compact(inputs)
        tf.debugging.check_numerics(inputs, f"{self.name}: {compact}")
        return compact

    def compute_output_shape(self, input_shape):
        return input_shape[0], 6, 1


@tf.keras.utils.register_keras_serializable()
class IsometryInverse(Layer):
    def call(self, inputs, **kwargs) -> tf.Tensor:
        return tf.linalg.inv(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4, 4


@tf.keras.utils.register_keras_serializable()
class IsometryMul(Layer):
    def call(self, inputs, **kwargs) -> tf.Tensor:
        iso0, iso1 = inputs
        return tf.matmul(iso0, iso1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4, 4


@tf.keras.utils.register_keras_serializable()
class CompactDiff(Layer):
    def call(self, compacts, **kwargs):
        compact1, compact2 = compacts
        tf.debugging.check_numerics(compact1, f"{self.name}: {compact1}")
        tf.debugging.check_numerics(compact2, f"{self.name}: {compact2}")
        return compact1 - compact2


@tf.keras.utils.register_keras_serializable()
class _CompactNorm(Layer):
    def __init__(self, translation_weight: float, rotation_weight: float, **kwargs):
        self._translation_weight = translation_weight
        self._rotation_weight = rotation_weight
        super(_CompactNorm, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1

    def get_config(self):
        config = super().get_config()
        config.update({
            "_translation_weight": self._translation_weight,
            "_rotation_weight": self._rotation_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config["_translation_weight"], config["_rotation_weight"])


@tf.keras.utils.register_keras_serializable()
class CompactL2Norm(_CompactNorm):
    def call(self, compact, **kwargs):
        tr_norm = tf.linalg.norm(compact[..., :3], axis=1)
        rot_norm = tf.linalg.norm(compact[..., 3:], axis=1)
        return tf.expand_dims(tr_norm * self._translation_weight + rot_norm * self._rotation_weight, axis=-1)


@tf.keras.utils.register_keras_serializable()
class CompactL1Norm(_CompactNorm):
    def call(self, compact, **kwargs):
        tr_norm = tf.linalg.norm(compact[..., :3], ord=1, axis=1)
        rot_norm = tf.linalg.norm(compact[..., 3:], ord=1, axis=1)
        return tf.expand_dims(tr_norm * self._translation_weight + rot_norm * self._rotation_weight, axis=-1)


