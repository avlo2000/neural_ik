import tensorflow as tf
from keras import layers


@tf.keras.utils.register_keras_serializable()
class WeightedSum(layers.Layer):
    def call(self, inputs, **kwargs):
        w, dt, t = inputs
        tf.debugging.check_numerics(w, f"{self.name}: {w}")
        tf.debugging.check_numerics(dt, f"{self.name}: {dt}")
        tf.debugging.check_numerics(t, f"{self.name}: {t}")
        return dt * w + t


@tf.keras.utils.register_keras_serializable()
class Sum(layers.Layer):
    def call(self, inputs, **kwargs):
        dt, t = inputs
        tf.debugging.check_numerics(dt, f"{self.name}: {dt}")
        tf.debugging.check_numerics(t, f"{self.name}: {t}")
        return dt + t
