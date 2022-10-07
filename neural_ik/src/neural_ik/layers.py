import tensorflow as tf
from keras import layers


@tf.keras.utils.register_keras_serializable()
class WeightedSum(layers.Layer):
    def call(self, inputs, **kwargs):
        w, dt, t = inputs
        return w[...] * dt[...] + t[...]


# class AdamOpt(layers.Layer):
#     def call(self, inputs, **kwargs):
#         w, dt, t = inputs
#         return w[...] * dt[...] + t[...]


@tf.keras.utils.register_keras_serializable()
class Sum(layers.Layer):
    def call(self, inputs, **kwargs):
        dt, t = inputs
        return dt[...] + t[...]
