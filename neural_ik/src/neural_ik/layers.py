import tensorflow as tf
from keras import layers


@tf.keras.utils.register_keras_serializable()
class WeightedSum(layers.Layer):
    def call(self, inputs, **kwargs):
        w, dt, t = inputs
        return tf.add(tf.multiply(w[...], dt[...]), t[...])


@tf.keras.utils.register_keras_serializable()
class Sum(layers.Layer):
    def call(self, inputs, **kwargs):
        dt, t = inputs
        return dt[...] + t[...]


@tf.keras.utils.register_keras_serializable()
class GradOpt(layers.Layer):
    def call(self, inputs, **kwargs):
        grad, lr, params = inputs
        return params - grad * lr


@tf.keras.utils.register_keras_serializable()
class AdamOpt(layers.Layer):
    def __init__(self, beta1: float, beta2: float, epsilon: float = 1e-7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__time_step = None
        self.m = None
        self.v = None
        self.shape = None
        self.timestamp = None
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def build(self, input_shape):
        grad_shape, _, _ = input_shape
        self.shape = grad_shape
        self.m = tf.Variable(tf.zeros(self.shape, tf.float32), trainable=False)
        self.v = tf.Variable(tf.zeros(self.shape, tf.float32), trainable=False)
        self.timestamp = tf.Variable(initial_value=tf.zeros(self.shape, dtype=tf.float32), trainable=False)
        self.__time_step = tf.ones_like(self.timestamp)

    def call(self, inputs, **kwargs):
        grad, lr, params = inputs

        tf.keras.backend.update_add(self.timestamp, self.__time_step)
        tf.keras.backend.moving_average_update(self.m, grad, self.beta1)
        tf.keras.backend.moving_average_update(self.v, tf.square(grad), self.beta2)

        m_hat = self.m / (1.0 - tf.pow(self.beta1, self.timestamp)) + \
                         (1.0 - self.beta1) * grad / (1.0 - tf.pow(self.beta1, self.timestamp))
        v_hat = self.v / (1.0 - tf.pow(self.beta2, self.timestamp))
        params_updated = params - lr * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        return params_updated

    def reset_state(self):
        tf.keras.backend.update(self.m, tf.zeros_like(self.m))
        tf.keras.backend.update(self.v, tf.zeros_like(self.v))
        tf.keras.backend.update(self.timestamp, tf.zeros_like(self.timestamp))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2
        })
        return config
