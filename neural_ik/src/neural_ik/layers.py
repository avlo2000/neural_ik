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
class MomentumOpt(layers.Layer):
    def __init__(self, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = None
        self.shape = None
        self.beta = beta

    def build(self, input_shape):
        grad_shape, _, _ = input_shape
        self.shape = grad_shape
        self.m = tf.Variable(tf.zeros(self.shape, tf.float32), trainable=False)

    @tf.function
    def call(self, inputs, **kwargs):
        grad, lr, params = inputs
        tf.keras.backend.moving_average_update(self.m, grad, self.beta)
        return params - self.m * lr

    @tf.function
    def reset_state(self):
        tf.keras.backend.update(self.m, tf.zeros_like(self.m))


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
        self.timestamp = 0

    def call(self, inputs, **kwargs):
        grad, lr, params = inputs

        self.timestamp += 1
        tf.keras.backend.moving_average_update(self.m, grad, self.beta1)
        tf.keras.backend.moving_average_update(self.v, tf.square(grad), self.beta2)

        beta1_pow_ts = tf.pow(self.beta1, self.timestamp)
        m_hat = self.m / (1.0 - beta1_pow_ts) + \
                         (1.0 - self.beta1) * grad / (1.0 - beta1_pow_ts)
        v_hat = self.v / (1.0 - tf.pow(self.beta2, self.timestamp))
        params_updated = params - lr * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        return params_updated

    def reset_state(self):
        self.timestamp = 0
        tf.keras.backend.update(self.m, tf.zeros_like(self.m))
        tf.keras.backend.update(self.v, tf.zeros_like(self.v))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2
        })
        return config
