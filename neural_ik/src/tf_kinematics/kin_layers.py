from .tf_transformations import tf_compact
from .dlkinematics import DLKinematics

import tensorflow as tf
from keras.layers import Layer
from tf_kinematics.kinematic_models import load as load_kin


@tf.function
def fk_and_jacobian(thetas: tf.Tensor, kernel: DLKinematics) -> (tf.Tensor, tf.Tensor):
    with tf.GradientTape() as g:
        g.watch(thetas)
        thetas_flat = tf.reshape(thetas, [-1])
        iso3d = kernel.forward(thetas_flat)
        gamma = tf_compact(iso3d)
    jac = g.batch_jacobian(gamma, thetas)
    return jac, gamma


@tf.function
def newton_iter(jac: tf.Tensor, gamma_expected: tf.Tensor, gamma_actual: tf.Tensor) -> tf.Tensor:
    jac_pinv = tf.linalg.pinv(jac)
    gamma_diff = tf.reshape(gamma_expected - gamma_actual, shape=(-1, 6, 1))
    d_thetas = tf.linalg.matmul(jac_pinv, gamma_diff)
    d_thetas = tf.squeeze(d_thetas, axis=2)
    return d_thetas


@tf.keras.utils.register_keras_serializable()
class _KinematicLayer(Layer):
    def __init__(self, kin_model_name: str, batch_size: int, **kwargs):
        self._kernel: DLKinematics = load_kin(kin_model_name, batch_size)
        self.__kin_model_name = kin_model_name
        self.__batch_size = batch_size
        super(_KinematicLayer, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, config: dict):
        return cls(config['kin_model_name'], config['batch_size'])

    def get_config(self):
        config = super(_KinematicLayer, self).get_config()
        config.update({'kin_model_name': self.__kin_model_name})
        config.update({'batch_size': self.__batch_size})
        return config


@tf.keras.utils.register_keras_serializable()
class ForwardKinematics(_KinematicLayer):
    def __init__(self, kin_model_name: str, batch_size: int, **kwargs):
        super(ForwardKinematics, self).__init__(kin_model_name, batch_size, **kwargs)

    def build(self, input_shape):
        assert self._kernel.dof == input_shape[1], f'DOF of kinematics chain must be same as input shape. ' \
                                                  f'DOF is {self._kernel.dof} got {input_shape[1]}'

    def call(self, inputs, **kwargs) -> tf.Tensor:
        tf.debugging.check_numerics(inputs, f"{self.name}: {inputs}")
        thetas = tf.reshape(inputs, [-1])
        iso = self._kernel.forward(thetas)
        tf.debugging.check_numerics(iso, f"{self.name}: {iso}")
        return iso

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4, 4


@tf.keras.utils.register_keras_serializable()
class JacobianForwardKinematics(_KinematicLayer):
    def __init__(self, kin_model_name: str, batch_size: int, **kwargs):
        super(JacobianForwardKinematics, self).__init__(kin_model_name, batch_size, **kwargs)

    def call(self, thetas: tf.Tensor, **kwargs) -> tf.Tensor:
        tf.debugging.check_numerics(thetas, f"{self.name}: {thetas}")
        jac, gamma = fk_and_jacobian(thetas, self._kernel)
        gamma = tf.expand_dims(gamma, axis=2)
        return tf.concat([jac, gamma], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 6, input_shape[1] + 1


@tf.keras.utils.register_keras_serializable()
class NewtonIter(_KinematicLayer):
    def __init__(self, kin_model_name: str, batch_size: int, return_diff: bool = True, learning_rate: float = 0.001,
                 **kwargs):
        self.__learning_rate = learning_rate
        self.__return_diff = return_diff
        super(NewtonIter, self).__init__(kin_model_name, batch_size, **kwargs)

    def call(self, inputs: (tf.Tensor, tf.Tensor), **kwargs) -> tf.Tensor:
        gamma_expected, thetas = inputs
        tf.debugging.check_numerics(gamma_expected, f"{self.name}: {gamma_expected}")
        tf.debugging.check_numerics(thetas, f"{self.name}: {thetas}")
        jac, gamma_actual = fk_and_jacobian(thetas, self._kernel)
        d_thetas = newton_iter(jac, gamma_expected, gamma_actual)
        if self.__return_diff:
            return d_thetas * self.__learning_rate
        return thetas + d_thetas * self.__learning_rate

    def compute_output_shape(self, input_shape: tf.TensorShape):
        return input_shape[0], self.__kernel.dof

    @property
    def learning_rate(self):
        return self.__learning_rate


@tf.keras.utils.register_keras_serializable()
class LimitsLerp(_KinematicLayer):
    def __init__(self, min_val: float, max_val: float, kin_model_name: str, batch_size: int, **kwargs):
        self.__min_val = min_val
        self.__max_val = max_val
        super(LimitsLerp, self).__init__(kin_model_name, batch_size, **kwargs)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        tf.debugging.check_numerics(inputs, f"{self.name}: {inputs}")
        limits = self._kernel.limits
        zero_one_range = tf.subtract(inputs, self.__min_val) / (self.__max_val - self.__min_val)
        return tf.multiply(zero_one_range[..., ], (limits[1] - limits[0])) - limits[1]

    def compute_output_shape(self, input_shape: tf.TensorShape):
        return input_shape

    @classmethod
    def from_config(cls, config: dict):
        return cls(config['min_val'], config['max_val'],
                   config['kin_model_name'], config['batch_size'])

    def get_config(self):
        config = super(LimitsLerp, self).get_config()
        config.update({'min_val': self.__min_val})
        config.update({'max_val': self.__max_val})
        return config