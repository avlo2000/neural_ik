from tf_kinematics.layers.base import _KinematicLayer
from tf_kinematics.tf_transformations import tf_compact
from tf_kinematics.dlkinematics import DLKinematics

import tensorflow as tf
from keras.layers import Layer
from tf_kinematics.kinematic_models_io import load as load_kin


@tf.function
def fk_and_jacobian(thetas: tf.Tensor, kernel: DLKinematics) -> (tf.Tensor, tf.Tensor):
    with tf.GradientTape(persistent=True) as g:
        g.watch(thetas)
        thetas_flat = tf.reshape(thetas, [-1])
        tf.debugging.check_numerics(thetas_flat, f"fk_and_jacobian: {thetas_flat}")

        iso3d = kernel.forward(thetas_flat)
        tf.debugging.check_numerics(iso3d, f"fk_and_jacobian: {iso3d}")

        gamma = tf_compact(iso3d)
        tf.debugging.check_numerics(gamma, f"fk_and_jacobian: {gamma}")
    jac = g.batch_jacobian(gamma, thetas)
    tf.debugging.check_numerics(jac, f"fk_and_jacobian: {jac}")

    return jac, gamma


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
    def __init__(self, kin_model_name: str, batch_size: int, **kwargs):
        super(NewtonIter, self).__init__(kin_model_name, batch_size, **kwargs)

    def call(self, inputs: (tf.Tensor, tf.Tensor), **kwargs) -> (tf.Tensor, tf.Tensor):
        gamma_expected, thetas = inputs
        tf.debugging.check_numerics(gamma_expected, f"{self.name}: {gamma_expected}")
        tf.debugging.check_numerics(thetas, f"{self.name}: {thetas}")

        jac, gamma_actual = fk_and_jacobian(thetas, self._kernel)

        tf.debugging.check_numerics(jac, f"{self.name}: {jac}")
        tf.debugging.check_numerics(gamma_actual, f"{self.name}: {gamma_actual}")

        jac_pinv = tf.stop_gradient(tf.linalg.pinv(jac))
        gamma_diff = tf.reshape(gamma_expected - gamma_actual, shape=(-1, 6, 1))

        tf.debugging.check_numerics(jac_pinv, f"{self.name}: {jac_pinv}")
        tf.debugging.check_numerics(gamma_diff, f"{self.name}: {gamma_diff}")

        d_thetas = tf.linalg.matmul(jac_pinv, gamma_diff)
        d_thetas = tf.squeeze(d_thetas, axis=2)
        tf.debugging.check_numerics(d_thetas, f"{self.name}: {d_thetas}")

        return d_thetas, gamma_actual

    def compute_output_shape(self, input_shape: tf.TensorShape):
        return (input_shape[0], self.__kernel.dof), (input_shape[0], 6)


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
