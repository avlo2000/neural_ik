from .tf_transformations import tf_compact
from .dlkinematics import DLKinematics

import tensorflow as tf
from keras.layers import Layer


@tf.function
def fk_and_jacobian(thetas: tf.Tensor, kernel: DLKinematics) -> (tf.Tensor, tf.Tensor):
    with tf.GradientTape() as g:
        g.watch(thetas)
        thetas_flat = tf.reshape(thetas, [-1])
        iso3d = kernel.forward(thetas_flat)
        gamma = tf_compact(iso3d)
    jac = g.batch_jacobian(gamma, thetas)
    return jac, gamma


class ForwardKinematics(Layer):
    def __init__(self, kernel: DLKinematics, **kwargs):
        self.batch_size = kernel.batch_size
        self.kernel = kernel
        super(ForwardKinematics, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.kernel.dof == input_shape[1], f'DOF of kinematics chain must be same as input shape. ' \
                                                  f'DOF is {self.kernel.dof} got {input_shape[1]}'

    def call(self, inputs, **kwargs) -> tf.Tensor:
        thetas = tf.reshape(inputs, [-1])
        gamma = self.kernel.forward(thetas)
        return gamma

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4, 4


class IsometryInverse(Layer):
    def __init__(self, **kwargs):
        super(IsometryInverse, self).__init__(**kwargs)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        return tf.linalg.inv(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 4, 4


class IsometryCompact(Layer):
    def __init__(self, **kwargs):
        super(IsometryCompact, self).__init__(**kwargs)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        return tf_compact(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 6, 1


class JacobianForwardKinematics(Layer):
    def __init__(self, kernel: DLKinematics, **kwargs):
        self.__kernel = kernel
        super(JacobianForwardKinematics, self).__init__(**kwargs)

    def call(self, thetas: tf.Tensor, **kwargs) -> tf.Tensor:
        jac, gamma = fk_and_jacobian(thetas, self.__kernel)
        gamma = tf.expand_dims(gamma, axis=2)
        return tf.concat([jac, gamma], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 6, input_shape[1] + 1


class NewtonIter(Layer):
    def __init__(self, kernel: DLKinematics, return_diff: bool = True, learning_rate: float = 0.001, **kwargs):
        self.__kernel = kernel
        self.__learning_rate = learning_rate
        self.__return_diff = return_diff
        super(NewtonIter, self).__init__(**kwargs)

    def call(self, inputs: (tf.Tensor, tf.Tensor), **kwargs) -> tf.Tensor:
        gamma_expected, thetas = inputs
        jac, gamma_actual = fk_and_jacobian(thetas, self.__kernel)
        jac_pinv = tf.linalg.pinv(jac)
        gamma_diff = tf.reshape(gamma_expected - gamma_actual, shape=(-1, 6, 1))
        d_thetas = tf.linalg.matmul(jac_pinv, gamma_diff)
        d_thetas = tf.squeeze(d_thetas, axis=2)
        if self.__return_diff:
            return d_thetas * self.__learning_rate
        return thetas + d_thetas * self.__learning_rate

    def compute_output_shape(self, input_shape: tf.TensorShape):
        return input_shape[0], self.__kernel.dof

    @property
    def learning_rate(self):
        return self.__learning_rate
