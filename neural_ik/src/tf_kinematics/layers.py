from .tf_transformations import tf_compact
from .dlkinematics import DLKinematics

import tensorflow as tf
from keras.layers import Layer


class ForwardKinematics(Layer):
    def __init__(self, kernel: DLKinematics, **kwargs):
        self.batch_size = kernel.batch_size
        self.kernel = kernel
        super(ForwardKinematics, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.kernel.dof == input_shape[1], f'DOF of kinematics chain must be same as input shape. ' \
                                                  f'DOF is {self.kernel.dof} got {input_shape[1]}'

    def call(self, inputs, **kwargs):
        return self.kernel.forward(tf.reshape(inputs, [-1]))

    def compute_output_shape(self, input_shape):
        return self.batch_size, 4, 4


class IsometryInverse(Layer):
    def __init__(self, **kwargs):
        super(IsometryInverse, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.linalg.inv(inputs)

    def compute_output_shape(self, input_shape):
        return self.batch_size, 4, 4


class IsometryCompact(Layer):
    def __init__(self, **kwargs):
        super(IsometryCompact, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf_compact(inputs)

    def compute_output_shape(self, input_shape):
        return self.batch_size, 6, 1
