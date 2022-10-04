from tf_kinematics.layers.base import _KinematicLayer
from tf_kinematics.sys_solve import solve_iter_grad
from tf_kinematics.tf_transformations import tf_compact

from keras.losses import LossFunctionWrapper
from keras.optimizers import TFOptimizer
from keras.losses import get as get_loss
from keras.optimizers import get as get_opt

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class SolveIterGrad(_KinematicLayer):
    def __init__(self, loss_ident, *args, **kwargs):
        self.loss_ident = loss_ident

        self.__loss: LossFunctionWrapper = get_loss(self.loss_ident)
        self.__sys_fn = tf.function(lambda theta: tf_compact(self._kernel.forward(tf.reshape(theta, [-1]))))

        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        y_goal, x = inputs
        return solve_iter_grad(y_goal, x, self.__sys_fn, self.__loss)

    def get_config(self):
        config = super(SolveIterGrad, self).get_config()
        config.update({'loss_ident': self.loss_ident})
        return config
