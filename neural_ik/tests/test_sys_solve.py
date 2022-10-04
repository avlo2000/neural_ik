from unittest import TestCase

from keras.losses import MeanSquaredError

from tf_kinematics.sys_solve import solve
from tf_kinematics.kinematic_models_io import load

import tensorflow as tf


class Test(TestCase):
    def test_newton_solve_converges(self):
        bs = 1
        kin = load('kuka_robot', bs)
        x_expected = tf.zeros(shape=(bs, kin.dof))
        y_expected = kin.forward(tf.reshape(x_expected, [-1]))
        x0 = x_expected + tf.ones(shape=(bs, kin.dof)) * 0.1
        sys_fn = tf.function(lambda x: kin.forward(tf.reshape(x, [-1])))

        loss = MeanSquaredError()
        opt = tf.keras.optimizers.SGD()

        x_pred = solve(y_expected, x0, sys_fn, n_iters=100, loss_fn=loss, optimizer=opt)
        loss = tf.reduce_sum(tf.square(x_pred - x_expected), axis=1)

        self.assertLess(loss, 0.05, "Loss expected to be less than 0.05")


