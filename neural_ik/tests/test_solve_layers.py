from unittest import TestCase
from tf_kinematics.layers.solve_layers import SolveIterGrad
import tensorflow as tf


class TestSolveIterGrad(TestCase):
    def test_call_batch1(self):
        bs = 1
        layer = SolveIterGrad("mse", "kuka_robot", bs)
        theta_test = tf.random.uniform(shape=(bs, 7))
        gamma_test = tf.random.uniform(shape=(bs, 6))

        layer.build((gamma_test.shape, theta_test.shape))
        theta = layer.call([gamma_test, theta_test])
        self.assertEqual(theta.shape, (bs, 7))

    def test_call_batch3(self):
        bs = 3
        layer = SolveIterGrad("mse", "kuka_robot", bs)
        theta_test = tf.random.uniform(shape=(bs, 7))
        gamma_test = tf.random.uniform(shape=(bs, 6))

        layer.build((gamma_test.shape, theta_test.shape))
        theta = layer.call([gamma_test, theta_test])
        self.assertEqual(theta.shape, (bs, 7))
