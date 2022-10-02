from unittest import TestCase

from tf_kinematics.kinematic_models import load
from tf_kinematics.kin_layers import ForwardKinematics, JacobianForwardKinematics, NewtonIter, LimitsLerp

import tensorflow as tf

from tf_kinematics.tf_transformations import tf_compact


class TestForwardKinematics(TestCase):
    def test_call_batch1(self):
        layer = ForwardKinematics("kuka_robot", 1)
        theta_test = tf.random.uniform(shape=(7, 1))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (1, 4, 4))

    def test_call_batch3(self):
        layer = ForwardKinematics("kuka_robot", 3)
        theta_test = tf.random.uniform(shape=(7, 3))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (3, 4, 4))


class TestJacobianForwardKinematics(TestCase):
    def test_call_batch1(self):
        layer = JacobianForwardKinematics("kuka_robot", 1)
        theta_test = tf.random.uniform(shape=(1, 7))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (1, 6, 8))

    def test_call_batch3(self):
        layer = JacobianForwardKinematics("kuka_robot", 3)
        theta_test = tf.random.uniform(shape=(3, 7))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (3, 6, 8))


class TestNewtonIter(TestCase):
    def test_call_batch1(self):
        layer = NewtonIter("kuka_robot", 1)
        theta_test = tf.random.uniform(shape=(1, 7))
        gamma_test = tf.random.uniform(shape=(1, 6))
        res = layer.call([gamma_test, theta_test])
        self.assertEqual(res.shape, (1, 7))

    def test_call_batch3(self):
        layer = NewtonIter("kuka_robot", 3)
        theta_test = tf.random.uniform(shape=(3, 7))
        gamma_test = tf.random.uniform(shape=(3, 6))
        res = layer.call([gamma_test, theta_test])
        self.assertEqual(res.shape, (3, 7))

    def test_converge(self):
        kin = load('kuka_robot', 1)
        layer = NewtonIter("kuka_robot", 1, return_diff=False)
        theta_expected = tf.zeros(shape=(1, kin.dof))
        gamma_expected = tf_compact(kin.forward(tf.reshape(theta_expected, [-1])))
        theta_seed = theta_expected + tf.ones(shape=(1, kin.dof)) * 0.1
        losses = []
        for _ in range(5):
            theta_seed = layer.call([gamma_expected, theta_seed])
            losses.append(float(tf.linalg.norm(theta_seed - theta_expected)))
        self.assertTrue(all(a >= b for a, b in zip(losses, losses[1:])))


class TestLimitsLerp(TestCase):
    def test_call_batch3(self):
        kin = load("kuka_robot", 3)
        layer = LimitsLerp(0, 1, "kuka_robot", 3)

        theta0 = tf.zeros(shape=(3, 7))
        lerped0 = layer.call(theta0)

        theta1 = tf.ones(shape=(3, 7))
        lerped1 = layer.call(theta1)

        self.assertTrue(tf.equal(kin.limits[1], lerped1).numpy().all())
        self.assertTrue(tf.equal(kin.limits[0], lerped0).numpy().all())
