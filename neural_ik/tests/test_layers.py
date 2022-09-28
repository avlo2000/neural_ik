from unittest import TestCase

from tf_kinematics.kinematic_models import kuka_robot
from tf_kinematics.layers import ForwardKinematics, JacobianForwardKinematics, IsometryCompact, NewtonIter

import tensorflow as tf

from tf_kinematics.tf_transformations import tf_compact


class TestForwardKinematics(TestCase):
    def test_call_batch1(self):
        kin = kuka_robot(1)
        layer = ForwardKinematics(kin)
        theta_test = tf.random.uniform(shape=(kin.dof, 1))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (1, 4, 4))

    def test_call_batch3(self):
        kin = kuka_robot(3)
        layer = ForwardKinematics(kin)
        theta_test = tf.random.uniform(shape=(kin.dof, 3))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (3, 4, 4))


class TestJacobianForwardKinematics(TestCase):
    def test_call_batch1(self):
        kin = kuka_robot(1)
        layer = JacobianForwardKinematics(kin)
        theta_test = tf.random.uniform(shape=(1, kin.dof))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (1, 6, 8))

    def test_call_batch3(self):
        kin = kuka_robot(3)
        layer = JacobianForwardKinematics(kin)
        theta_test = tf.random.uniform(shape=(3, kin.dof))
        res = layer.call(theta_test)
        self.assertEqual(res.shape, (3, 6, 8))


class TestNewtonIter(TestCase):
    def test_call_batch1(self):
        kin = kuka_robot(1)
        layer = NewtonIter(kin)
        theta_test = tf.random.uniform(shape=(1, kin.dof))
        gamma_test = tf.random.uniform(shape=(1, 6))
        res = layer.call([gamma_test, theta_test])
        self.assertEqual(res.shape, (1, kin.dof))

    def test_call_batch3(self):
        kin = kuka_robot(3)
        layer = NewtonIter(kin)
        theta_test = tf.random.uniform(shape=(3, kin.dof))
        gamma_test = tf.random.uniform(shape=(3, 6))
        res = layer.call([gamma_test, theta_test])
        self.assertEqual(res.shape, (3, kin.dof))

    def test_converge(self):
        kin = kuka_robot(1)
        layer = NewtonIter(kin, return_diff=False)
        theta_expected = tf.zeros(shape=(1, kin.dof))
        gamma_expected = tf_compact(kin.forward(tf.reshape(theta_expected, [-1])))
        theta_seed = theta_expected + tf.ones(shape=(1, kin.dof)) * 0.1
        losses = []
        for _ in range(5):
            theta_seed = layer.call([gamma_expected, theta_seed])
            losses.append(float(tf.linalg.norm(theta_seed - theta_expected)))
        self.assertTrue(all(a >= b for a, b in zip(losses, losses[1:])))


class TestIsometryCompact(TestCase):
    def test_call_batch16(self):
        layer = IsometryCompact()
        test_iso = tf.stack([tf.eye(4)]*16)
        res = layer.call(test_iso)
        self.assertEqual(res.shape, (16, 6))

    def test_call_batch1(self):
        layer = IsometryCompact()
        test_iso = tf.stack([tf.eye(4)])
        res = layer.call(test_iso)
        self.assertEqual(res.shape, (1, 6))
