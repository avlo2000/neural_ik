from unittest import TestCase

from keras import Input, Model

from tf_kinematics.kinematic_models_io import load
from tf_kinematics.layers.kin_layers import ForwardKinematics, JacobianForwardKinematics, NewtonIter, LimitsLerp

import tensorflow as tf


class TestForwardKinematics(TestCase):
    def test_call_batch1(self):
        layer = ForwardKinematics("kuka_robot", 1)
        theta_test = tf.random.uniform(shape=(7, 1))
        layer.build(theta_test.shape)
        res = layer.call(theta_test)

        self.assertEqual(res.shape, (1, 4, 4))

    def test_call_batch3(self):
        layer = ForwardKinematics("kuka_robot", 3)
        theta_test = tf.random.uniform(shape=(7, 3))
        layer.build(theta_test.shape)
        res = layer.call(theta_test)

        self.assertEqual(res.shape, (3, 4, 4))


class TestJacobianForwardKinematics(TestCase):
    def test_call_batch1(self):
        layer = JacobianForwardKinematics("kuka_robot", 1)
        theta_test = tf.random.uniform(shape=(1, 7))
        layer.build(theta_test.shape)
        res = layer.call(theta_test)

        self.assertEqual(res.shape, (1, 6, 8))

    def test_call_batch3(self):
        layer = JacobianForwardKinematics("kuka_robot", 3)
        theta_test = tf.random.uniform(shape=(3, 7))
        layer.build(theta_test.shape)
        res = layer.call(theta_test)

        self.assertEqual(res.shape, (3, 6, 8))


class TestNewtonIter(TestCase):
    def test_call_batch1(self):
        bs = 1
        layer = NewtonIter("kuka_robot", bs)
        theta_test = tf.random.uniform(shape=(bs, 7))
        gamma_test = tf.random.uniform(shape=(bs, 6))
        layer.build([gamma_test.shape, theta_test.shape])
        d_thetas, gamma_actual = layer.call([gamma_test, theta_test])

        self.assertEqual(d_thetas.shape, (bs, 7))
        self.assertEqual(gamma_actual.shape, (bs, 6))

    def test_call_batch3(self):
        bs = 3
        layer = NewtonIter("kuka_robot", bs)
        theta_test = tf.random.uniform(shape=(bs, 7))
        gamma_test = tf.random.uniform(shape=(bs, 6))
        layer.build([gamma_test.shape, theta_test.shape])
        d_thetas, gamma_actual = layer.call([gamma_test, theta_test])

        self.assertEqual(d_thetas.shape, (bs, 7))
        self.assertEqual(gamma_actual.shape, (bs, 6))

    def test_building_in_model(self):
        kin = load('kuka_robot', 1)
        inp_gamma = Input(6)
        inp_dof = Input(kin.dof)
        d_theta, gamma_fk = NewtonIter("kuka_robot", 1)([inp_gamma, inp_dof])
        model = Model(inputs=[inp_gamma, inp_dof], outputs=[d_theta, gamma_fk], name="test")
        model.compile()


class TestLimitsLerp(TestCase):
    def test_call_batch3(self):
        kin = load("kuka_robot", 3)
        layer = LimitsLerp(0, 1, "kuka_robot", 3)

        theta0 = tf.zeros(shape=(3, 7))
        layer.build([theta0.shape])

        lerped0 = layer.call(theta0)

        theta1 = tf.ones(shape=(3, 7))
        lerped1 = layer.call(theta1)

        self.assertTrue(tf.equal(kin.limits[1], lerped1).numpy().all())
        self.assertTrue(tf.equal(kin.limits[0], lerped0).numpy().all())
