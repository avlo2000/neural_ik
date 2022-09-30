from unittest import TestCase

from tf_kinematics.iso_layers import IsometryCompact, IsometryWeightedL2Norm
import tensorflow as tf


class TestIsometryCompact(TestCase):
    def test_call_batch16(self):
        layer = IsometryCompact()
        test_iso = tf.stack([tf.eye(4)] * 16)
        res = layer.call(test_iso)
        self.assertEqual(res.shape, (16, 6))

    def test_call_batch1(self):
        layer = IsometryCompact()
        test_iso = tf.stack([tf.eye(4)])
        res = layer.call(test_iso)
        self.assertEqual(res.shape, (1, 6))


class TestIsometryWeightedL2Norm(TestCase):
    def test_call_batch16(self):
        layer = IsometryWeightedL2Norm(1.0, 1.0)
        test_iso = tf.stack([tf.eye(4)] * 16)
        res = layer.call(test_iso)
        self.assertEqual(res.shape, (16, 1))
        self.assertTrue(all((res == 0.0).numpy()))

    def test_call_batch1(self):
        layer = IsometryWeightedL2Norm(1.0, 1.0)
        test_iso = tf.convert_to_tensor([1, 0, 0, 1,
                                         0, 1, 0, 0,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1], dtype=tf.float32)
        test_iso = tf.reshape(test_iso, shape=(1, 4, 4))
        res = layer.call(test_iso)
        self.assertEqual(res.shape, (1, 1))
        self.assertEqual(res, 1.0)
