from unittest import TestCase

from tf_kinematics.layers.iso_layers import IsometryCompact, CompactL2Norm, CompactL1Norm, IsometryMul
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


class TestCompactL2Norm(TestCase):
    def test_call_batch16(self):
        batch_size = 16
        layer = CompactL2Norm(1.0, 1.0)
        test_compact = tf.constant([0] * batch_size * 6, dtype=tf.float64, shape=(batch_size, 6))
        res = layer.call(test_compact)
        self.assertEqual(res.shape, (batch_size, 1))
        self.assertTrue(all((res == 0.0).numpy()))

    def test_call_batch1(self):
        batch_size = 1
        layer = CompactL2Norm(1.0, 1.0)
        test_compact = tf.constant([0] * batch_size * 6, dtype=tf.float64, shape=(batch_size, 6))
        res = layer.call(test_compact)
        self.assertEqual(res.shape, (batch_size, 1))
        self.assertTrue(all((res == 0.0).numpy()))


class TestCompactL1Norm(TestCase):
    def test_call_batch16(self):
        batch_size = 16
        layer = CompactL1Norm(1.0, 1.0)
        test_compact = tf.constant([0] * batch_size * 6, dtype=tf.float64, shape=(batch_size, 6))
        res = layer.call(test_compact)
        self.assertEqual(res.shape, (batch_size, 1))
        self.assertTrue(all((res == 0.0).numpy()))

    def test_call_batch1(self):
        batch_size = 1
        layer = CompactL1Norm(1.0, 1.0)
        test_compact = tf.constant([0] * batch_size * 6, dtype=tf.float64, shape=(batch_size, 6))
        res = layer.call(test_compact)
        self.assertEqual(res.shape, (batch_size, 1))
        self.assertTrue(all((res == 0.0).numpy()))


class TestIsometryDiff(TestCase):
    def test_call_batch16(self):
        layer = IsometryMul()
        test_iso = tf.stack([tf.eye(4)] * 16)
        res = layer.call([test_iso, test_iso])
        self.assertEqual(res.shape, (16, 4, 4))

    def test_call_batch1(self):
        layer = IsometryMul()
        test_iso = tf.stack([tf.eye(4)])
        res = layer.call([test_iso, test_iso])
        self.assertEqual(res.shape, (1, 4, 4))
