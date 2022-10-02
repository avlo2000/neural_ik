from unittest import TestCase
from neural_ik.losses import PowWeightedMSE
import tensorflow as tf


class TestExpWeightedMSE(TestCase):
    def test_call_batch1(self):
        n = 10
        y_pred = tf.ones(shape=(1, n), dtype=tf.float32)
        y_expected = tf.zeros(shape=(1, n), dtype=tf.float32)
        loss = PowWeightedMSE()
        res = loss.call(y_pred, y_expected)
        res_transitive = loss.call(y_expected, y_pred)
        self.assertEqual(res, res_transitive)
        self.assertEqual(res.shape, (1, ))

    def test_call_batch3(self):
        n = 10
        y_pred = tf.ones(shape=(3, n), dtype=tf.float32)
        y_expected = tf.zeros(shape=(3, n), dtype=tf.float32)
        loss = PowWeightedMSE()
        res = loss.call(y_pred, y_expected)
        res_transitive = loss.call(y_expected, y_pred)
        self.assertTrue(all(res == res_transitive))
        self.assertEqual(res.shape, (3,))
