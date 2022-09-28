from unittest import TestCase
from neural_ik.losses import ExpWeightedMSE
import tensorflow as tf


class TestExpWeightedMSE(TestCase):
    def test_call_batch1(self):
        n = 10
        y_pred = tf.ones(shape=(1, n), dtype=tf.float32)
        y_expected = tf.zeros(shape=(1, n), dtype=tf.float32)
        loss = ExpWeightedMSE()
        res = loss.call(y_pred, y_expected)
        self.assertEqual(res, 0.15819049)
        self.assertEqual(res.shape, (1, ))

    def test_call_batch3(self):
        n = 10
        y_pred = tf.ones(shape=(3, n), dtype=tf.float32)
        y_expected = tf.zeros(shape=(3, n), dtype=tf.float32)
        loss = ExpWeightedMSE()
        res = loss.call(y_pred, y_expected)
        self.assertTrue(all(res == 0.15819049))
        self.assertEqual(res.shape, (3,))
