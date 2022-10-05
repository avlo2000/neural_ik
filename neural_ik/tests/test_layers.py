from unittest import TestCase
from neural_ik.layers import WeightedSum
import tensorflow as tf


class TestWeightedSum(TestCase):
    def test_call_batch16(self):
        dx = tf.random.uniform(shape=(16, 7))
        w = tf.random.uniform(shape=(16, 7))
        x = tf.random.uniform(shape=(16, 7))
        eager = WeightedSum()([dx, w, x])
        self.assertEqual(eager.shape, (16, 7))

