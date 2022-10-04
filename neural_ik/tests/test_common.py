from unittest import TestCase
from neural_ik.models.common import linear_identity
from keras.layers import Input
from keras.models import Model

import tensorflow as tf


class Test(TestCase):
    def test_linear_identity(self):
        inp = Input(50)
        ident = linear_identity(inp)
        model = Model(inputs=inp, outputs=ident, name="test")

        test_x = tf.random.uniform(shape=(3, 50))
        self.assertTrue(tf.equal(test_x, model(test_x)).numpy().all())

