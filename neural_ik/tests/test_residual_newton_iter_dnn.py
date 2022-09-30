import tempfile
from unittest import TestCase
from neural_ik.models.residual_newton_iter_dnn import residual_newton_iter_dnn
from tf_kinematics import kinematic_models
import tensorflow as tf


class Test(TestCase):
    def test_residual_newton_iter_dnn_run_forward(self):
        batch_size = 2
        blocks_count = 11
        kin = kinematic_models.omnipointer_robot(batch_size)
        model_dist, model_ik = residual_newton_iter_dnn(kin, blocks_count=blocks_count, corrector_units=10)
        model_dist.compile()
        thetas = tf.ones(shape=(batch_size, kin.dof))
        iso_goals = tf.stack([tf.eye(4, 4)]*batch_size)

        res = model_dist.predict_on_batch([thetas, iso_goals])
        self.assertEqual(res.size, batch_size*blocks_count)

        res = model_ik.predict_on_batch([thetas, iso_goals])
        self.assertEqual(len(res), blocks_count)
        self.assertTrue(all(map(lambda out: out.shape == (2, 4), res)))

    def test_residual_save(self):
        batch_size = 2
        blocks_count = 11
        kin = kinematic_models.omnipointer_robot(batch_size)
        model_dist, model_ik = residual_newton_iter_dnn(kin, blocks_count=blocks_count, corrector_units=10)

        with tempfile.TemporaryDirectory() as tmp:
            model_dist.save(tmp.name)
        with tempfile.TemporaryDirectory() as tmp:
            model_ik.save(tmp.name)
