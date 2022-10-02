import tempfile
from unittest import TestCase
from neural_ik.models.residual_fk_dnn import residual_fk_dnn
from neural_ik.losses import PowWeightedMSE
from tf_kinematics.iso_layers import IsometryCompact, IsometryInverse, CompactL2Norm, CompactL1Norm
from tf_kinematics.kin_layers import ForwardKinematics
from tf_kinematics.kinematic_models import load as load_kin
from keras.models import load_model
import tensorflow as tf


class Test(TestCase):
    def test_residual_fk_dnn(self):
        batch_size = 2
        blocks_count = 2
        kin = load_kin("kuka_robot", batch_size)
        model_dist, model_ik = residual_fk_dnn("kuka_robot", batch_size,
                                               blocks_count=blocks_count, corrector_units=2)
        model_dist.compile()
        thetas = tf.ones(shape=(batch_size, kin.dof))
        iso_goals = tf.stack([tf.eye(4, 4)] * batch_size)

        res = model_dist.predict_on_batch([thetas, iso_goals])
        self.assertEqual(res.size, batch_size * blocks_count)

        res = model_ik.predict_on_batch([thetas, iso_goals])
        self.assertEqual(len(res), blocks_count)
        self.assertTrue(all(map(lambda out: out.shape == (batch_size, kin.dof), res)))

    def test_residual_save(self):
        batch_size = 2
        blocks_count = 2
        model_dist, model_ik = residual_fk_dnn("omnipointer_robot", batch_size,
                                               blocks_count=blocks_count, corrector_units=2)

        with tempfile.NamedTemporaryFile() as tmp:
            model_dist.compile(loss=PowWeightedMSE)
            model_dist.save(tmp.name + '.h5', include_optimizer=False)
            load_model(tmp.name + '.h5')

        with tempfile.NamedTemporaryFile() as tmp:
            model_ik.save(tmp.name + '.h5')
            load_model(tmp.name + '.h5')
