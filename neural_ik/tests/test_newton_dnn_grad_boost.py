import tempfile
from unittest import TestCase

from keras.saving.save import load_model

from neural_ik.models.newton_dnn_grad_boost import newton_dnn_grad_boost


class Test(TestCase):
    def test_newton_dnn_grad_boost_serialization(self):
        batch_size = 1
        blocks_count = 16
        model = newton_dnn_grad_boost("omnipointer_robot", batch_size,
                                      blocks_count=blocks_count)

        with tempfile.NamedTemporaryFile() as tmp:
            model.compile()
            model.save(tmp.name + '.hdf5')
            model_loaded = load_model(tmp.name + '.hdf5')

