import tempfile
from unittest import TestCase

from keras.saving.save import load_model

from neural_ik.models.residual_solver_dnn import residual_solver_dnn


class Test(TestCase):
    def test_residual_solver_dnn_serialization(self):
        batch_size = 1
        blocks_count = 16
        model = residual_solver_dnn("omnipointer_robot", batch_size,
                                    blocks_count=blocks_count)

        with tempfile.NamedTemporaryFile() as tmp:
            model.compile()
            model.save(tmp.name + '.hdf5')
            model_loaded = load_model(tmp.name + '.hdf5')

